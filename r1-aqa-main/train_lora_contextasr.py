"""
LoRA Fine-tuning Script for Qwen2-Audio on ContextASR-Bench Dataset

This script performs Parameter-Efficient Fine-Tuning (PEFT) using LoRA
on Qwen2-Audio for Contextual Automatic Speech Recognition (ASR) using
the ContextASR-Bench dataset.

The key difference from standard ASR is the use of entity hints in the prompt:
"You are a speech transcription system. Output ONLY the exact words spoken.
The following names/terms may appear: {entity_str}. Use correct spelling for these terms."

Usage:
    # Single GPU training
    python train_lora_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --output_dir ./outputs/qwen2audio_lora_contextasr \
        --num_train_epochs 4

    # Multi-GPU training with DeepSpeed
    torchrun --nproc_per_node=8 train_lora_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --output_dir ./outputs/qwen2audio_lora_contextasr \
        --deepspeed configs/lora_zero2.json \
        --num_train_epochs 4

    # Train on specific config/language
    python train_lora_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --config ContextASR-Dialogue \
        --language English \
        --output_dir ./outputs/qwen2audio_lora_dialogue_en \
        --num_train_epochs 4

    # Stream directly from HuggingFace
    python train_lora_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --from_hf \
        --output_dir ./outputs/qwen2audio_lora_contextasr \
        --num_train_epochs 4

    # Limit number of training/eval examples (useful for debugging)
    python train_lora_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --output_dir ./outputs/qwen2audio_lora_contextasr \
        --max_train_samples 1000 \
        --max_eval_samples 100 \
        --num_train_epochs 4

Requirements:
    pip install -r requirements_lora.txt
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Context ASR Prompt Template
# ============================================================================

CONTEXT_ASR_SYSTEM_PROMPT = (
    "You are a speech transcription system. Output ONLY the exact words spoken. "
    "The following names/terms may appear: {entity_str}. Use correct spelling for these terms."
)

CONTEXT_ASR_USER_PROMPT = "Transcribe the speech word-for-word:"

# Fallback for samples without entities
NO_CONTEXT_SYSTEM_PROMPT = (
    "You are a speech transcription system. Output ONLY the exact words spoken in the audio, nothing else."
)


# ============================================================================
# Dataset Classes
# ============================================================================

def load_audio(audio_path: str, target_sr: int = 16000):
    """Load and resample audio file to target sample rate."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(waveform)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform[0].numpy()


class ContextASRLoRADataset(Dataset):
    """
    ContextASR-Bench dataset for LoRA fine-tuning with audio chunking.

    Expected directory structure (from download_contextasr.py):
        contextasr_data/
        ├── ContextASR-Dialogue/
        │   ├── English/
        │   │   ├── metadata.json
        │   │   └── audio/
        │   └── Mandarin/
        └── ContextASR-Speech/
            ├── English/
            └── Mandarin/

    Each metadata.json entry has:
        - uniq_id: unique identifier
        - text: ground truth transcription
        - entity_list: list of entities/terms that may appear
        - audio_path: relative path to audio file
        - duration: audio duration in seconds
        - dialogue: (optional) list of dialogue turns with timing
    """

    def __init__(
        self,
        data_dir: str,
        config: str = "ContextASR-Dialogue",
        language: str = "English",
        sample_rate: int = 16000,
        max_duration: float = 180.0,  # ContextASR has long audio (99-253s)
        min_duration: float = 0.5,
        max_audio_chunk: float = 30.0,  # Chunk long audio for model input
        num_examples: Optional[int] = None,
        include_no_context: bool = False,
    ):
        """
        Initialize ContextASR dataset for LoRA fine-tuning.

        Args:
            data_dir: Path to contextasr_data directory
            config: Dataset config ("ContextASR-Dialogue" or "ContextASR-Speech")
            language: Language ("English" or "Mandarin")
            sample_rate: Audio sample rate (default: 16000)
            max_duration: Maximum audio duration in seconds (skip longer)
            min_duration: Minimum audio duration in seconds (skip shorter)
            max_audio_chunk: Maximum chunk size for model input (30s default)
            num_examples: Maximum number of examples to load
            include_no_context: Also include samples without entity context for robustness
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.language = language
        self.sample_rate = sample_rate
        self.max_audio_chunk = max_audio_chunk
        self.include_no_context = include_no_context

        # Load metadata
        metadata_path = self.data_dir / config / language / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Filter samples and create chunks
        self.samples = []
        skipped = {"duration": 0, "missing": 0, "no_entities": 0}
        total_chunks = 0

        for entry in metadata:
            if num_examples and len(self.samples) >= num_examples:
                break

            duration = entry.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                skipped["duration"] += 1
                continue

            audio_path = entry.get("audio_path", "")
            if not audio_path:
                skipped["missing"] += 1
                continue

            # Handle different audio path formats
            possible_paths = [
                self.data_dir / audio_path,
                self.data_dir / config / language / audio_path,
                self.data_dir / config / language / "audio" / Path(audio_path).name,
            ]

            full_audio_path = None
            for p in possible_paths:
                if p.exists():
                    full_audio_path = p
                    break

            if full_audio_path is None:
                skipped["missing"] += 1
                continue

            entity_list = entry.get("entity_list", [])

            # Skip samples without entities unless include_no_context is True
            if not entity_list and not include_no_context:
                skipped["no_entities"] += 1
                continue

            # Create chunks from dialogue timing (if available)
            dialogue = entry.get("dialogue", [])
            full_text = entry.get("text", "")

            if dialogue and len(dialogue) > 0:
                # Group dialogue turns into chunks of max_audio_chunk seconds
                chunks = self._create_dialogue_chunks(dialogue, max_audio_chunk)
                for chunk in chunks:
                    if num_examples and len(self.samples) >= num_examples:
                        break
                    self.samples.append({
                        "audio_path": str(full_audio_path),
                        "transcript": chunk["text"],
                        "entity_list": entity_list,
                        "start_time": chunk["start"],
                        "end_time": chunk["end"],
                        "duration": chunk["end"] - chunk["start"],
                        "uniq_id": f"{entry.get('uniq_id', '')}_{chunk['chunk_id']}",
                    })
                    total_chunks += 1
            else:
                # No dialogue timing - create chunks with proportionally estimated transcript
                chunk_id = 0
                current_time = 0.0
                while current_time < duration:
                    if num_examples and len(self.samples) >= num_examples:
                        break

                    chunk_end = min(current_time + max_audio_chunk, duration)
                    chunk_duration = chunk_end - current_time

                    # Proportionally estimate transcript for this chunk
                    start_ratio = current_time / duration
                    end_ratio = chunk_end / duration
                    start_char = int(len(full_text) * start_ratio)
                    end_char = int(len(full_text) * end_ratio)
                    chunk_text = full_text[start_char:end_char].strip()

                    if chunk_text:  # Only add if there's text
                        self.samples.append({
                            "audio_path": str(full_audio_path),
                            "transcript": chunk_text,
                            "entity_list": entity_list,
                            "start_time": current_time,
                            "end_time": chunk_end,
                            "duration": chunk_duration,
                            "uniq_id": f"{entry.get('uniq_id', '')}_{chunk_id}",
                        })
                        total_chunks += 1

                    current_time = chunk_end
                    chunk_id += 1

        logger.info(
            f"ContextASR {config}/{language}: loaded {len(self.samples)} samples "
            f"({total_chunks} chunks, skipped: {skipped['duration']} duration, "
            f"{skipped['missing']} missing, {skipped['no_entities']} no_entities)"
        )

    def _create_dialogue_chunks(self, dialogue: List[dict], max_chunk_duration: float) -> List[dict]:
        """Create chunks from dialogue turns, grouping turns to fit within max duration."""
        chunks = []
        current_chunk = {"start": 0.0, "end": 0.0, "text": "", "chunk_id": 0}

        for turn in dialogue:
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            turn_text = turn.get("text", "")

            # If this turn would exceed max duration, save current chunk and start new one
            if current_chunk["text"] and (turn_end - current_chunk["start"]) > max_chunk_duration:
                chunks.append(current_chunk)
                current_chunk = {
                    "start": turn_start,
                    "end": turn_end,
                    "text": turn_text,
                    "chunk_id": len(chunks),
                }
            else:
                # Add turn to current chunk
                if not current_chunk["text"]:
                    current_chunk["start"] = turn_start
                current_chunk["end"] = turn_end
                current_chunk["text"] = (current_chunk["text"] + " " + turn_text).strip()

        # Add final chunk
        if current_chunk["text"]:
            chunks.append(current_chunk)

        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio with retry on failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                full_audio = load_audio(sample["audio_path"], self.sample_rate)

                # Extract chunk based on timing
                start_sample = int(sample["start_time"] * self.sample_rate)
                end_sample = int(sample["end_time"] * self.sample_rate)
                audio = full_audio[start_sample:end_sample]

                # Ensure audio is not empty
                if len(audio) < self.sample_rate * 0.1:  # Less than 0.1 second
                    raise ValueError(f"Audio chunk too short: {len(audio)} samples")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return a fallback sample
                    fallback_idx = (idx + 1) % len(self.samples)
                    return self.__getitem__(fallback_idx)
                logger.warning(f"Failed to load {sample['audio_path']}: {e}, retrying...")

        # Build entity string for context
        entity_list = sample["entity_list"]
        if entity_list:
            entity_str = ", ".join(entity_list)
            system_prompt = CONTEXT_ASR_SYSTEM_PROMPT.format(entity_str=entity_str)
        else:
            system_prompt = NO_CONTEXT_SYSTEM_PROMPT

        return {
            "audio": audio,
            "transcript": sample["transcript"],
            "entity_list": entity_list,
            "system_prompt": system_prompt,
        }


class ContextASRLoRADatasetHF(Dataset):
    """Load ContextASR-Bench directly from HuggingFace datasets."""

    def __init__(
        self,
        config: str = "ContextASR-Dialogue",
        language: str = "English",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        num_examples: Optional[int] = None,
        include_no_context: bool = False,
    ):
        from datasets import load_dataset
        import librosa

        self.sample_rate = sample_rate
        self.include_no_context = include_no_context

        logger.info(f"Loading ContextASR {config}/{language} from HuggingFace...")
        dataset = load_dataset(
            "MrSupW/ContextASR-Bench",
            config,
            split=language,
            trust_remote_code=True,
        )

        self.samples = []
        for sample in dataset:
            if num_examples and len(self.samples) >= num_examples:
                break

            duration = sample.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            entity_list = list(sample.get("entity_list", []))
            if not entity_list and not include_no_context:
                continue

            # Get audio data
            audio_data = sample.get("audio", {})
            if not audio_data:
                continue

            self.samples.append({
                "audio": audio_data,
                "transcript": sample.get("text", ""),
                "entity_list": entity_list,
            })

        logger.info(f"ContextASR {config}/{language}: loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio_data = sample["audio"]

        # Handle different audio formats from HF datasets
        if isinstance(audio_data, dict):
            if "array" in audio_data:
                audio_array = np.array(audio_data["array"])
                orig_sr = audio_data.get("sampling_rate", 16000)
            elif "path" in audio_data:
                import librosa
                audio_array, orig_sr = librosa.load(audio_data["path"], sr=None)
            else:
                raise ValueError(f"Unknown audio format: {audio_data.keys()}")
        else:
            audio_array = np.array(audio_data)
            orig_sr = 16000

        # Resample if needed
        if orig_sr != self.sample_rate:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0).float()
            audio_array = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )(audio_tensor)[0].numpy()

        # Build entity string for context
        entity_list = sample["entity_list"]
        if entity_list:
            entity_str = ", ".join(entity_list)
            system_prompt = CONTEXT_ASR_SYSTEM_PROMPT.format(entity_str=entity_str)
        else:
            system_prompt = NO_CONTEXT_SYSTEM_PROMPT

        return {
            "audio": audio_array,
            "transcript": sample["transcript"],
            "entity_list": entity_list,
            "system_prompt": system_prompt,
        }


# ============================================================================
# Data Collator
# ============================================================================

class ContextASRDataCollator:
    """
    Data collator for Context ASR LoRA fine-tuning.

    Handles audio processing and creates proper input format with entity context.
    """

    def __init__(
        self,
        processor,
        max_length: int = 512,
        sample_rate: int = 16000,
    ):
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build conversations for each sample with context
        conversations = []
        audios = []
        transcripts = []

        for feature in features:
            system_prompt = feature["system_prompt"]

            # Build conversation format with system prompt containing entities
            conversation = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": feature["audio"]},
                        {"type": "text", "text": CONTEXT_ASR_USER_PROMPT},
                    ],
                },
            ]
            conversations.append(conversation)
            audios.append(feature["audio"])
            transcripts.append(feature["transcript"])

        # Process inputs using the Qwen2-Audio processor
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        # Process with audio
        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Create labels for training
        labels_list = []
        extended_input_ids = []
        extended_attention_mask = []

        for i, transcript in enumerate(transcripts):
            # Tokenize the transcript response
            full_response = transcript + self.processor.tokenizer.eos_token
            response_ids = self.processor.tokenizer(
                full_response,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]

            # Get prompt length
            prompt_length = inputs.input_ids[i].shape[0]

            # Create labels: -100 for prompt tokens, then response tokens
            labels = torch.full((prompt_length,), -100, dtype=torch.long)
            labels = torch.cat([labels, response_ids])

            # Concatenate prompt with response
            full_ids = torch.cat([inputs.input_ids[i], response_ids])
            full_mask = torch.cat([
                inputs.attention_mask[i],
                torch.ones(len(response_ids), dtype=torch.long)
            ])

            labels_list.append(labels)
            extended_input_ids.append(full_ids)
            extended_attention_mask.append(full_mask)

        # Pad to same length
        max_len = max(len(ids) for ids in extended_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(extended_input_ids)):
            pad_len = max_len - len(extended_input_ids[i])

            padded_input_ids.append(
                torch.cat([extended_input_ids[i], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            padded_attention_mask.append(
                torch.cat([extended_attention_mask[i], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([labels_list[i], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
        }

        return batch


# ============================================================================
# Custom Trainer for Qwen2-Audio
# ============================================================================

class Qwen2AudioContextASRTrainer(Trainer):
    """Custom trainer that handles Qwen2-Audio's multi-modal inputs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with audio features."""
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Training Arguments
# ============================================================================

@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (float16, bfloat16, float32)"},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation (flash_attention_2, sdpa, eager)"},
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    data_dir: Optional[str] = field(
        default="./contextasr_data",
        metadata={"help": "Directory containing downloaded ContextASR-Bench data"},
    )
    from_hf: bool = field(
        default=False,
        metadata={"help": "Load dataset directly from HuggingFace instead of local files"},
    )
    config: str = field(
        default="ContextASR-Dialogue",
        metadata={"help": "Dataset config (ContextASR-Dialogue or ContextASR-Speech)"},
    )
    language: str = field(
        default="English",
        metadata={"help": "Language (English or Mandarin)"},
    )
    max_duration: float = field(
        default=180.0,
        metadata={"help": "Maximum audio duration in seconds (ContextASR has long audio 99-253s)"},
    )
    min_duration: float = field(
        default=0.5,
        metadata={"help": "Minimum audio duration in seconds"},
    )
    max_audio_chunk: float = field(
        default=30.0,
        metadata={"help": "Maximum chunk size for model input (chunks long audio)"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples"},
    )
    sample_rate: int = field(
        default=16000,
        metadata={"help": "Audio sample rate"},
    )
    include_no_context: bool = field(
        default=False,
        metadata={"help": "Include samples without entity context for robustness"},
    )
    eval_language: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation language (defaults to same as training language)"},
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""

    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)"},
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha parameter (scaling)"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"},
    )
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"},
    )
    freeze_audio_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the audio encoder during training"},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use RSLoRA (rank-stabilized LoRA)"},
    )


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Disable removal of unused columns (we need custom keys like system_prompt, audio, etc.)
    training_args.remove_unused_columns = False

    # Setup logging
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"LoRA arguments: {lora_args}")
    logger.info(f"Training arguments: {training_args}")

    # Determine torch dtype
    torch_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    # Load processor
    logger.info(f"Loading processor from {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_args.trust_remote_code,
    }

    # Add attention implementation if flash attention is available
    if model_args.attn_implementation:
        try:
            model_kwargs["attn_implementation"] = model_args.attn_implementation
        except Exception as e:
            logger.warning(f"Could not use {model_args.attn_implementation}: {e}")

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Freeze audio encoder if specified
    if lora_args.freeze_audio_encoder and hasattr(model, "audio_tower"):
        logger.info("Freezing audio encoder (audio_tower)")
        for param in model.audio_tower.parameters():
            param.requires_grad = False

    # Configure LoRA
    target_modules = lora_args.target_modules.split(",") if lora_args.target_modules else None

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_args.use_rslora,
    )

    logger.info(f"LoRA config: {lora_config}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Load datasets
    logger.info("Loading datasets...")

    if data_args.from_hf:
        train_dataset = ContextASRLoRADatasetHF(
            config=data_args.config,
            language=data_args.language,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            num_examples=data_args.max_train_samples,
            include_no_context=data_args.include_no_context,
        )

        eval_dataset = None
        eval_language = data_args.eval_language or data_args.language
        if eval_language:
            try:
                eval_dataset = ContextASRLoRADatasetHF(
                    config=data_args.config,
                    language=eval_language,
                    sample_rate=data_args.sample_rate,
                    max_duration=data_args.max_duration,
                    min_duration=data_args.min_duration,
                    num_examples=data_args.max_eval_samples,
                    include_no_context=data_args.include_no_context,
                )
            except Exception as e:
                logger.warning(f"Could not load eval dataset: {e}")
                eval_dataset = None
    else:
        train_dataset = ContextASRLoRADataset(
            data_dir=data_args.data_dir,
            config=data_args.config,
            language=data_args.language,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            max_audio_chunk=data_args.max_audio_chunk,
            num_examples=data_args.max_train_samples,
            include_no_context=data_args.include_no_context,
        )

        eval_dataset = None
        eval_language = data_args.eval_language or data_args.language
        if eval_language:
            try:
                eval_dataset = ContextASRLoRADataset(
                    data_dir=data_args.data_dir,
                    config=data_args.config,
                    language=eval_language,
                    sample_rate=data_args.sample_rate,
                    max_duration=data_args.max_duration,
                    min_duration=data_args.min_duration,
                    max_audio_chunk=data_args.max_audio_chunk,
                    num_examples=data_args.max_eval_samples,
                    include_no_context=data_args.include_no_context,
                )
            except FileNotFoundError:
                logger.warning(f"Eval data for '{eval_language}' not found, skipping evaluation")
                eval_dataset = None

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Create data collator
    data_collator = ContextASRDataCollator(
        processor=processor,
        sample_rate=data_args.sample_rate,
    )

    # Create trainer
    trainer = Qwen2AudioContextASRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate
    if eval_dataset:
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
