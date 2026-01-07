"""
LoRA Fine-tuning Script for Qwen2-Audio on AfriSpeech Dataset

This script performs Parameter-Efficient Fine-Tuning (PEFT) using LoRA
on Qwen2-Audio for Automatic Speech Recognition (ASR) with the AfriSpeech-200 dataset.

Usage:
    # Single GPU training
    python train_lora_afrispeech.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./afrispeech_data \
        --output_dir ./outputs/qwen2audio_lora_afrispeech

    # Multi-GPU training with DeepSpeed
    torchrun --nproc_per_node=8 train_lora_afrispeech.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./afrispeech_data \
        --output_dir ./outputs/qwen2audio_lora_afrispeech \
        --deepspeed configs/zero2.json

    # Load directly from HuggingFace
    python train_lora_afrispeech.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --from_hf \
        --output_dir ./outputs/qwen2audio_lora_afrispeech

Requirements:
    pip install transformers>=4.40.0 peft>=0.10.0 trl accelerate deepspeed
    pip install torchaudio librosa datasets huggingface_hub
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    prepare_model_for_kbit_training,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    return waveform[0].numpy()


class AfriSpeechLoRADataset(Dataset):
    """
    AfriSpeech dataset for LoRA fine-tuning.

    Expected directory structure (from download_afrispeech.py):
        afrispeech_data/
        ├── train/
        │   ├── metadata.json
        │   └── audio/
        │       ├── xxx.wav
        │       └── ...
        ├── dev/
        └── test/
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        accent_filter: Optional[str] = None,
        num_examples: Optional[int] = None,
    ):
        """
        Initialize AfriSpeech dataset for LoRA fine-tuning.

        Args:
            data_dir: Path to afrispeech_data directory
            split: Dataset split ("train", "dev", "test")
            sample_rate: Audio sample rate (default: 16000)
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            accent_filter: Filter for specific accent (e.g., "yoruba")
            num_examples: Maximum number of examples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate

        # Load metadata
        metadata_path = self.data_dir / split / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Filter samples
        self.samples = []
        skipped = {"duration": 0, "missing": 0, "accent": 0}

        for entry in metadata:
            if num_examples and len(self.samples) >= num_examples:
                break

            duration = entry.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                skipped["duration"] += 1
                continue

            if accent_filter and entry.get("accent", "").lower() != accent_filter.lower():
                skipped["accent"] += 1
                continue

            audio_path = entry.get("audio_path", "")
            if not audio_path:
                skipped["missing"] += 1
                continue

            full_audio_path = self.data_dir / audio_path
            if not full_audio_path.exists():
                skipped["missing"] += 1
                continue

            self.samples.append({
                "audio_path": str(full_audio_path),
                "transcript": entry.get("transcript", ""),
                "accent": entry.get("accent", ""),
                "duration": duration,
            })

        logger.info(
            f"AfriSpeech {split}: loaded {len(self.samples)} samples "
            f"(skipped: {skipped['duration']} duration, {skipped['missing']} missing, {skipped['accent']} accent)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio with retry on failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                audio = load_audio(sample["audio_path"], self.sample_rate)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return a fallback sample
                    fallback_idx = (idx + 1) % len(self.samples)
                    return self.__getitem__(fallback_idx)
                logger.warning(f"Failed to load {sample['audio_path']}: {e}, retrying...")

        return {
            "audio": audio,
            "transcript": sample["transcript"],
            "accent": sample["accent"],
        }


class AfriSpeechLoRADatasetHF(Dataset):
    """Load AfriSpeech directly from HuggingFace datasets."""

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        accent_filter: Optional[str] = None,
        num_examples: Optional[int] = None,
    ):
        from datasets import load_dataset

        self.sample_rate = sample_rate

        logger.info(f"Loading AfriSpeech {split} from HuggingFace...")
        dataset = load_dataset("intronhealth/afrispeech-200", split=split)

        self.samples = []
        for sample in dataset:
            if num_examples and len(self.samples) >= num_examples:
                break

            duration = sample.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            if accent_filter and sample.get("accent", "").lower() != accent_filter.lower():
                continue

            self.samples.append({
                "audio": sample["audio"],
                "transcript": sample.get("transcript", ""),
                "accent": sample.get("accent", ""),
            })

        logger.info(f"AfriSpeech {split}: loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        orig_sr = audio_data["sampling_rate"]

        # Resample if needed
        if orig_sr != self.sample_rate:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0).float()
            audio_array = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )(audio_tensor)[0].numpy()

        return {
            "audio": audio_array,
            "transcript": sample["transcript"],
            "accent": sample["accent"],
        }


# ============================================================================
# Data Collator
# ============================================================================

class Qwen2AudioDataCollator:
    """
    Data collator for Qwen2-Audio LoRA fine-tuning.

    Handles audio processing and creates proper input format for the model.
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
        # Build conversations for each sample
        conversations = []
        audios = []
        transcripts = []

        for feature in features:
            # Build conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": ""},
                        {"type": "text", "text": "Transcribe the following audio to text."},
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
            audios=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Create labels for training
        # Labels are the transcript tokens, with -100 for prompt tokens
        labels_list = []
        for i, transcript in enumerate(transcripts):
            # Tokenize the full input + transcript response
            full_response = transcript + self.processor.tokenizer.eos_token
            response_ids = self.processor.tokenizer(
                full_response,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]

            # Create label tensor: -100 for prompt, then response tokens
            prompt_length = inputs.input_ids[i].shape[0]

            # We need to extend input_ids with response tokens for training
            labels = torch.full((prompt_length,), -100, dtype=torch.long)
            labels = torch.cat([labels, response_ids])
            labels_list.append(labels)

        # Extend input_ids with response tokens
        extended_input_ids = []
        extended_attention_mask = []

        for i, transcript in enumerate(transcripts):
            full_response = transcript + self.processor.tokenizer.eos_token
            response_ids = self.processor.tokenizer(
                full_response,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]

            # Concatenate prompt with response
            full_ids = torch.cat([inputs.input_ids[i], response_ids])
            full_mask = torch.cat([
                inputs.attention_mask[i],
                torch.ones(len(response_ids), dtype=torch.long)
            ])

            extended_input_ids.append(full_ids)
            extended_attention_mask.append(full_mask)

        # Pad to same length
        max_len = max(len(ids) for ids in extended_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_token_id = self.processor.tokenizer.pad_token_id or 0

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

class Qwen2AudioLoRATrainer(Trainer):
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
        default="./afrispeech_data",
        metadata={"help": "Directory containing downloaded AfriSpeech data"},
    )
    from_hf: bool = field(
        default=False,
        metadata={"help": "Load dataset directly from HuggingFace instead of local files"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Training split name"},
    )
    eval_split: Optional[str] = field(
        default="dev",
        metadata={"help": "Evaluation split name (set to None to skip evaluation)"},
    )
    accent: Optional[str] = field(
        default=None,
        metadata={"help": "Filter for specific accent (e.g., yoruba, igbo)"},
    )
    max_duration: float = field(
        default=30.0,
        metadata={"help": "Maximum audio duration in seconds"},
    )
    min_duration: float = field(
        default=0.5,
        metadata={"help": "Minimum audio duration in seconds"},
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
        train_dataset = AfriSpeechLoRADatasetHF(
            split=data_args.train_split,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            accent_filter=data_args.accent,
            num_examples=data_args.max_train_samples,
        )

        eval_dataset = None
        if data_args.eval_split:
            eval_dataset = AfriSpeechLoRADatasetHF(
                split=data_args.eval_split,
                sample_rate=data_args.sample_rate,
                max_duration=data_args.max_duration,
                min_duration=data_args.min_duration,
                accent_filter=data_args.accent,
                num_examples=data_args.max_eval_samples,
            )
    else:
        train_dataset = AfriSpeechLoRADataset(
            data_dir=data_args.data_dir,
            split=data_args.train_split,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            accent_filter=data_args.accent,
            num_examples=data_args.max_train_samples,
        )

        eval_dataset = None
        if data_args.eval_split:
            try:
                eval_dataset = AfriSpeechLoRADataset(
                    data_dir=data_args.data_dir,
                    split=data_args.eval_split,
                    sample_rate=data_args.sample_rate,
                    max_duration=data_args.max_duration,
                    min_duration=data_args.min_duration,
                    accent_filter=data_args.accent,
                    num_examples=data_args.max_eval_samples,
                )
            except FileNotFoundError:
                logger.warning(f"Eval split '{data_args.eval_split}' not found, skipping evaluation")
                eval_dataset = None

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # Create data collator
    data_collator = Qwen2AudioDataCollator(
        processor=processor,
        sample_rate=data_args.sample_rate,
    )

    # Create trainer
    trainer = Qwen2AudioLoRATrainer(
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
