"""
LoRA Fine-tuning Script for Qwen2-Audio on CS-FLEURS Code-Switched Speech

This script performs Parameter-Efficient Fine-Tuning (PEFT) using LoRA
on Qwen2-Audio for code-switched Automatic Speech Recognition (ASR) using
the CS-FLEURS dataset.

CS-FLEURS contains 113 unique code-switched language pairs across 52 languages
with 300 hours of speech data.

Usage:
    # Single GPU training
    python train_lora_csfleurs.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./csfleurs_data \
        --output_dir ./outputs/qwen2audio_lora_csfleurs \
        --num_train_epochs 4

    # Multi-GPU training with DeepSpeed
    torchrun --nproc_per_node=8 train_lora_csfleurs.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./csfleurs_data \
        --output_dir ./outputs/qwen2audio_lora_csfleurs \
        --deepspeed configs/lora_zero2.json \
        --num_train_epochs 4

    # Train on specific subset/language pair
    python train_lora_csfleurs.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./csfleurs_data \
        --subset xtts_train \
        --language_pair ara-eng \
        --output_dir ./outputs/qwen2audio_lora_csfleurs_ara \
        --num_train_epochs 4

Requirements:
    pip install peft torchaudio datasets
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# CS-FLEURS Prompt Templates
# ============================================================================

# Language code to name mapping
LANG_CODE_TO_NAME = {
    "ara": "Arabic", "cmn": "Chinese", "zho": "Chinese", "hin": "Hindi",
    "spa": "Spanish", "fra": "French", "deu": "German", "por": "Portuguese",
    "rus": "Russian", "jpn": "Japanese", "kor": "Korean", "vie": "Vietnamese",
    "tha": "Thai", "ind": "Indonesian", "tur": "Turkish", "pol": "Polish",
    "nld": "Dutch", "ita": "Italian", "eng": "English",
}

# Subset name to directory path mapping
SUBSET_TO_PATH = {
    "xtts_train": "xtts/train",
    "xtts_test1": "xtts/test1",
    "xtts_test2": "xtts/test2",
    "read_test": "read/test",
    "mms_test": "mms/test",
}

# Languages that don't use spaces between words
SPACELESS_LANGUAGES = {"jpn", "cmn", "zho", "tha", "yue"}


def _get_language_name(lang_code: str) -> str:
    """Convert language code (e.g., 'ara-eng') to readable name."""
    if "-" in lang_code:
        primary = lang_code.split("-")[0]
    elif "_" in lang_code:
        primary = lang_code.split("_")[0]
    else:
        primary = lang_code
    return LANG_CODE_TO_NAME.get(primary, primary.capitalize())


def _remove_spaces_for_language(text: str, lang_code: str) -> str:
    """Remove spaces from text for languages that don't use word spacing."""
    if lang_code not in SPACELESS_LANGUAGES:
        return text

    cjk = '[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0e00-\u0e7f\u3000-\u303f\uff00-\uffef]'
    latin = '[a-zA-Z\u00c0-\u00ff]'

    result = text
    prev = None
    while prev != result:
        prev = result
        result = re.sub(f'({cjk}) +({cjk})', r'\1\2', result)
        result = re.sub(f'({cjk}) +({latin})', r'\1\2', result)
        result = re.sub(f'({latin}) +({cjk})', r'\1\2', result)
        result = re.sub(f'({cjk}) +([,.!?;:，。！？；：、])', r'\1\2', result)
        result = re.sub(f'([,.!?;:，。！？；：、]) +({cjk})', r'\1\2', result)

    return result


def _extract_code_switch_entities(text: str, language: str = None) -> tuple:
    """
    Extract code-switched entities from CS-FLEURS text.
    The ** markers annotate English code-switched portions.
    """
    import unicodedata

    def is_punctuation_only(s):
        return all(unicodedata.category(c).startswith('P') or c.isspace() for c in s)

    def strip_edge_punctuation(s):
        while s and unicodedata.category(s[0]).startswith('P'):
            s = s[1:]
        while s and unicodedata.category(s[-1]).startswith('P'):
            s = s[:-1]
        return s

    raw_entities = re.findall(r'\*\*([^*]+)\*\*', text)
    entities = []
    for e in raw_entities:
        e = e.strip()
        e = strip_edge_punctuation(e)
        if e and not is_punctuation_only(e):
            entities.append(e)

    if language and "-" in language:
        primary_lang = language.split("-")[0]
    elif language and "_" in language:
        primary_lang = language.split("_")[0]
    else:
        primary_lang = language or ""

    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    result_parts = []

    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            english_content = part[2:-2]
            result_parts.append(english_content)
        else:
            processed = _remove_spaces_for_language(part, primary_lang)
            result_parts.append(processed)

    clean_text = ''.join(result_parts)
    return clean_text, entities


CSFLEURS_SYSTEM_PROMPT = (
    "You are a speech transcription system for code-switched speech. "
    "The audio contains speech mixing {language} and English. "
    "Output ONLY the exact words spoken, preserving the language switches."
)

CSFLEURS_USER_PROMPT = "Transcribe the speech word-for-word:"


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
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform[0].numpy()


class CSFleursLoRADataset(Dataset):
    """
    CS-FLEURS dataset for LoRA fine-tuning.

    Expected local directory structure:
        csfleurs_data/
        ├── xtts/
        │   ├── train/
        │   │   ├── metadata.jsonl
        │   │   └── audio/
        │   │       ├── cs_ara_eng_n1_resample/
        │   │       ├── cs_cmn_eng_n1_resample/
        │   │       └── ...
        │   ├── test1/
        │   └── test2/
        ├── read/
        │   └── test/
        └── mms/
            └── test/
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = "xtts_train",
        language_pair: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        num_examples: Optional[int] = None,
        data_seed: int = 42,
        filter_unsupported: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.language_pair = language_pair
        self.sample_rate = sample_rate

        # Get subset path
        subset_path = SUBSET_TO_PATH.get(subset, subset)
        base_path = self.data_dir / subset_path

        if not base_path.exists():
            raise FileNotFoundError(f"Subset path not found: {base_path}")

        all_samples = []
        skipped = {"duration": 0, "missing": 0, "language": 0}

        # Load metadata - prefer .marked.jsonl (with ** markers) over original
        # (matches GRPO dataset behavior)
        marked_path = base_path / "metadata.marked.jsonl"
        metadata_path = base_path / "metadata.jsonl"
        if marked_path.exists():
            metadata_path = marked_path
            logger.info(f"Using marked metadata (with ** markers): {marked_path}")
        elif not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                entry = json.loads(line)

                # Filter by language pair if specified
                lang_code = entry.get("language", "")
                if language_pair and lang_code.lower() != language_pair.lower():
                    skipped["language"] += 1
                    continue

                # Filter unsupported languages (matches GRPO dataset behavior)
                if filter_unsupported:
                    UNSUPPORTED_LANGS = {"hin", "tur", "pol", "nld", "hun", "ces", "vie", "tha", "ind", "slk", "tel"}
                    lang_parts = lang_code.lower().split("-") if "-" in lang_code else [lang_code.lower()]
                    if any(lp in UNSUPPORTED_LANGS for lp in lang_parts):
                        skipped["language"] += 1
                        continue

                duration = entry.get("duration", 0)
                if duration < min_duration or duration > max_duration:
                    skipped["duration"] += 1
                    continue

                # Get audio file path
                audio_file = entry.get("file_name") or entry.get("audio_path", "")
                if not audio_file:
                    skipped["missing"] += 1
                    continue

                # Resolve audio path
                audio_path = base_path / audio_file
                if not audio_path.exists():
                    # Try without double slashes
                    audio_file_clean = audio_file.replace("//", "/")
                    audio_path = base_path / audio_file_clean

                if not audio_path.exists():
                    skipped["missing"] += 1
                    continue

                # Extract text and entities
                raw_text = entry.get("text", "")
                clean_text, entities = _extract_code_switch_entities(raw_text, lang_code)

                all_samples.append({
                    "audio_path": str(audio_path),
                    "transcript": clean_text,
                    "raw_text": raw_text,
                    "language": lang_code,
                    "entity_list": entities,
                    "duration": duration,
                    "sample_id": entry.get("id", f"{lang_code}_{len(all_samples)}"),
                })

        logger.info(
            f"CS-FLEURS {subset}: found {len(all_samples)} valid samples "
            f"(skipped: {skipped['duration']} duration, {skipped['missing']} missing, "
            f"{skipped['language']} language filter)"
        )

        # Stratified sampling by language pair (matches GRPO dataset logic)
        import random
        from collections import defaultdict

        if num_examples and num_examples < len(all_samples):
            samples_by_lang = defaultdict(list)
            for sample in all_samples:
                samples_by_lang[sample["language"]].append(sample)

            num_langs = len(samples_by_lang)
            samples_per_lang = num_examples // num_langs
            remainder = num_examples % num_langs

            logger.info(f"Stratified sampling: {num_examples} examples across {num_langs} language pairs")
            logger.info(f"  ~{samples_per_lang} examples per language pair")

            self.samples = []
            lang_counts = {}

            for i, lang in enumerate(sorted(samples_by_lang.keys())):
                lang_samples = samples_by_lang[lang]
                n_samples = samples_per_lang + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(lang_samples))

                random.seed(data_seed)
                random.shuffle(lang_samples)
                self.samples.extend(lang_samples[:n_samples])
                lang_counts[lang] = n_samples

            random.seed(data_seed)
            random.shuffle(self.samples)

            logger.info(f"  Language distribution: {lang_counts}")
        else:
            self.samples = all_samples

        logger.info(f"CS-FLEURS {subset}: using {len(self.samples)} samples (data_seed={data_seed})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            audio = load_audio(sample["audio_path"], self.sample_rate)
        except Exception as e:
            logger.warning(f"Failed to load {sample['audio_path']}: {e}")
            fallback_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(fallback_idx)

        language_name = _get_language_name(sample["language"])
        system_prompt = CSFLEURS_SYSTEM_PROMPT.format(language=language_name)

        return {
            "audio": audio,
            "transcript": sample["transcript"],
            "raw_text": sample.get("raw_text", ""),
            "language": sample["language"],
            "entity_list": sample["entity_list"],
            "system_prompt": system_prompt,
        }


class CSFleursLoRADatasetHF(Dataset):
    """Load CS-FLEURS directly from HuggingFace datasets."""

    def __init__(
        self,
        subset: str = "xtts_train",
        language_pair: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        num_examples: Optional[int] = None,
    ):
        from datasets import load_dataset

        self.sample_rate = sample_rate

        logger.info(f"Loading CS-FLEURS {subset} from HuggingFace...")
        try:
            dataset = load_dataset("byan/cs-fleurs", subset, split="train")
        except Exception:
            dataset = load_dataset("byan/cs-fleurs", subset)
            if hasattr(dataset, 'keys'):
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]

        self.samples = []
        for sample in dataset:
            if num_examples and len(self.samples) >= num_examples:
                break

            if language_pair and sample.get("language", "").lower() != language_pair.lower():
                continue

            duration = sample.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            audio_data = sample.get("audio", {})
            if not audio_data:
                continue

            raw_text = sample.get("text", "")
            language = sample.get("language", "unknown")
            clean_text, entities = _extract_code_switch_entities(raw_text, language)

            self.samples.append({
                "audio": audio_data,
                "transcript": clean_text,
                "raw_text": raw_text,
                "language": language,
                "entity_list": entities,
            })

        logger.info(f"CS-FLEURS {subset}: loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_data = sample["audio"]

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

        if orig_sr != self.sample_rate:
            audio_tensor = torch.tensor(audio_array).unsqueeze(0).float()
            audio_array = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )(audio_tensor)[0].numpy()

        language_name = _get_language_name(sample["language"])
        system_prompt = CSFLEURS_SYSTEM_PROMPT.format(language=language_name)

        return {
            "audio": audio_array,
            "transcript": sample["transcript"],
            "raw_text": sample.get("raw_text", ""),
            "language": sample["language"],
            "entity_list": sample["entity_list"],
            "system_prompt": system_prompt,
        }


# ============================================================================
# Data Collator
# ============================================================================

class CSFleursDataCollator:
    """Data collator for CS-FLEURS LoRA fine-tuning."""

    def __init__(self, processor, max_length: int = 512, sample_rate: int = 16000):
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = []
        audios = []
        transcripts = []

        for feature in features:
            conversation = [
                {"role": "system", "content": feature["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": feature["audio"]},
                        {"type": "text", "text": CSFLEURS_USER_PROMPT},
                    ],
                },
            ]
            conversations.append(conversation)
            audios.append(feature["audio"])
            transcripts.append(feature["transcript"])

        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        labels_list = []
        extended_input_ids = []
        extended_attention_mask = []

        for i, transcript in enumerate(transcripts):
            full_response = transcript + self.processor.tokenizer.eos_token
            response_ids = self.processor.tokenizer(
                full_response,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]

            prompt_length = inputs.input_ids[i].shape[0]
            labels = torch.full((prompt_length,), -100, dtype=torch.long)
            labels = torch.cat([labels, response_ids])

            full_ids = torch.cat([inputs.input_ids[i], response_ids])
            full_mask = torch.cat([
                inputs.attention_mask[i],
                torch.ones(len(response_ids), dtype=torch.long)
            ])

            labels_list.append(labels)
            extended_input_ids.append(full_ids)
            extended_attention_mask.append(full_mask)

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
# Custom Trainer
# ============================================================================

class Qwen2AudioCSFleursTrainer(Trainer):
    """Custom trainer that handles Qwen2-Audio's multi-modal inputs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate CER and bCER on the validation set during training."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent / "src"))
        from evaluate_csfleurs import compute_cer, compute_bcer
        from utils.rewards import _remove_sp
        from tqdm import tqdm

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        model = self.model
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)

        was_training = unwrapped_model.training
        unwrapped_model.eval()

        is_gc = getattr(unwrapped_model, 'is_gradient_checkpointing', False)
        if is_gc:
            unwrapped_model.gradient_checkpointing_disable()

        metrics = {}

        if self.accelerator.is_main_process:
            model_device = next(unwrapped_model.parameters()).device
            num_eval = len(eval_dataset)
            cer_values = []
            bcer_values = []

            print(f"\n{'='*60}")
            print(f"Validation at step {self.state.global_step} ({num_eval} samples)")
            print(f"{'='*60}")

            for idx in tqdm(range(num_eval), desc="Eval"):
                sample = eval_dataset[idx]

                conversation = [
                    {"role": "system", "content": sample["system_prompt"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": sample["audio"]},
                            {"type": "text", "text": CSFLEURS_USER_PROMPT},
                        ],
                    },
                ]

                text = self.processing_class.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                try:
                    inputs = self.processing_class(
                        text=[text],
                        audio=[sample["audio"]],
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = unwrapped_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                        )

                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                    transcription = self.processing_class.decode(
                        generated_ids, skip_special_tokens=True
                    )
                except Exception as e:
                    logger.warning(f"Eval sample {idx} failed: {e}")
                    continue

                pred = transcription.strip()
                ref = sample["transcript"]

                language = sample.get("language", "")
                lang = "zh" if language and (
                    "cmn" in language.lower() or "zho" in language.lower()
                ) else "en"

                pred_norm = _remove_sp(pred, lang)
                ref_norm = _remove_sp(ref, lang)

                cer = compute_cer(ref_norm, pred_norm)
                cer_values.append(cer)

                raw_text = sample.get("raw_text", "")
                bcer_result = compute_bcer(raw_text, pred_norm, k=15)
                if bcer_result["bcer"] is not None and bcer_result["boundary_chars"] > 0:
                    bcer_values.append(bcer_result["bcer"])

            avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
            avg_bcer = sum(bcer_values) / len(bcer_values) if bcer_values else None

            print(f"  Eval CER:  {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
            if avg_bcer is not None:
                print(f"  Eval bCER: {avg_bcer:.4f} ({avg_bcer * 100:.2f}%)")
            else:
                print(f"  Eval bCER: N/A (no boundary markers)")
            print(f"{'='*60}\n")

            metrics[f"{metric_key_prefix}_cer"] = avg_cer
            metrics[f"{metric_key_prefix}_bcer"] = avg_bcer if avg_bcer is not None else 0.0

        # Sync across processes
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

        if is_gc:
            unwrapped_model.gradient_checkpointing_enable()
        if was_training:
            unwrapped_model.train()

        # Log to wandb
        if self.accelerator.is_main_process and metrics:
            from transformers import is_wandb_available
            if is_wandb_available():
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=self.state.global_step)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics


# ============================================================================
# Training Arguments
# ============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path to pretrained model"},
    )
    trust_remote_code: bool = field(default=True)
    torch_dtype: Optional[str] = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="flash_attention_2")


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default="./csfleurs_data",
        metadata={"help": "Directory containing CS-FLEURS data"},
    )
    from_hf: bool = field(
        default=False,
        metadata={"help": "Load dataset directly from HuggingFace"},
    )
    subset: str = field(
        default="xtts_train",
        metadata={"help": "Dataset subset"},
    )
    language_pair: Optional[str] = field(
        default=None,
        metadata={"help": "Filter for specific language pair (e.g., ara-eng)"},
    )
    max_duration: float = field(default=30.0)
    min_duration: float = field(default=0.5)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    sample_rate: int = field(default=16000)


@dataclass
class LoRAArguments:
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05)
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    freeze_audio_encoder: bool = field(default=True)
    use_rslora: bool = field(default=False)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"LoRA arguments: {lora_args}")

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
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Freeze audio encoder
    if lora_args.freeze_audio_encoder and hasattr(model, "audio_tower"):
        logger.info("Freezing audio encoder")
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Load datasets
    logger.info("Loading datasets...")

    if data_args.from_hf:
        full_dataset = CSFleursLoRADatasetHF(
            subset=data_args.subset,
            language_pair=data_args.language_pair,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            num_examples=data_args.max_train_samples,
        )
    else:
        full_dataset = CSFleursLoRADataset(
            data_dir=data_args.data_dir,
            subset=data_args.subset,
            language_pair=data_args.language_pair,
            sample_rate=data_args.sample_rate,
            max_duration=data_args.max_duration,
            min_duration=data_args.min_duration,
            num_examples=data_args.max_train_samples,
            data_seed=training_args.seed,
        )

    # Split dataset 9:1 for training and validation, stratified by language pair
    from collections import defaultdict
    from torch.utils.data import Subset
    import random as _random

    rng = _random.Random(42)

    lang_to_indices = defaultdict(list)
    for idx in range(len(full_dataset)):
        lang = full_dataset.samples[idx].get("language", "unknown")
        lang_to_indices[lang].append(idx)

    train_indices = []
    val_indices = []
    lang_split_info = {}

    for lang in sorted(lang_to_indices.keys()):
        indices = lang_to_indices[lang]
        rng.shuffle(indices)
        n_val = max(1, len(indices) // 10)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        lang_split_info[lang] = {"train": len(indices) - n_val, "val": n_val}

    train_indices.sort()
    val_indices.sort()

    # Cap validation samples for eval speed
    if data_args.max_eval_samples and len(val_indices) > data_args.max_eval_samples:
        rng2 = _random.Random(42)
        rng2.shuffle(val_indices)
        val_indices = sorted(val_indices[:data_args.max_eval_samples])

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, val_indices)

    logger.info(f"Split: {len(train_dataset)} train, {len(eval_dataset)} validation (stratified by language)")
    for lang, counts in sorted(lang_split_info.items()):
        logger.info(f"  {lang}: {counts['train']} train, {counts['val']} val")

    # Create data collator
    data_collator = CSFleursDataCollator(
        processor=processor,
        sample_rate=data_args.sample_rate,
    )

    # Create optimizer manually to avoid scheduler param group mismatch with PEFT
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.0,
    )

    # Create a dummy constant scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # Create trainer with custom optimizer and scheduler
    trainer = Qwen2AudioCSFleursTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        optimizers=(optimizer, scheduler),
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
