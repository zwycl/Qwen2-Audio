"""
Train Qwen2-Audio on ContextASR-Bench with WER-based reward.

This script trains an audio model for Contextual Automatic Speech Recognition
using GRPO (Group Relative Policy Optimization) with WER as the reward signal.

The key feature is the use of entity hints in the prompt to help the model
recognize specific names/terms in the audio:
"You are a speech transcription system. Output ONLY the exact words spoken.
The following names/terms may appear: {entity_str}. Use correct spelling for these terms."

Usage:
    # Basic training (Dialogue config, English)
    python train_contextasr.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --out_dir ./outputs/contextasr

    # Train on Speech config
    python train_contextasr.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --dataset_config ContextASR-Speech \
        --out_dir ./outputs/contextasr_speech

    # Train on Mandarin
    python train_contextasr.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --data_dir ./contextasr_data \
        --language Mandarin \
        --out_dir ./outputs/contextasr_mandarin

    # Load directly from HuggingFace
    python train_contextasr.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --from_hf \
        --out_dir ./outputs/contextasr
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import wer_reward, format_reward
from dataset.contextasr_dataset import ContextASRDataset, ContextASRDatasetFromHF


@dataclass
class ContextASRTrainingArguments:
    """Arguments for ContextASR GRPO training."""

    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )

    # Data arguments
    data_dir: Optional[str] = field(
        default="./contextasr_data",
        metadata={"help": "Directory containing downloaded ContextASR-Bench data"},
    )
    from_hf: bool = field(
        default=False,
        metadata={"help": "Load dataset directly from HuggingFace instead of local files"},
    )
    dataset_config: str = field(
        default="ContextASR-Dialogue",
        metadata={"help": "Dataset config (ContextASR-Dialogue or ContextASR-Speech)"},
    )
    language: str = field(
        default="English",
        metadata={"help": "Language (English or Mandarin)"},
    )
    max_duration: float = field(
        default=180.0,
        metadata={"help": "Maximum audio duration in seconds (ContextASR samples are 99-253s)"},
    )
    min_duration: float = field(
        default=0.5,
        metadata={"help": "Minimum audio duration in seconds"},
    )
    max_audio_chunk: float = field(
        default=30.0,
        metadata={"help": "Maximum chunk size for model input (Qwen2-Audio works best under 30s)"},
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training examples to use (limits dataset size)"},
    )
    include_no_context: bool = field(
        default=False,
        metadata={"help": "Include samples without entity list for robustness"},
    )

    # Training arguments
    config_path: str = field(
        default="configs/zero2.json",
        metadata={"help": "DeepSpeed config path"},
    )
    out_dir: str = field(
        default="./outputs/contextasr",
        metadata={"help": "Output directory for model checkpoints"},
    )

    # Reward arguments
    use_format_reward: bool = field(
        default=True,
        metadata={"help": "Include format reward (checks for <answer> tags)"},
    )

    # WandB arguments
    use_wandb: str = field(
        default="false",
        metadata={"help": "Whether to use wandb for logging (true/false)"},
    )
    run_name: str = field(
        default="ContextASR-GRPO",
        metadata={"help": "WandB run name"},
    )

    # Training hyperparameters
    num_train_epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    max_steps: int = field(default=-1, metadata={"help": "Maximum training steps (-1 for no limit)"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per device"})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=1e-6, metadata={"help": "Learning rate"})
    num_generations: int = field(default=4, metadata={"help": "Number of generations per prompt for GRPO"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature for generation"})
    max_completion_length: int = field(default=512, metadata={"help": "Maximum completion length"})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every N steps"})

    # Distributed training (set by launcher)
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})


def main():
    parser = HfArgumentParser(ContextASRTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    transformers.logging.set_verbosity_info()
    logging.info(f"Training arguments: {args}")

    # Setup reward functions
    # WER reward: lower WER = higher reward (returns negative WER)
    reward_funcs = [wer_reward]
    if args.use_format_reward:
        reward_funcs.append(format_reward)
    logging.info(f"Using reward functions: {[f.__name__ for f in reward_funcs]}")

    # Load dataset
    if args.from_hf:
        logging.info(f"Loading ContextASR {args.dataset_config}/{args.language} from HuggingFace...")
        train_dataset = ContextASRDatasetFromHF(
            config=args.dataset_config,
            language=args.language,
            max_duration=args.max_duration,
            min_duration=args.min_duration,
            max_audio_chunk=args.max_audio_chunk,
            num_examples=args.num_examples,
            include_no_context=args.include_no_context,
        )
    else:
        logging.info(f"Loading ContextASR {args.dataset_config}/{args.language} from local directory: {args.data_dir}")
        train_dataset = ContextASRDataset(
            data_dir=args.data_dir,
            config=args.dataset_config,
            language=args.language,
            max_duration=args.max_duration,
            min_duration=args.min_duration,
            max_audio_chunk=args.max_audio_chunk,
            num_examples=args.num_examples,
            include_no_context=args.include_no_context,
        )

    logging.info(f"Dataset size: {len(train_dataset)} samples")

    # Setup training config
    training_args = GRPOConfig(
        seed=42,
        data_seed=42,
        output_dir=args.out_dir,
        deepspeed=args.config_path,
        max_prompt_length=512,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        bf16=True,
        report_to="wandb" if args.use_wandb == "true" else [],
        gradient_checkpointing=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        run_name=args.run_name,
        save_steps=args.save_steps,
        save_only_model=True,
        temperature=args.temperature,
        num_generations=args.num_generations,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    # Train
    logging.info("Starting training...")
    trainer.train()

    # Save final model
    logging.info(f"Saving model to {args.out_dir}")
    trainer.save_model(args.out_dir)
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
