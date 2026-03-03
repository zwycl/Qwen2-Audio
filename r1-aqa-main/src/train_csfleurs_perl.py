"""
Train Qwen2-Audio on CS-FLEURS using PeRL (DoRA fine-tuning) with CGPR rewards.

This script integrates the PeRL parameter-efficient RL framework with
CS-FLEURS code-switched speech recognition training.

Key features:
- DoRA (Weight-decomposed LoRA) - outperforms standard LoRA for RLVR tasks
- All existing reward types: WER, CER, mixed, CGPR
- Two-step training support
- VAD-based audio chunking

Usage:
    # Train with DoRA (recommended)
    python train_csfleurs_perl.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --peft_type dora \
        --reward_type cgpr \
        --out_dir ./outputs/csfleurs_dora

    # Compare with standard LoRA
    python train_csfleurs_perl.py \
        --config_path configs/zero2.json \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --peft_type lora \
        --reward_type cgpr \
        --out_dir ./outputs/csfleurs_lora
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import HfArgumentParser

# Import our custom trainer and dataset
from trainer.grpo_trainer import GRPOTrainer
from dataset.csfleurs_dataset import CSFleursDataset, CSFleursDatasetLocal
from utils.rewards import wer_reward, cer_reward, mixed_wer_cer_reward, cgpr_shaped_reward, format_reward

# Using HuggingFace PEFT (DeepSpeed ZeRO-2 compatible)


@dataclass
class PeftArguments:
    """PEFT configuration arguments (PeRL-style)."""

    peft_type: str = field(
        default="dora",
        metadata={"help": "PEFT type: lora, dora, adalora, pissa, rslora, lorafa, lora_plus, etc."},
    )
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use PEFT (parameter-efficient fine-tuning)"},
    )
    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "PEFT task type"},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha scaling factor"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout rate"},
    )
    target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated target modules (default: q_proj,v_proj,k_proj,o_proj,up_proj,down_proj)"},
    )


@dataclass
class CSFleursPerlTrainingArguments:
    """Arguments for CS-FLEURS GRPO training with PeRL."""

    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen2-Audio-7B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_path: str = field(
        default="configs/zero2.json",
        metadata={"help": "Path to DeepSpeed config file"},
    )

    # Dataset arguments
    subset: str = field(
        default="xtts_train",
        metadata={"help": "CS-FLEURS subset (read_test, xtts_train, xtts_test1, xtts_test2, mms_test)"},
    )
    language_pair: Optional[str] = field(
        default=None,
        metadata={"help": "Filter for specific language pair (e.g., chinese_english)"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Local directory for CS-FLEURS data (if not using HuggingFace)"},
    )
    num_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training examples (None for all)"},
    )
    max_audio_duration: float = field(
        default=60.0,
        metadata={"help": "Maximum audio duration in seconds (longer audio will be VAD-chunked)"},
    )
    max_audio_chunk: float = field(
        default=30.0,
        metadata={"help": "Maximum chunk size after VAD splitting (default 30s)"},
    )
    use_vad_chunking: bool = field(
        default=True,
        metadata={"help": "Use VAD-based chunking for long audio"},
    )

    # Output arguments
    out_dir: str = field(
        default="./outputs/csfleurs_perl",
        metadata={"help": "Output directory for checkpoints"},
    )

    # WandB arguments
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use WandB logging"})
    run_name: str = field(
        default="CSFleurs-PeRL-GRPO",
        metadata={"help": "WandB run name"},
    )

    # Two-step training
    two_step_training: bool = field(
        default=False,
        metadata={"help": "Enable two-step training: first generate draft, then refine using draft+context+audio"},
    )

    # Reward type
    reward_type: str = field(
        default="cgpr",
        metadata={"help": "Reward type: 'wer', 'cer', 'mixed', or 'cgpr' (confidence-gated process rewards)"},
    )
    wer_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for WER in mixed reward (default 0.5)"},
    )
    cer_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for CER in mixed reward (default 0.5)"},
    )
    # CGPR-specific parameters
    cgpr_alpha: float = field(
        default=0.1,
        metadata={"help": "CGPR: reward coefficient for correct entities (default 0.1)"},
    )
    cgpr_beta: float = field(
        default=0.2,
        metadata={"help": "CGPR: penalty coefficient for incorrect entities (default 0.2)"},
    )

    # Format reward
    use_format_reward: bool = field(
        default=True,
        metadata={"help": "Include format reward (checks for <answer> tags)"},
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

    # Resume training
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume training from"},
    )


def apply_peft_to_model(model, peft_args: PeftArguments):
    """Apply PEFT method using HuggingFace PEFT (DeepSpeed ZeRO-2 compatible)."""
    from peft import LoraConfig, get_peft_model, TaskType

    peft_type = peft_args.peft_type.lower()

    # Parse target modules
    if peft_args.target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    else:
        target_modules = [m.strip() for m in peft_args.target_modules.split(",")]

    # DoRA = LoRA with use_dora=True
    use_dora = peft_type == "dora"

    logging.info(f"Applying HuggingFace PEFT: {'DoRA' if use_dora else 'LoRA'}")
    logging.info(f"  rank={peft_args.lora_r}, alpha={peft_args.lora_alpha}, dropout={peft_args.lora_dropout}")
    logging.info(f"  target_modules={target_modules}")

    lora_config = LoraConfig(
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=use_dora,
    )

    model = get_peft_model(model, lora_config)
    return model


def main():
    # Parse arguments
    parser = HfArgumentParser((CSFleursPerlTrainingArguments, PeftArguments))
    args, peft_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("=" * 60)
    logging.info("CS-FLEURS PEFT GRPO Training (LoRA/DoRA)")
    logging.info("=" * 60)
    logging.info(f"Model: {args.model_name_or_path}")
    logging.info(f"PEFT type: {peft_args.peft_type}")
    logging.info(f"  rank={peft_args.lora_r}, alpha={peft_args.lora_alpha}")
    logging.info(f"Subset: {args.subset}")
    logging.info(f"Language pair: {args.language_pair or 'all'}")
    logging.info(f"Num examples: {args.num_examples or 'all'}")
    logging.info(f"Two-step training: {args.two_step_training}")
    logging.info(f"Reward type: {args.reward_type}")
    if args.reward_type == "mixed":
        logging.info(f"  WER weight: {args.wer_weight}, CER weight: {args.cer_weight}")
    elif args.reward_type == "cgpr":
        logging.info(f"  CGPR alpha: {args.cgpr_alpha}, beta: {args.cgpr_beta}")
    logging.info(f"Output: {args.out_dir}")
    logging.info("=" * 60)

    # Load dataset
    chunking_method = "VAD" if args.use_vad_chunking else "none"
    if args.data_dir:
        logging.info(f"Loading CS-FLEURS from local directory: {args.data_dir} ({chunking_method} chunking)")
        train_dataset = CSFleursDatasetLocal(
            data_dir=args.data_dir,
            subset=args.subset,
            language_pair=args.language_pair,
            num_examples=args.num_examples,
            max_audio_duration=args.max_audio_duration,
            max_audio_chunk=args.max_audio_chunk,
            use_vad_chunking=args.use_vad_chunking,
        )
    else:
        logging.info(f"Loading CS-FLEURS from HuggingFace ({chunking_method} chunking)...")
        train_dataset = CSFleursDataset(
            subset=args.subset,
            language_pair=args.language_pair,
            num_examples=args.num_examples,
            max_audio_duration=args.max_audio_duration,
            max_audio_chunk=args.max_audio_chunk,
            use_vad_chunking=args.use_vad_chunking,
        )

    logging.info(f"Loaded {len(train_dataset)} training examples")

    if len(train_dataset) == 0:
        raise ValueError("No training examples loaded! Check subset and language_pair settings.")

    # Helper to convert CS-FLEURS language codes (ara-eng, cmn-eng) to normalizer codes (en, zh)
    def _convert_language_codes(language, num_completions):
        if language is None:
            return ["en"] * num_completions
        elif isinstance(language, str):
            return ["zh" if "cmn" in language or "zho" in language else "en"] * num_completions
        else:
            lang_codes = []
            for lang in language:
                if lang and ("cmn" in lang.lower() or "zho" in lang.lower()):
                    lang_codes.append("zh")
                else:
                    lang_codes.append("en")
            return lang_codes

    # Setup reward function based on reward_type
    if args.reward_type == "cer":
        def csfleurs_reward(prompts, completions, solution, language=None, **kwargs):
            """CER reward for CS-FLEURS."""
            lang_codes = _convert_language_codes(language, len(completions))
            return cer_reward(completions, solution, language=lang_codes, **kwargs)
        logging.info("Using CER (Character Error Rate) reward")
    elif args.reward_type == "mixed":
        def csfleurs_reward(prompts, completions, solution, language=None, **kwargs):
            """Mixed WER+CER reward for CS-FLEURS."""
            lang_codes = _convert_language_codes(language, len(completions))
            return mixed_wer_cer_reward(
                completions, solution, language=lang_codes,
                wer_weight=args.wer_weight, cer_weight=args.cer_weight, **kwargs
            )
        logging.info(f"Using mixed WER+CER reward (WER:{args.wer_weight}, CER:{args.cer_weight})")
    elif args.reward_type == "cgpr":
        def csfleurs_reward(prompts, completions, solution, language=None, entity_list=None, **kwargs):
            """CGPR reward for CS-FLEURS with code-switched entities."""
            lang_codes = _convert_language_codes(language, len(completions))
            return cgpr_shaped_reward(
                completions, solution,
                bias_list=entity_list,
                language=lang_codes,
                alpha=args.cgpr_alpha,
                beta=args.cgpr_beta,
                **kwargs
            )
        logging.info(f"Using CGPR (Confidence-Gated Process Rewards) with α={args.cgpr_alpha}, β={args.cgpr_beta}")
        logging.info("  Code-switched English phrases will be treated as entities")
    elif args.reward_type == "format":
        # Format-only baseline - no ASR reward, just format compliance
        reward_funcs = [format_reward]
        logging.info("Using FORMAT-ONLY reward (baseline - no ASR metrics)")
    else:  # default: wer
        def csfleurs_reward(prompts, completions, solution, language=None, **kwargs):
            """WER reward for CS-FLEURS."""
            lang_codes = _convert_language_codes(language, len(completions))
            return wer_reward(completions, solution, language=lang_codes, **kwargs)
        logging.info("Using WER (Word Error Rate) reward")

    # For non-format-only modes, setup reward_funcs list
    if args.reward_type != "format":
        reward_funcs = [csfleurs_reward]

        # Add format reward if enabled
        if args.use_format_reward:
            reward_funcs.append(format_reward)
            logging.info("Format reward enabled (checks for <answer> tags)")

    # Setup training arguments
    from trl import GRPOConfig

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
        lr_scheduler_type="constant",  # Avoid scheduler/optimizer param group mismatch with PEFT
        logging_steps=1,
        bf16=True,
        report_to="wandb" if args.use_wandb else [],
        gradient_checkpointing=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        run_name=args.run_name,
        save_steps=args.save_steps,
        save_only_model=False,  # Enables DeepSpeed checkpoint resume
        temperature=args.temperature,
        num_generations=args.num_generations,
    )

    # Load model and apply PEFT
    from transformers import Qwen2AudioForConditionalGeneration

    logging.info(f"Loading model: {args.model_name_or_path}")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # PyTorch native attention (flash_attn build failed)
    )

    # Apply HuggingFace PEFT (DoRA/LoRA)
    if peft_args.use_peft:
        model = apply_peft_to_model(model, peft_args)
        model.print_trainable_parameters()

    # Callback to skip saving until step > 40
    from transformers import TrainerCallback

    class SkipEarlySaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step <= 40:
                control.should_save = False
            return control

    # Create trainer - DeepSpeed manages the optimizer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        two_step_training=args.two_step_training,
        callbacks=[SkipEarlySaveCallback()],
    )

    if args.two_step_training:
        logging.info("Two-step training enabled: Pass 1 (draft) + Pass 2 (refinement) per step")

    # Train
    if args.resume_from_checkpoint:
        logging.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logging.info(f"Saving model to {args.out_dir}")
    trainer.save_model(args.out_dir)
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
