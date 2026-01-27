"""
Evaluate Qwen2-Audio on CS-FLEURS Code-Switched Speech Recognition.

This script evaluates either the raw Qwen2-Audio model or a trained checkpoint
on CS-FLEURS examples that are NOT part of the training set.
Metrics: CER (Character Error Rate) and Entity Recall.

Usage:
    # Single GPU evaluation
    python evaluate_csfleurs.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Multi-GPU evaluation (8 GPUs in parallel)
    torchrun --nproc_per_node 8 evaluate_csfleurs.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Evaluate a trained checkpoint
    python evaluate_csfleurs.py \
        --model_name_or_path ./outputs/csfleurs_xtts_train_cmn-eng_cgpr/checkpoint-30 \
        --subset xtts_test1 \
        --language_pair cmn-eng \
        --num_examples 100
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset.csfleurs_dataset import CSFleursDatasetLocal, _get_language_name
from utils.rewards import (
    _remove_sp,
    _strip_punctuation,
)


def compute_cer(ref: str, hyp: str) -> float:
    """
    Compute Character Error Rate between reference and hypothesis.

    Args:
        ref: Reference string
        hyp: Hypothesis string

    Returns:
        CER as a float (0.0 to 1.0+)
    """
    # Strip punctuation and remove spaces for CER computation
    ref_clean = _strip_punctuation(ref).replace(" ", "")
    hyp_clean = _strip_punctuation(hyp).replace(" ", "")
    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Levenshtein distance
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def compute_entity_recall(pred_norm: str, entity_list: list, lang: str = "en") -> tuple:
    """
    Compute entity recall: fraction of code-switched entities that appear in the prediction.

    Args:
        pred_norm: Normalized prediction string
        entity_list: List of code-switched entities to check
        lang: Language code

    Returns:
        Tuple of (recall score, num_found, num_total, found_entities, missing_entities)
    """
    if not entity_list:
        return 1.0, 0, 0, [], []

    found_entities = []
    missing_entities = []

    # Normalize prediction for matching
    pred_lower = pred_norm.lower()

    for entity in entity_list:
        # Normalize entity the same way as prediction
        entity_norm = _remove_sp(entity, lang).lower()

        if entity_norm in pred_lower:
            found_entities.append(entity)
        else:
            missing_entities.append(entity)

    num_found = len(found_entities)
    num_total = len(entity_list)
    recall = num_found / num_total if num_total > 0 else 1.0

    return recall, num_found, num_total, found_entities, missing_entities


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CS-FLEURS with CER and Entity Recall")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Path to pretrained model or checkpoint directory",
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/Qwen2-Audio/csfleurs_data",
        help="Directory containing CS-FLEURS data (git clone of byan/cs-fleurs)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="xtts_train",
        choices=["read_test", "xtts_train", "xtts_test1", "xtts_test2", "mms_test"],
        help="Dataset subset to evaluate on",
    )
    parser.add_argument(
        "--language_pair",
        type=str,
        default=None,
        help="Filter by language pair (e.g., 'ara-eng', 'cmn-eng'). None for all.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--skip_examples",
        type=int,
        default=0,
        help="Number of examples to skip (use for held-out eval on training subset)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--max_audio_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--filter_unsupported",
        action="store_true",
        default=True,
        help="Filter out languages not supported by Qwen2-Audio (hin, tur, pol, nld, hun, ces)",
    )
    parser.add_argument(
        "--no_filter_unsupported",
        action="store_true",
        help="Include all languages (don't filter unsupported)",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--raw_model_prompt",
        action="store_true",
        help="Use simple prompt without <answer> tags (for raw/untrained models)",
    )

    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each example",
    )

    return parser.parse_args()


def get_rank_and_world_size():
    """Get distributed rank and world size, or defaults for single GPU."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    # Check environment variables for torchrun/accelerate
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def load_model_and_processor(model_path: str, local_rank: int = 0):
    """Load model and processor from path or HuggingFace."""
    logging.info(f"Loading model from: {model_path} to GPU {local_rank}")

    # Check if it's a checkpoint directory
    is_checkpoint = os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "config.json")) or
        os.path.exists(os.path.join(model_path, "model.safetensors"))
    )

    # For distributed: load to specific GPU using explicit cuda device string
    if torch.cuda.device_count() > 1:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = "auto"

    if is_checkpoint:
        logging.info("Loading from checkpoint directory")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        # Try to load processor from checkpoint, fall back to base model
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            logging.info("Processor not found in checkpoint, loading from base model")
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                trust_remote_code=True,
            )
    else:
        logging.info("Loading from HuggingFace model hub")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    return model, processor


def load_eval_dataset(args):
    """Load evaluation dataset, skipping training examples if needed."""
    # Load more examples than needed to account for skipping
    total_needed = args.skip_examples + args.num_examples

    logging.info(f"Loading CS-FLEURS {args.subset} from: {args.data_dir}")
    if args.language_pair:
        logging.info(f"Filtering by language pair: {args.language_pair}")

    # Determine if we should filter unsupported languages
    filter_unsupported = args.filter_unsupported and not args.no_filter_unsupported

    dataset = CSFleursDatasetLocal(
        data_dir=args.data_dir,
        subset=args.subset,
        language_pair=args.language_pair,
        num_examples=total_needed,
        max_audio_duration=args.max_audio_duration,
        filter_unsupported=filter_unsupported,
    )

    # Check if we have enough examples
    if len(dataset) <= args.skip_examples:
        raise ValueError(
            f"Dataset has only {len(dataset)} examples, "
            f"cannot skip {args.skip_examples} and evaluate {args.num_examples}"
        )

    # Return indices for evaluation (skip training examples)
    eval_indices = list(range(args.skip_examples, min(len(dataset), total_needed)))
    logging.info(f"Evaluating on {len(eval_indices)} examples (indices {eval_indices[0]} to {eval_indices[-1]})")

    return dataset, eval_indices


def generate_transcription(model, processor, sample, args):
    """Generate transcription for a single sample."""
    language = sample.get("language", "")

    if args.raw_model_prompt:
        # Simple prompt for raw/untrained models (no <answer> tags)
        if language:
            lang_name = _get_language_name(language)
            prompt_text = (
                f"Transcribe the audio exactly as spoken. "
                f"The speech contains {lang_name} and English code-switching."
            )
        else:
            prompt_text = "Transcribe the audio exactly as spoken."
    else:
        # Training-style prompt with <answer> tags (for fine-tuned models)
        if language:
            lang_name = _get_language_name(language)
            prompt_text = (
                f"You are a speech transcription system for code-switched speech. "
                f"The audio contains speech mixing {lang_name} and English. "
                f"Output ONLY the exact words spoken, preserving the language switches. "
                f"Output the transcription in <answer> </answer>."
            )
        else:
            prompt_text = (
                "You are a speech transcription system for code-switched speech. "
                "Output ONLY the exact words spoken, preserving any language switches. "
                "Output the transcription in <answer> </answer>."
            )

    # Build conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template
    from trl.data_utils import maybe_apply_chat_template
    example = {"prompt": conversation}
    prompt_text_processed = maybe_apply_chat_template(example, processor)["prompt"]

    # Get audio
    audio = sample["audio"]
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)

    # Process inputs
    inputs = processor(
        text=[prompt_text_processed],
        audio=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        if args.temperature == 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )

    # Decode output
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    transcription = processor.decode(generated_ids, skip_special_tokens=True)

    return transcription


def extract_answer(text: str) -> str:
    """Extract content from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def evaluate(args):
    """Main evaluation function with multi-GPU support."""
    # Get distributed info
    rank, world_size = get_rank_and_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main = (rank == 0)

    # Initialize distributed if using multiple GPUs
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # Setup logging (only main process logs info)
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"[GPU {rank}] %(asctime)s - %(levelname)s - %(message)s",
    )

    if is_main:
        logging.info(f"Running evaluation with {world_size} GPU(s)")

    # Load model and processor
    model, processor = load_model_and_processor(args.model_name_or_path, local_rank)

    # Load dataset
    dataset, eval_indices = load_eval_dataset(args)

    # Split indices across GPUs
    indices_per_gpu = len(eval_indices) // world_size
    start_idx = rank * indices_per_gpu
    end_idx = start_idx + indices_per_gpu if rank < world_size - 1 else len(eval_indices)
    my_indices = eval_indices[start_idx:end_idx]

    if is_main:
        logging.info(f"Total examples: {len(eval_indices)}, this GPU: {len(my_indices)}")

    # Results storage
    results = []

    # Progress bar only on main process
    iterator = tqdm(my_indices, desc=f"GPU {rank}", disable=not is_main)

    for idx in iterator:
        sample = dataset[idx]

        # Generate transcription
        try:
            generated = generate_transcription(model, processor, sample, args)
        except Exception as e:
            logging.warning(f"Failed to generate for sample {idx}: {e}")
            continue

        # Extract answer from tags
        pred = extract_answer(generated)

        # For raw models, truncate after the first ':' if present
        if args.raw_model_prompt and ':' in pred:
            pred = pred.split(':', 1)[1].strip()

        # Get reference
        solution = sample.get("solution", "")
        ref = extract_answer(solution)

        # Get language for normalization
        language = sample.get("language", "")
        # Use 'zh' for Chinese, 'en' otherwise
        if language and ("cmn" in language.lower() or "zho" in language.lower()):
            lang = "zh"
        else:
            lang = "en"

        # Normalize for metrics
        pred_norm = _remove_sp(pred, lang)
        ref_norm = _remove_sp(ref, lang)

        # Compute CER
        cer = compute_cer(ref_norm, pred_norm)

        # Compute entity recall for code-switched entities
        entity_list = sample.get("entity_list", [])
        entity_recall, num_found, num_total, found_ents, missing_ents = compute_entity_recall(
            pred_norm, entity_list, lang
        )

        result = {
            "index": idx,
            "uniq_id": sample.get("uniq_id", f"sample_{idx}"),
            "language": language,
            "reference": ref,
            "prediction": pred,
            "ref_norm": ref_norm,
            "pred_norm": pred_norm,
            "cer": cer,
            "entity_list": entity_list,
            "entity_recall": entity_recall,
            "found_entities": found_ents,
            "missing_entities": missing_ents,
        }
        results.append(result)

        if args.verbose and is_main:
            print(f"\n[{idx}] {result['uniq_id']} ({language})")
            print(f"  Ref:  {ref[:80]}{'...' if len(ref) > 80 else ''}")
            print(f"  Pred: {pred[:80]}{'...' if len(pred) > 80 else ''}")
            print(f"  CER: {cer:.4f}, Entity Recall: {entity_recall:.4f}")

    # Gather results from all GPUs
    if world_size > 1:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if is_main:
            results = [r for gpu_results in all_results for r in gpu_results]

    # Compute aggregate metrics (only on main process)
    if is_main or world_size == 1:
        num_samples = len(results)

        if num_samples > 0:
            avg_cer = sum(r["cer"] for r in results) / num_samples

            # Entity recall only for samples with entities
            samples_with_entities = [r for r in results if r["entity_list"]]
            num_with_entities = len(samples_with_entities)
            avg_entity_recall = (
                sum(r["entity_recall"] for r in samples_with_entities) / num_with_entities
                if num_with_entities > 0 else 0
            )

            # Per-language breakdown
            lang_stats = {}
            for r in results:
                lang = r["language"]
                if lang not in lang_stats:
                    lang_stats[lang] = {"cer": [], "entity_recall": []}
                lang_stats[lang]["cer"].append(r["cer"])
                if r["entity_list"]:
                    lang_stats[lang]["entity_recall"].append(r["entity_recall"])
        else:
            avg_cer = avg_entity_recall = 0
            num_with_entities = 0
            lang_stats = {}

        # Print summary
        print("\n" + "=" * 60)
        print("CS-FLEURS EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model:           {args.model_name_or_path}")
        print(f"Subset:          {args.subset}")
        print(f"Language pair:   {args.language_pair or 'all'}")
        print(f"Num samples:     {num_samples}")
        print(f"Skipped:         {args.skip_examples}")
        print(f"GPUs used:       {world_size}")
        print("-" * 60)
        print(f"Average CER:           {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
        print(f"Average Entity Recall: {avg_entity_recall:.4f} ({avg_entity_recall * 100:.2f}%)")
        print(f"  (computed on {num_with_entities} samples with code-switched entities)")

        # Per-language breakdown
        if len(lang_stats) > 1:
            print("-" * 60)
            print("Per-language breakdown:")
            for lang, stats in sorted(lang_stats.items()):
                lang_cer = sum(stats["cer"]) / len(stats["cer"])
                lang_er = sum(stats["entity_recall"]) / len(stats["entity_recall"]) if stats["entity_recall"] else 0
                print(f"  {lang}: CER={lang_cer:.4f}, ER={lang_er:.4f} (n={len(stats['cer'])})")

        print("=" * 60)

        # Save detailed results
        if args.output_file:
            output_data = {
                "model": args.model_name_or_path,
                "subset": args.subset,
                "language_pair": args.language_pair,
                "num_samples": num_samples,
                "skip_examples": args.skip_examples,
                "avg_cer": avg_cer,
                "avg_entity_recall": avg_entity_recall,
                "num_with_entities": num_with_entities,
                "num_gpus": world_size,
                "timestamp": datetime.now().isoformat(),
                "per_language": {
                    lang: {
                        "avg_cer": sum(s["cer"]) / len(s["cer"]),
                        "avg_entity_recall": sum(s["entity_recall"]) / len(s["entity_recall"]) if s["entity_recall"] else 0,
                        "count": len(s["cer"]),
                    }
                    for lang, s in lang_stats.items()
                },
                "results": results,
            }

            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logging.info(f"Detailed results saved to: {args.output_file}")

    return avg_cer if is_main else 0


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
