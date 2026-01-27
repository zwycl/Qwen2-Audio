"""
Evaluate Qwen2-Audio on ContextASR-Bench with WER and Entity Recall metrics.

This script evaluates either the raw Qwen2-Audio model or a trained checkpoint
on ContextASR examples that are NOT part of the training set.

Usage:
    # Single GPU evaluation
    python evaluate_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Multi-GPU evaluation (8 GPUs in parallel)
    accelerate launch --num_processes 8 evaluate_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Or use torchrun
    torchrun --nproc_per_node 8 evaluate_contextasr.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Evaluate a trained checkpoint
    accelerate launch --num_processes 8 evaluate_contextasr.py \
        --model_name_or_path ./outputs/contextasr_ContextASR-Dialogue_English_cgpr/checkpoint-30 \
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

from dataset.contextasr_dataset import ContextASRDataset, ContextASRDatasetFromHF
from utils.rewards import (
    _compute_single_wer,
    _remove_sp,
    EvaluationTokenizer,
    english_normalizer,
    basic_normalizer,
)


def compute_entity_recall(pred_norm: str, entity_list: list, lang: str = "en") -> tuple:
    """
    Compute entity recall: fraction of entities that appear in the prediction.

    Args:
        pred_norm: Normalized prediction string
        entity_list: List of entities to check
        lang: Language code ('en' or 'zh')

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
    parser = argparse.ArgumentParser(description="Evaluate ContextASR with WER and Entity Recall")

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
        default="/home/ubuntu/Qwen2-Audio/contextasr_data",
        help="Directory containing downloaded ContextASR-Bench data",
    )
    parser.add_argument(
        "--from_hf",
        action="store_true",
        help="Load dataset directly from HuggingFace instead of local files",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="ContextASR-Dialogue",
        help="Dataset config (ContextASR-Dialogue or ContextASR-Speech)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language (English or Mandarin)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--skip_examples",
        type=int,
        default=1000,
        help="Number of examples to skip (training set size)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--max_audio_chunk",
        type=float,
        default=30.0,
        help="Maximum audio chunk duration in seconds",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (only 1 supported currently)",
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
    """Load evaluation dataset, skipping training examples."""
    # Load more examples than needed to account for skipping
    total_needed = args.skip_examples + args.num_examples

    if args.from_hf:
        logging.info(f"Loading ContextASR {args.dataset_config}/{args.language} from HuggingFace...")
        dataset = ContextASRDatasetFromHF(
            config=args.dataset_config,
            language=args.language,
            max_audio_chunk=args.max_audio_chunk,
            num_examples=total_needed,
            include_no_context=False,
        )
    else:
        logging.info(f"Loading ContextASR {args.dataset_config}/{args.language} from: {args.data_dir}")
        dataset = ContextASRDataset(
            data_dir=args.data_dir,
            config=args.dataset_config,
            language=args.language,
            max_audio_chunk=args.max_audio_chunk,
            num_examples=total_needed,
            include_no_context=False,
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
    # Build the conversation/prompt
    entity_list = sample.get("entity_list", [])

    if args.raw_model_prompt:
        # Simple prompt for raw/untrained models (no <answer> tags)
        if entity_list:
            entity_str = ", ".join(entity_list)
            prompt_text = (
                f"Transcribe the audio exactly as spoken. "
                f"These names/terms may appear: {entity_str}."
            )
        else:
            prompt_text = "Transcribe the audio exactly as spoken."
    else:
        # Training-style prompt with <answer> tags (for fine-tuned models)
        if entity_list:
            entity_str = ", ".join(entity_list)
            prompt_text = (
                f"You are a speech transcription system. Output ONLY the exact words spoken. "
                f"The following names/terms may appear: {entity_str}. Use correct spelling for these terms. "
                f"Output the transcription in <answer> </answer>."
            )
        else:
            prompt_text = "Transcribe the following audio. Output the transcription in <answer> </answer>."

    # Build conversation format (same as dataset/training)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template using trl's maybe_apply_chat_template (same as training)
    from trl.data_utils import maybe_apply_chat_template
    example = {"prompt": conversation}
    prompt_text_processed = maybe_apply_chat_template(example, processor)["prompt"]

    # Get audio
    audio = sample["audio"]
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)

    # Process inputs (match training: text as list, audio parameter name)
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


def smart_join_chunks(texts: list) -> str:
    """
    Join text chunks intelligently, avoiding double spaces and handling
    word boundaries correctly when chunks may have been split mid-word.

    Heuristic for mid-word split detection:
    - Previous chunk ends with lowercase letter (not end of sentence)
    - Next chunk starts with lowercase letter (not new sentence/proper noun)
    - No punctuation at boundary
    """
    if not texts:
        return ""

    result = texts[0]
    for text in texts[1:]:
        if not text:
            continue
        if not result:
            result = text
            continue

        # Check if we need a space between chunks
        ends_with_space = result[-1].isspace()
        starts_with_space = text[0].isspace()

        if ends_with_space or starts_with_space:
            # Already have space, just concatenate and normalize
            result = result.rstrip() + " " + text.lstrip()
        elif result[-1].islower() and text[0].islower():
            # Both lowercase at boundary - likely mid-word split, NO space
            result = result + text
        elif result[-1].isalnum() and text[0].isalnum():
            # Alphanumeric but not both lowercase - add space (new word/sentence)
            result = result + " " + text
        else:
            # Punctuation at boundary - add space
            result = result + " " + text

    return result


def evaluate(args):
    """Main evaluation function with multi-GPU support."""
    # Get distributed info
    rank, world_size = get_rank_and_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main = (rank == 0)

    # Initialize distributed if using multiple GPUs
    if world_size > 1 and not dist.is_initialized():
        # CRITICAL: Set the CUDA device BEFORE initializing NCCL
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # Setup logging (only main process logs info)
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"[GPU {rank}] %(asctime)s - %(levelname)s - %(message)s",
    )

    if is_main:
        logging.info(f"Running evaluation with {world_size} GPU(s)")

    # Load model and processor to this GPU
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

    # Language code for normalization
    lang = "en" if args.language == "English" else "zh"

    # Results storage (per-chunk, later grouped by segment)
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

        # For raw models, truncate after the first ':' (model often outputs "Transcription: ...")
        if args.raw_model_prompt and ':' in pred:
            pred = pred.split(':', 1)[1].strip()

        # Get reference
        solution = sample.get("solution", "")
        ref = extract_answer(solution)

        # Normalize for WER computation
        pred_norm = _remove_sp(pred, lang)
        ref_norm = _remove_sp(ref, lang)

        # Get entity list (metrics computed later by grouping chunks per segment)
        entity_list = sample.get("entity_list", [])

        # Store chunk result (WER and entity recall computed after grouping by segment)
        result = {
            "index": idx,
            "uniq_id": sample.get("uniq_id", f"sample_{idx}"),
            "reference": ref,
            "prediction": pred,
            "ref_norm": ref_norm,
            "pred_norm": pred_norm,
            "entity_list": entity_list,
            "chunk_start": sample.get("chunk_start", 0),
            "chunk_end": sample.get("chunk_end", 0),
        }
        results.append(result)

        if args.verbose and is_main:
            print(f"\n[{idx}] {result['uniq_id']}")
            print(f"  Ref: {ref[:100]}{'...' if len(ref) > 100 else ''}")
            print(f"  Pred: {pred[:100]}{'...' if len(pred) > 100 else ''}")

    # Gather results from all GPUs
    if world_size > 1:
        # Gather all results to rank 0
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if is_main:
            # Flatten results from all GPUs
            results = [r for gpu_results in all_results for r in gpu_results]

    # Compute segment-level metrics (only on main process or single GPU)
    if is_main or world_size == 1:
        # Group chunks by base segment ID (remove _0, _1 suffix)
        from collections import defaultdict
        segments = defaultdict(lambda: {"chunks": [], "entity_list": []})

        for r in results:
            uniq_id = r["uniq_id"]
            # Extract base segment ID (remove _0, _1, _2 suffix)
            if "_" in uniq_id and uniq_id.rsplit("_", 1)[1].isdigit():
                base_id = uniq_id.rsplit("_", 1)[0]
            else:
                base_id = uniq_id

            segments[base_id]["chunks"].append(r)
            # Entity list is same for all chunks of same segment
            if r["entity_list"] and not segments[base_id]["entity_list"]:
                segments[base_id]["entity_list"] = r["entity_list"]

        # Compute metrics per segment
        segment_results = []
        total_wer = 0.0
        total_entity_recall = 0.0
        num_segments = 0
        num_with_entities = 0

        for base_id, seg_data in segments.items():
            # Sort chunks by chunk_start to maintain order
            chunks = sorted(seg_data["chunks"], key=lambda x: x["chunk_start"])

            # Combine normalized text from all chunks (smart join to handle mid-word splits)
            combined_ref_norm = smart_join_chunks([c["ref_norm"] for c in chunks])
            combined_pred_norm = smart_join_chunks([c["pred_norm"] for c in chunks])

            # Compute segment-level WER
            seg_wer = _compute_single_wer(combined_ref_norm, combined_pred_norm, lang)
            total_wer += seg_wer
            num_segments += 1

            # Compute segment-level entity recall
            entity_list = seg_data["entity_list"]
            seg_entity_recall = 0.0
            found_entities = []
            missing_entities = []
            if entity_list:
                seg_entity_recall, num_found, num_total, found_entities, missing_entities = compute_entity_recall(
                    combined_pred_norm, entity_list, lang
                )
                total_entity_recall += seg_entity_recall
                num_with_entities += 1

            # Store segment result
            segment_results.append({
                "segment_id": base_id,
                "num_chunks": len(chunks),
                "combined_reference": smart_join_chunks([c["reference"] for c in chunks]),
                "combined_prediction": smart_join_chunks([c["prediction"] for c in chunks]),
                "wer": seg_wer,
                "entity_recall": seg_entity_recall,
                "entity_list": entity_list,
                "found_entities": found_entities,
                "missing_entities": missing_entities,
                "chunks": chunks,
            })

        # Replace results with segment results for output
        results = segment_results

        # Compute averages
        avg_wer = total_wer / num_segments if num_segments > 0 else 0
        avg_entity_recall = total_entity_recall / num_with_entities if num_with_entities > 0 else 0
        num_samples = num_segments
    else:
        # Non-main processes in distributed mode
        avg_wer = 0
        avg_entity_recall = 0
        num_samples = 0
        num_with_entities = 0

    # Print summary (only main process)
    if is_main:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (Segment-Level)")
        print("=" * 60)
        print(f"Model:           {args.model_name_or_path}")
        print(f"Dataset:         {args.dataset_config}/{args.language}")
        print(f"Num segments:    {num_samples}")
        print(f"Skipped:         {args.skip_examples} (training set chunks)")
        print(f"GPUs used:       {world_size}")
        print("-" * 60)
        print(f"Average WER:          {avg_wer:.4f} ({avg_wer * 100:.2f}%)")
        print(f"Average Entity Recall: {avg_entity_recall:.4f} ({avg_entity_recall * 100:.2f}%)")
        print(f"  (computed on {num_with_entities} segments with entities)")
        print("=" * 60)

    # Save detailed results if requested (only main process)
    if args.output_file and is_main:
        output_data = {
            "model": args.model_name_or_path,
            "dataset_config": args.dataset_config,
            "language": args.language,
            "num_segments": num_samples,
            "skip_examples": args.skip_examples,
            "avg_wer": avg_wer,
            "avg_entity_recall": avg_entity_recall,
            "num_with_entities": num_with_entities,
            "num_gpus": world_size,
            "timestamp": datetime.now().isoformat(),
            "segment_results": results,
        }

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Detailed results saved to: {args.output_file}")

    return avg_wer, avg_entity_recall


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
