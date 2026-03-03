
"""
Evaluate Qwen2-Audio on SwitchLingua Code-Switched Speech Recognition.

SwitchLingua (NeurIPS 2025): 80+ hours, 11 X-English language pairs.

Metrics:
- CER (Character Error Rate): Overall transcription accuracy
- Per-language CER breakdown

Usage:
    # Single GPU evaluation
    python src/evaluate_switchlingua.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100 --raw_model_prompt

    # Multi-GPU evaluation
    torchrun --nproc_per_node 8 src/evaluate_switchlingua.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --num_examples 100

    # Specific language
    python src/evaluate_switchlingua.py \
        --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
        --language Italian --num_examples 50 --verbose --raw_model_prompt
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset.switchlingua_dataset import SwitchLinguaDataset, SWITCHLINGUA_LANG_TO_CODE
from dataset import REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE
from utils.rewards import _remove_sp, _strip_punctuation, _get_allowed_scripts, _compute_script_contamination

# Mapping from SwitchLingua language names to ISO 639-3 pair codes for script detection
_SWITCHLINGUA_LANG_TO_PAIR = {
    "Arabic": "ara-eng",
    "Mandarin": "cmn-eng",
    "Chinese": "cmn-eng",
    "German": "deu-eng",
    "French": "fra-eng",
    "Italian": "ita-eng",
    "Japanese": "jpn-eng",
    "Korean": "kor-eng",
    "Russian": "rus-eng",
    "Spanish": "spa-eng",
    "Hindi": "hin-eng",
    "Cantonese": "cmn-eng",
}


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    ref_clean = _strip_punctuation(ref).replace(" ", "")
    hyp_clean = _strip_punctuation(hyp).replace(" ", "")
    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

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
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def extract_answer(text: str) -> str:
    """Extract content from <answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def get_rank_and_world_size():
    """Get distributed rank and world size, or defaults for single GPU."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def load_model_and_processor(model_path: str, local_rank: int = 0):
    """Load model and processor from path or HuggingFace."""
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    logging.info(f"Loading model from: {model_path} to GPU {local_rank}")

    is_checkpoint = os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "config.json"))
        or os.path.exists(os.path.join(model_path, "model.safetensors"))
    )

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
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            logging.info("Processor not found in checkpoint, loading from base model")
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
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


def generate_transcription(model, processor, sample, args):
    """Generate transcription for a single sample."""
    language = sample.get("language", "")
    first_lang = language
    second_lang = "English"

    if args.raw_model_prompt:
        if language:
            prompt_text = (
                f"Transcribe the audio exactly as spoken. "
                f"The speech contains {first_lang} and {second_lang} code-switching."
            )
        else:
            prompt_text = "Transcribe the audio exactly as spoken."
    else:
        if language:
            prompt_text = (
                f"You are a speech transcription system for code-switched speech. "
                f"The audio contains speech mixing {first_lang} and {second_lang}. "
                f"Output ONLY the exact words spoken, preserving the language switches. "
                f"Output the transcription in <answer> </answer>."
            )
        else:
            prompt_text = (
                "You are a speech transcription system for code-switched speech. "
                "Output ONLY the exact words spoken, preserving any language switches. "
                "Output the transcription in <answer> </answer>."
            )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    from trl.data_utils import maybe_apply_chat_template

    example = {"prompt": conversation}
    prompt_text_processed = maybe_apply_chat_template(example, processor)["prompt"]

    audio = sample["audio"]
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)

    inputs = processor(
        text=[prompt_text_processed],
        audio=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        if args.temperature == 0:
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    transcription = processor.decode(generated_ids, skip_special_tokens=True)
    return transcription


def generate_refinement(model, processor, sample, draft_text, args):
    """Generate a refined transcription using the draft from the first pass."""
    draft_match = re.search(r"<answer>(.*?)</answer>", draft_text, re.DOTALL)
    draft_transcription = draft_match.group(1).strip() if draft_match else draft_text.strip()
    if not draft_transcription:
        draft_transcription = "[empty transcription]"
    if len(draft_transcription) > 2000:
        draft_transcription = draft_transcription[:2000]

    prompt_text = REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE.format(
        draft_transcription=draft_transcription,
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    from trl.data_utils import maybe_apply_chat_template

    example = {"prompt": conversation}
    prompt_text_processed = maybe_apply_chat_template(example, processor)["prompt"]

    audio = sample["audio"]
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)

    inputs = processor(
        text=[prompt_text_processed],
        audio=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    transcription = processor.decode(generated_ids, skip_special_tokens=True)
    return transcription


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SwitchLingua with CER"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Path to pretrained model or checkpoint directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/Qwen2-Audio/SwitchLingua_audio",
        help="Directory containing SwitchLingua data",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter by language (e.g., 'Arabic', 'Mandarin') or None/all for all",
    )
    parser.add_argument(
        "--skip_examples",
        type=int,
        default=0,
        help="Number of examples to skip (for train/eval split)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to evaluate",
    )
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
        "--filter_unsupported",
        action="store_true",
        default=True,
        help="Filter out languages not supported by Qwen2-Audio (Cantonese, Hindi)",
    )
    parser.add_argument(
        "--no_filter_unsupported",
        action="store_true",
        help="Include all languages (don't filter unsupported)",
    )
    parser.add_argument(
        "--raw_model_prompt",
        action="store_true",
        help="Use simple prompt without <answer> tags (for raw/untrained models)",
    )
    parser.add_argument(
        "--two_step",
        action="store_true",
        help="Two-step evaluation: generate draft, then refine with a second pass",
    )
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


def evaluate(args):
    """Main evaluation function with multi-GPU support."""
    rank, world_size = get_rank_and_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main = rank == 0

    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"[GPU {rank}] %(asctime)s - %(levelname)s - %(message)s",
    )

    if is_main:
        logging.info(f"Running SwitchLingua evaluation with {world_size} GPU(s)")

    # Load model and processor
    model, processor = load_model_and_processor(args.model_name_or_path, local_rank)

    # Load dataset
    filter_unsupported = args.filter_unsupported and not args.no_filter_unsupported
    dataset = SwitchLinguaDataset(
        data_dir=args.data_dir,
        language=args.language,
        num_examples=args.num_examples,
        skip_examples=args.skip_examples,
        sample_rate=16000,
        filter_unsupported=filter_unsupported,
    )

    if len(dataset) == 0:
        logging.error("No samples found. Check --data_dir and --language.")
        return 0

    eval_indices = list(range(len(dataset)))

    # Split indices across GPUs
    indices_per_gpu = len(eval_indices) // world_size
    start_idx = rank * indices_per_gpu
    end_idx = start_idx + indices_per_gpu if rank < world_size - 1 else len(eval_indices)
    my_indices = eval_indices[start_idx:end_idx]

    if is_main:
        logging.info(f"Total examples: {len(eval_indices)}, this GPU: {len(my_indices)}")

    results = []
    iterator = tqdm(my_indices, desc=f"GPU {rank}", disable=not is_main)

    for idx in iterator:
        sample = dataset[idx]
        if sample is None:
            continue

        try:
            generated = generate_transcription(model, processor, sample, args)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            logging.warning(f"Failed to generate for sample {idx}: {e}")
            continue
        except Exception as e:
            logging.warning(f"Failed to generate for sample {idx}: {e}")
            continue

        if args.two_step:
            try:
                generated = generate_refinement(model, processor, sample, generated, args)
            except Exception as e:
                logging.warning(f"Refinement failed for sample {idx}, using draft: {e}")

        pred = extract_answer(generated)

        # For raw models, truncate after the first ':' if present
        if args.raw_model_prompt and ":" in pred:
            pred = pred.split(":", 1)[1].strip()

        solution = sample.get("solution", "")
        ref = extract_answer(solution)

        # Language code for normalization
        lang_code = sample.get("lang_code", "en")
        pred_norm = _remove_sp(pred, lang_code)
        ref_norm = _remove_sp(ref, lang_code)

        cer = compute_cer(ref_norm, pred_norm)

        # Compute script hallucination rate (binary: 1 if any hallucination, 0 otherwise)
        script_hall_rate = 0.0
        lang_name = sample.get("language", "")
        lang_pair = _SWITCHLINGUA_LANG_TO_PAIR.get(lang_name, "")
        if lang_pair:
            allowed_scripts = _get_allowed_scripts(lang_pair)
            script_hall_rate = 1.0 if _compute_script_contamination(pred_norm, allowed_scripts) > 0 else 0.0

        result = {
            "index": idx,
            "uniq_id": sample.get("uniq_id", f"sample_{idx}"),
            "language": lang_name,
            "reference": ref,
            "prediction": pred,
            "ref_norm": ref_norm,
            "pred_norm": pred_norm,
            "cer": cer,
            "script_hall_rate": script_hall_rate,
        }
        results.append(result)

        if args.verbose and is_main:
            print(f"\n[{idx}] {result['uniq_id']} ({result['language']})")
            print(f"  Ref:  {ref[:100]}{'...' if len(ref) > 100 else ''}")
            print(f"  Pred: {pred[:100]}{'...' if len(pred) > 100 else ''}")
            print(f"  CER: {cer:.4f}, SHR: {script_hall_rate:.4f}")

        # Print intermediate results every 800 samples (on main process only)
        if is_main and len(results) % 100 == 0 and len(results) > 0:
            n = len(results)
            avg = sum(r["cer"] for r in results) / n
            lstats = {}
            for r in results:
                lstats.setdefault(r["language"], []).append(r["cer"])
            print(f"\n{'='*60}")
            print(f"INTERMEDIATE RESULTS ({n} samples on GPU 0, ~{n * world_size} total)")
            print(f"{'='*60}")
            print(f"Model:           {args.model_name_or_path}")
            avg_shr = sum(r["script_hall_rate"] for r in results) / n
            print(f"Average CER:           {avg:.4f} ({avg*100:.2f}%)")
            print(f"Script hall. rate:     {avg_shr:.4f} ({avg_shr*100:.2f}%)")
            print(f"{'-'*60}")
            print("Per-language breakdown:")
            for lang, cers in sorted(lstats.items()):
                lc = sum(cers)/len(cers)
                lang_results = [r for r in results if r["language"] == lang]
                lshr = sum(r["script_hall_rate"] for r in lang_results) / len(lang_results) if lang_results else 0
                print(f"  {lang}: CER={lc:.4f}, SHR={lshr:.4f} (n={len(cers)})")
            print(f"{'='*60}\n")

    # Save per-GPU results to disk (avoids NCCL allgather which deadlocks if any rank crashes)
    if world_size > 1:
        output_base = Path(args.output_file).stem if args.output_file else "eval_switchlingua"
        output_dir = Path(args.output_file).parent if args.output_file else Path("./outputs/eval_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        shard_path = output_dir / f"{output_base}_rank{rank}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        logging.info(f"Rank {rank}: saved {len(results)} results to {shard_path}")

        # Rank 0 polls for all shard files (no NCCL needed)
        if is_main:
            import time
            for r in range(world_size):
                rpath = output_dir / f"{output_base}_rank{r}.json"
                for _ in range(120):  # wait up to 60s per shard
                    if rpath.exists() and rpath.stat().st_size > 0:
                        break
                    time.sleep(0.5)

            results = []
            for r in range(world_size):
                rpath = output_dir / f"{output_base}_rank{r}.json"
                if rpath.exists():
                    with open(rpath) as f:
                        results.extend(json.load(f))
                    rpath.unlink()
                else:
                    logging.warning(f"Missing shard for rank {r}")

    # Compute aggregate metrics (only on main process)
    if is_main or world_size == 1:
        num_samples = len(results)

        if num_samples > 0:
            avg_cer = sum(r["cer"] for r in results) / num_samples

            # Per-language breakdown
            lang_stats = {}
            for r in results:
                lang = r["language"]
                if lang not in lang_stats:
                    lang_stats[lang] = {"cer": [], "script_hall_rate": []}
                lang_stats[lang]["cer"].append(r["cer"])
                lang_stats[lang]["script_hall_rate"].append(r["script_hall_rate"])
        else:
            avg_cer = 0
            lang_stats = {}

        # Print summary
        print("\n" + "=" * 60)
        print("SWITCHLINGUA EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model:           {args.model_name_or_path}")
        print(f"Language:        {args.language or 'all'}")
        print(f"Num samples:     {num_samples}")
        print(f"Skipped:         {args.skip_examples}")
        print(f"GPUs used:       {world_size}")
        print("-" * 60)
        avg_script_hall = sum(r["script_hall_rate"] for r in results) / num_samples if num_samples > 0 else 0
        print(f"Average CER:           {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
        print(f"Script hall. rate:     {avg_script_hall:.4f} ({avg_script_hall * 100:.2f}%)")

        # Per-language breakdown
        if len(lang_stats) > 1:
            print("-" * 60)
            print("Per-language breakdown:")
            for lang, stats in sorted(lang_stats.items()):
                lang_cer = sum(stats["cer"]) / len(stats["cer"])
                lang_shr = sum(stats["script_hall_rate"]) / len(stats["script_hall_rate"])
                print(f"  {lang}: CER={lang_cer:.4f}, SHR={lang_shr:.4f} (n={len(stats['cer'])})")
        elif len(lang_stats) == 1:
            lang = list(lang_stats.keys())[0]
            print(f"Language:              {lang} (n={len(lang_stats[lang]['cer'])})")

        print("=" * 60)

        # Save detailed results
        if args.output_file:
            output_data = {
                "model": args.model_name_or_path,
                "dataset": "switchlingua",
                "language": args.language,
                "num_samples": num_samples,
                "skip_examples": args.skip_examples,
                "avg_cer": avg_cer,
                "avg_script_hall_rate": avg_script_hall,
                "num_gpus": world_size,
                "timestamp": datetime.now().isoformat(),
                "per_language": {
                    lang: {
                        "avg_cer": sum(s["cer"]) / len(s["cer"]),
                        "avg_script_hall_rate": sum(s["script_hall_rate"]) / len(s["script_hall_rate"]),
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
