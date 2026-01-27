"""
CS-FLEURS Dataset for Code-Switched Speech Recognition Training.

This dataset class handles the CS-FLEURS dataset from HuggingFace:
https://huggingface.co/datasets/byan/cs-fleurs

CS-FLEURS contains 113 unique code-switched language pairs across 52 languages
with 300 hours of speech data (both read and synthetic).

Subsets:
- read_test: 14 X-English pairs, 17 hours (read speech)
- xtts_train: 16 X-English pairs, 128 hours (generative TTS)
- xtts_test1: 16 X-English pairs, 36 hours (generative TTS)
- xtts_test2: 60 {Arabic, Chinese, Hindi, Spanish}-X pairs, 42 hours
- mms_test: 45 X-English pairs, 56 hours (concatenative TTS)
"""

import logging
from typing import Optional

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset


# Subset name to directory path mapping
SUBSET_TO_PATH = {
    "xtts_train": "xtts/train",
    "xtts_test1": "xtts/test1",
    "xtts_test2": "xtts/test2",
    "read_test": "read/test",
    "mms_test": "mms/test",
}

# ISO 639-3 language code to language name mapping
LANG_CODE_TO_NAME = {
    "ara": "Arabic",
    "cmn": "Chinese",
    "zho": "Chinese",
    "hin": "Hindi",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "por": "Portuguese",
    "rus": "Russian",
    "jpn": "Japanese",
    "kor": "Korean",
    "vie": "Vietnamese",
    "tha": "Thai",
    "ind": "Indonesian",
    "tur": "Turkish",
    "pol": "Polish",
    "nld": "Dutch",
    "ita": "Italian",
    "eng": "English",
}

# ISO 639-3 to Qwen2-Audio language token mapping
# Qwen2-Audio supports: en, zh, zh_tw, ar, de, es, fr, it, pt, ja, ko, ru
LANG_CODE_TO_TOKEN = {
    "ara": "<|ar|>",
    "cmn": "<|zh|>",
    "zho": "<|zh|>",
    "spa": "<|es|>",
    "fra": "<|fr|>",
    "deu": "<|de|>",
    "por": "<|pt|>",
    "rus": "<|ru|>",
    "jpn": "<|ja|>",
    "kor": "<|ko|>",
    "ita": "<|it|>",
    "eng": "<|en|>",
    # Languages without Qwen2 tokens - use empty string (no token)
    "hin": "",
    "vie": "",
    "tha": "",
    "ind": "",
    "tur": "",
    "pol": "",
    "nld": "",
    "hun": "",
    "ces": "",
}


def _get_language_name(lang_code: str) -> str:
    """Convert language code (e.g., 'ara-eng') to readable name."""
    if "-" in lang_code:
        # Format: "ara-eng" -> extract first language
        primary = lang_code.split("-")[0]
    elif "_" in lang_code:
        # Format: "arabic_english" -> extract first language
        primary = lang_code.split("_")[0]
    else:
        primary = lang_code
    return LANG_CODE_TO_NAME.get(primary, primary.capitalize())


def _get_language_token(lang_code: str) -> str:
    """Get Qwen2-Audio language token for a language code."""
    if "-" in lang_code:
        primary = lang_code.split("-")[0]
    elif "_" in lang_code:
        primary = lang_code.split("_")[0]
    else:
        primary = lang_code
    return LANG_CODE_TO_TOKEN.get(primary, "")


# Languages that don't use spaces between words
SPACELESS_LANGUAGES = {"jpn", "cmn", "zho", "tha", "yue"}


def _remove_spaces_for_language(text: str, lang_code: str) -> str:
    """
    Remove spaces from text for languages that don't use word spacing.
    Removes spaces between CJK characters and between CJK and Latin characters.
    Preserves spaces between Latin words only.
    """
    if lang_code not in SPACELESS_LANGUAGES:
        return text

    import re

    # CJK Unicode ranges
    cjk = '[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0e00-\u0e7f\u3000-\u303f\uff00-\uffef]'
    # Latin characters (basic + extended)
    latin = '[a-zA-Z\u00c0-\u00ff]'

    result = text

    # Loop until no more changes
    prev = None
    while prev != result:
        prev = result
        # Remove spaces between CJK characters
        result = re.sub(f'({cjk}) +({cjk})', r'\1\2', result)
        # Remove spaces between CJK and Latin (both directions)
        result = re.sub(f'({cjk}) +({latin})', r'\1\2', result)
        result = re.sub(f'({latin}) +({cjk})', r'\1\2', result)

    # Remove spaces between CJK and punctuation
    result = re.sub(f'({cjk}) +([,.!?;:，。！？；：、])', r'\1\2', result)
    result = re.sub(f'([,.!?;:，。！？；：、]) +({cjk})', r'\1\2', result)

    return result


def _extract_code_switch_entities(text: str, language: str = None) -> tuple:
    """
    Extract code-switched entities from CS-FLEURS text.

    The ** markers annotate English code-switched portions in the dataset.
    This function removes the ** markers and extracts entity list.

    For Japanese/Chinese, spaces are removed from non-English segments:
    e.g., "この マインド **focus** は"
    becomes "このマインド focus は"

    Args:
        text: Raw text with ** markers
        language: Language pair code (e.g., 'ara-eng') for space handling

    Returns:
        tuple: (clean_text, entity_list)
            - clean_text: Text with ** markers removed
            - entity_list: List of code-switched English phrases
    """
    import re
    import unicodedata

    def is_punctuation_only(s):
        """Check if string contains only punctuation characters."""
        return all(unicodedata.category(c).startswith('P') or c.isspace() for c in s)

    def strip_edge_punctuation(s):
        """Strip punctuation from start and end of string."""
        while s and unicodedata.category(s[0]).startswith('P'):
            s = s[1:]
        while s and unicodedata.category(s[-1]).startswith('P'):
            s = s[:-1]
        return s

    # Extract all text between ** markers (for entity list)
    raw_entities = re.findall(r'\*\*([^*]+)\*\*', text)
    # Clean entities: strip whitespace/punctuation, filter out punctuation-only
    entities = []
    for e in raw_entities:
        e = e.strip()
        e = strip_edge_punctuation(e)
        if e and not is_punctuation_only(e):
            entities.append(e)

    # Get primary language code
    if language and "-" in language:
        primary_lang = language.split("-")[0]
    elif language and "_" in language:
        primary_lang = language.split("_")[0]
    else:
        primary_lang = language or ""

    # Split text by ** markers to process each segment
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    result_parts = []

    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            # English code-switched segment - keep spaces, strip ** markers
            english_content = part[2:-2]
            result_parts.append(english_content)
        else:
            # Primary language segment - remove spaces if applicable
            processed = _remove_spaces_for_language(part, primary_lang)
            result_parts.append(processed)

    clean_text = ''.join(result_parts)

    return clean_text, entities


# CS-FLEURS Prompt Templates
# For code-switched ASR, we inform the model about the language pair
CSFLEURS_PROMPT_TEMPLATE = (
    "You are a speech transcription system for code-switched speech. "
    "The audio contains speech mixing {language} and English. "
    "Output ONLY the exact words spoken, preserving the language switches. "
    "Output the transcription in <answer> </answer>."
)

# Fallback without language info
CSFLEURS_BASIC_PROMPT_TEMPLATE = (
    "You are a speech transcription system for code-switched speech. "
    "Output ONLY the exact words spoken, preserving any language switches. "
    "Output the transcription in <answer> </answer>."
)


def _load_audio(audio_path: str, target_rate: int = 16000):
    """
    Load audio file and resample to target rate.
    Returns numpy array of audio samples.
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()


class CSFleursDataset(Dataset):
    """
    Dataset class for CS-FLEURS code-switched speech recognition.

    Args:
        subset: Dataset subset ('read_test', 'xtts_train', 'xtts_test1', 'xtts_test2', 'mms_test')
        language_pair: Optional filter for specific language pair (e.g., 'chinese_english')
        num_examples: Number of examples to load (None for all)
        max_audio_duration: Maximum audio duration in seconds (filter out longer)
        sample_rate: Target sample rate for audio
    """

    def __init__(
        self,
        subset: str = "xtts_train",
        language_pair: Optional[str] = None,
        num_examples: Optional[int] = None,
        max_audio_duration: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.subset = subset
        self.language_pair = language_pair
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate

        logging.info(f"Loading CS-FLEURS dataset: subset={subset}, language_pair={language_pair}")

        # Load dataset from HuggingFace
        try:
            self.hf_dataset = load_dataset("byan/cs-fleurs", subset, split="train")
        except Exception as e:
            logging.warning(f"Failed to load subset '{subset}' with split 'train': {e}")
            # Try without split specification
            self.hf_dataset = load_dataset("byan/cs-fleurs", subset)
            if hasattr(self.hf_dataset, 'keys'):
                # It's a DatasetDict, get the first available split
                split_name = list(self.hf_dataset.keys())[0]
                self.hf_dataset = self.hf_dataset[split_name]

        logging.info(f"Loaded {len(self.hf_dataset)} examples from CS-FLEURS {subset}")

        # Build sample list
        self.samples = []
        for idx, item in enumerate(self.hf_dataset):
            # Filter by language pair if specified
            if language_pair and item.get("language", "").lower() != language_pair.lower():
                continue

            # Filter by duration
            duration = item.get("duration", 0)
            if duration > max_audio_duration:
                continue

            self.samples.append({
                "index": idx,
                "text": item.get("text", ""),
                "language": item.get("language", "unknown"),
                "duration": duration,
                "speaker": item.get("speaker", ""),
                "id": item.get("id", f"sample_{idx}"),
            })

            if num_examples and len(self.samples) >= num_examples:
                break

        logging.info(f"Filtered to {len(self.samples)} samples (max_duration={max_audio_duration}s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        hf_idx = sample["index"]
        item = self.hf_dataset[hf_idx]

        # Load audio
        audio_info = item.get("audio", {})
        if isinstance(audio_info, dict):
            # HuggingFace audio format: {"path": ..., "array": ..., "sampling_rate": ...}
            audio = np.array(audio_info.get("array", []), dtype=np.float32)
            sr = audio_info.get("sampling_rate", 16000)

            # Resample if needed
            if sr != self.sample_rate:
                audio = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio).squeeze().numpy()
        else:
            # Fallback: try to load from path
            audio = _load_audio(str(audio_info), self.sample_rate)

        # Build prompt
        language = sample["language"]
        if language and language.lower() != "unknown":
            lang_name = _get_language_name(language)
            prompt_text = CSFLEURS_PROMPT_TEMPLATE.format(language=lang_name)
        else:
            prompt_text = CSFLEURS_BASIC_PROMPT_TEMPLATE

        # Extract code-switched entities
        raw_text = sample["text"]
        clean_text, entity_list = _extract_code_switch_entities(raw_text, language)

        # Build conversation format
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": ""},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        solution = f"<answer>{clean_text}</answer>"

        return {
            "prompt": prompt,
            "audio": audio,
            "solution": solution,
            "language": language,
            "uniq_id": sample["id"],
            "duration": sample["duration"],
            "speaker": sample["speaker"],
            # Code-switched English phrases for CGPR reward
            "entity_list": entity_list,
        }


class CSFleursDatasetLocal(Dataset):
    """
    Dataset class for CS-FLEURS loaded from local directory.

    Use this if you've cloned the dataset with git lfs.

    Args:
        data_dir: Path to cloned cs-fleurs directory
        subset: Dataset subset (xtts_train, xtts_test1, xtts_test2, read_test, mms_test)
        language_pair: Optional filter for specific language pair (e.g., 'ara-eng')
        num_examples: Number of examples to load
        max_audio_duration: Maximum audio duration in seconds
        sample_rate: Target sample rate
        stratify_languages: If True and language_pair is None, sample evenly across
            all language pairs instead of taking first N examples (default: True)
        filter_unsupported: If True, exclude languages not supported by Qwen2-Audio
            (hin, tur, pol, nld, hun, ces, vie, tha, ind) (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = "xtts_train",
        language_pair: Optional[str] = None,
        num_examples: Optional[int] = None,
        max_audio_duration: float = 30.0,
        sample_rate: int = 16000,
        stratify_languages: bool = True,
        filter_unsupported: bool = True,
    ):
        import json
        import random
        from collections import defaultdict
        from pathlib import Path

        self.data_dir = Path(data_dir)
        self.subset = subset
        self.language_pair = language_pair
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate

        # Map subset name to actual directory path
        subset_path = SUBSET_TO_PATH.get(subset, subset)
        subset_dir = self.data_dir / subset_path

        # Load metadata
        metadata_path = subset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            # Try alternative paths
            alt_paths = [
                self.data_dir / subset / "metadata.jsonl",
                self.data_dir / f"{subset}_metadata.jsonl",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    metadata_path = alt_path
                    subset_dir = alt_path.parent
                    break

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found. Tried: {metadata_path}, {alt_paths}. "
                f"Available subsets: {list(SUBSET_TO_PATH.keys())}"
            )

        logging.info(f"Loading CS-FLEURS from {metadata_path}")

        # Load all samples, grouped by language pair for stratification
        samples_by_lang = defaultdict(list)

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                # Filter by language pair if specified
                item_lang = item.get("language", "unknown").lower()
                if language_pair:
                    filter_lang = language_pair.lower()
                    if item_lang != filter_lang:
                        continue

                # Filter out unsupported languages if enabled
                # Qwen2-Audio only supports: ar, zh, es, fr, de, pt, ru, ja, ko, it, en
                if filter_unsupported:
                    UNSUPPORTED_LANGS = {"hin", "tur", "pol", "nld", "hun", "ces", "vie", "tha", "ind", "slk", "tel"}
                    primary_lang = item_lang.split("-")[0] if "-" in item_lang else item_lang
                    if primary_lang in UNSUPPORTED_LANGS:
                        continue

                # Filter by duration
                duration = item.get("duration", 0)
                if duration > max_audio_duration:
                    continue

                # Build audio path - file_name already includes "audio/" prefix
                audio_filename = item.get("file_name", "")
                audio_path = subset_dir / audio_filename

                sample = {
                    "audio_path": str(audio_path),
                    "text": item.get("text", ""),
                    "language": item.get("language", "unknown"),
                    "duration": duration,
                    "speaker": item.get("speaker", ""),
                    "id": item.get("id", audio_filename),
                }

                samples_by_lang[item_lang].append(sample)

        # Stratified sampling when training on all language pairs with limited examples
        if num_examples and not language_pair and stratify_languages and len(samples_by_lang) > 1:
            num_langs = len(samples_by_lang)
            samples_per_lang = num_examples // num_langs
            remainder = num_examples % num_langs

            logging.info(f"Stratified sampling: {num_examples} examples across {num_langs} language pairs")
            logging.info(f"  ~{samples_per_lang} examples per language pair")

            self.samples = []
            lang_counts = {}

            # Sort languages for reproducibility
            sorted_langs = sorted(samples_by_lang.keys())

            for i, lang in enumerate(sorted_langs):
                lang_samples = samples_by_lang[lang]
                # Distribute remainder to first few languages
                n_samples = samples_per_lang + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(lang_samples))

                # Shuffle and take n_samples
                random.seed(42)  # For reproducibility
                random.shuffle(lang_samples)
                self.samples.extend(lang_samples[:n_samples])
                lang_counts[lang] = n_samples

            # Shuffle final samples
            random.seed(42)
            random.shuffle(self.samples)

            logging.info(f"  Language distribution: {lang_counts}")
        else:
            # Original behavior: take first num_examples (or all if not specified)
            self.samples = []
            for lang_samples in samples_by_lang.values():
                self.samples.extend(lang_samples)

            if num_examples and len(self.samples) > num_examples:
                self.samples = self.samples[:num_examples]

        logging.info(f"Loaded {len(self.samples)} samples from CS-FLEURS {subset}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio = _load_audio(sample["audio_path"], self.sample_rate)

        # Build prompt
        language = sample["language"]
        if language and language.lower() != "unknown":
            lang_name = _get_language_name(language)
            prompt_text = CSFLEURS_PROMPT_TEMPLATE.format(language=lang_name)
        else:
            prompt_text = CSFLEURS_BASIC_PROMPT_TEMPLATE

        # Extract code-switched entities
        raw_text = sample["text"]
        clean_text, entity_list = _extract_code_switch_entities(raw_text, language)

        # Build conversation format
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": sample["audio_path"]},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        solution = f"<answer>{clean_text}</answer>"

        return {
            "prompt": prompt,
            "audio": audio,
            "solution": solution,
            "language": language,
            "uniq_id": sample["id"],
            "duration": sample["duration"],
            "speaker": sample["speaker"],
            # Code-switched English phrases for CGPR reward
            "entity_list": entity_list,
        }
