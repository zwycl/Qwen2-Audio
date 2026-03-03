"""
SwitchLingua Dataset for Code-Switched Speech Recognition Evaluation.

SwitchLingua (NeurIPS 2025) is a large-scale multilingual code-switching dataset
with 80+ hours of audio across 11 language pairs (all X-English).

Data layout (local directory):
  switchlingua_audio/
    Arabic.csv, Cantonese.csv, ..., Spanish.csv    # per-language CSVs
    Arabic/, Cantonese/, ..., Spanish/              # per-language audio dirs

CSV columns vary by language:
  - All have: file_name, text
  - Most have: topic, tense, perspective, cs_ratio, gender, age, ...
  - Column name quirks: "education Level" vs "education_level", etc.

Audio quirks:
  - CSV may list "0_1.m4a" but file on disk is "0_1" (no extension)
  - Format is M4A (AAC-LC), files are playable without extension
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torchaudio
from torch.utils.data import Dataset

from .csfleurs_dataset import (
    CSFLEURS_PROMPT_TEMPLATE,
    CSFLEURS_BASIC_PROMPT_TEMPLATE,
    _remove_spaces_for_language,
)

# Language name -> ISO 639 code for CER normalization
SWITCHLINGUA_LANG_TO_CODE = {
    "Arabic": "ar",
    "Cantonese": "yue",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Mandarin": "zh",
    "Chinese": "zh",
    "Russian": "ru",
    "Spanish": "es",
}

# All expected language directories
SWITCHLINGUA_LANGUAGES = [
    "Arabic", "Cantonese", "French", "German", "Hindi",
    "Italian", "Japanese", "Korean", "Mandarin", "Russian", "Spanish",
]

# Languages to exclude from evaluation:
# - Cantonese (yue), Hindi (hi): not supported by Qwen2-Audio
# - Spanish: mislabeled in upstream dataset (681/688 samples are actually Korean)
SWITCHLINGUA_UNSUPPORTED = {"Cantonese", "Hindi", "Spanish"}

# Languages that don't use spaces between words (for CER normalization)
SPACELESS_LANG_CODES = {"ja", "zh", "yue"}


def _load_audio(audio_path: str, target_rate: int = 16000):
    """Load audio file and resample to target rate. Returns numpy array."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError:
        # torchcodec can't handle mp3 or some m4a files; fall back to ffmpeg
        import subprocess, io, torch
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(target_rate), "-ac", "1", "-"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {audio_path}: {result.stderr.decode()[:200]}")
        pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()


def _find_audio_file(audio_dir: Path, file_name: str) -> Optional[Path]:
    """
    Find audio file, handling extension mismatches.

    Tries in order:
      1. Exact file_name as given
      2. file_name with extension stripped
      3. file_name with .m4a extension added (if missing)
      4. file_name with .wav extension added (if missing)
    """
    # 1. Exact match
    path = audio_dir / file_name
    if path.exists():
        return path

    # 2. Strip extension
    stem = Path(file_name).stem
    path = audio_dir / stem
    if path.exists():
        return path

    # 3. Add .m4a
    path = audio_dir / f"{stem}.m4a"
    if path.exists():
        return path

    # 4. Add .wav
    path = audio_dir / f"{stem}.wav"
    if path.exists():
        return path

    return None


class SwitchLinguaDataset(Dataset):
    """
    Dataset class for SwitchLingua code-switched speech recognition.

    Args:
        data_dir: Path to switchlingua_audio directory
        language: Filter by language name (e.g., "Arabic", "Mandarin") or None for all
        num_examples: Number of examples to load (None for all)
        skip_examples: Number of examples to skip from start (for train/eval split)
        sample_rate: Target sample rate for audio
        filter_unsupported: If True, exclude languages not supported by Qwen2-Audio
            (Cantonese, Hindi) (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        language: Optional[str] = None,
        num_examples: Optional[int] = None,
        skip_examples: int = 0,
        sample_rate: int = 16000,
        filter_unsupported: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate

        # Determine which languages to load
        if language and language.lower() != "all":
            languages = [language]
        else:
            languages = SWITCHLINGUA_LANGUAGES

        # Filter out languages not supported by Qwen2-Audio
        if filter_unsupported:
            filtered = [l for l in languages if l not in SWITCHLINGUA_UNSUPPORTED]
            if len(filtered) < len(languages):
                removed = [l for l in languages if l in SWITCHLINGUA_UNSUPPORTED]
                logging.info(f"Filtering unsupported languages: {removed}")
            languages = filtered

        logging.info(f"Loading SwitchLingua from: {self.data_dir}")
        logging.info(f"Languages: {languages}")

        self.samples = []
        skipped = 0
        missing_audio = 0

        for lang_name in languages:
            csv_path = self.data_dir / f"{lang_name}.csv"
            if not csv_path.exists():
                logging.warning(f"CSV not found for {lang_name}: {csv_path}")
                continue

            audio_dir = self.data_dir / lang_name

            # Read CSV
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_name = row.get("file_name", "").strip()
                    text = row.get("text", "").strip()

                    if not file_name or not text:
                        continue

                    # Resolve audio path
                    audio_path = _find_audio_file(audio_dir, file_name)
                    if audio_path is None:
                        missing_audio += 1
                        continue

                    # Normalize column names (handle "education Level" etc.)
                    first_lang = row.get("first_language") or row.get("first Language") or lang_name
                    second_lang = row.get("second_language") or row.get("second Language") or "English"

                    self.samples.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "language": lang_name,
                        "first_language": first_lang,
                        "second_language": second_lang,
                        "file_name": file_name,
                        "topic": row.get("topic", ""),
                        "cs_ratio": row.get("cs_ratio", ""),
                    })

        logging.info(f"Found {len(self.samples)} samples total, {missing_audio} missing audio files")

        # Apply skip/limit
        if skip_examples > 0:
            if skip_examples >= len(self.samples):
                logging.warning(
                    f"skip_examples ({skip_examples}) >= total samples ({len(self.samples)}), "
                    f"dataset will be empty"
                )
            self.samples = self.samples[skip_examples:]
            logging.info(f"Skipped {skip_examples} examples, {len(self.samples)} remaining")

        if num_examples is not None and len(self.samples) > num_examples:
            self.samples = self.samples[:num_examples]

        logging.info(f"SwitchLingua dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio (skip corrupt files)
        try:
            audio = _load_audio(sample["audio_path"], self.sample_rate)
        except Exception as e:
            logging.warning(f"Skipping corrupt audio: {sample['audio_path']}: {e}")
            return None
        duration = len(audio) / self.sample_rate

        # Build prompt
        lang_name = sample["language"]
        first_lang = sample["first_language"]
        second_lang = sample["second_language"]

        if first_lang and second_lang:
            prompt_text = CSFLEURS_PROMPT_TEMPLATE.format(
                lang1=first_lang, lang2=second_lang
            )
        else:
            prompt_text = CSFLEURS_BASIC_PROMPT_TEMPLATE

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

        text = sample["text"]
        solution = f"<answer>{text}</answer>"

        # Language code for CER normalization
        lang_code = SWITCHLINGUA_LANG_TO_CODE.get(lang_name, "en")

        return {
            "prompt": prompt,
            "audio": audio,
            "solution": solution,
            "language": lang_name,
            "lang_code": lang_code,
            "uniq_id": f"{lang_name}_{sample['file_name']}",
            "duration": duration,
            "speaker": "",
            "entity_list": [],
            "raw_text": text,
            "chunk_start": 0.0,
            "chunk_end": duration,
        }
