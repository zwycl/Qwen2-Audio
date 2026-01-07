"""
AfriSpeech ASR Dataset for GRPO training.

Loads the AfriSpeech-200 dataset for Automatic Speech Recognition training
with WER-based rewards.
"""

import json
import logging
import os
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset


def _load_audio(audio_path: str, target_rate: int = 16000):
    """
    Load and resample audio file.

    Args:
        audio_path: Path to audio file
        target_rate: Target sample rate (default 16000)

    Returns:
        numpy array of audio samples
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)
    return waveform[0].numpy()


class AfriSpeechASRDataset(Dataset):
    """
    Dataset for AfriSpeech ASR training with GRPO.

    Expected directory structure (from download_afrispeech.py):
        afrispeech_data/
        ├── train/
        │   ├── metadata.json
        │   └── audio/
        │       ├── xxx.wav
        │       └── ...
        ├── dev/
        └── test/

    Each item in metadata.json has:
        - audio_id: unique identifier
        - audio_path: relative path to audio file
        - transcript: ground truth transcription
        - accent: speaker accent
        - duration: audio duration in seconds
    """

    ASR_PROMPT_TEMPLATE = "Transcribe the following audio. Output the transcription in <answer> </answer>."

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        accent_filter: str = None,
        num_examples: int = None,
    ):
        """
        Initialize AfriSpeech ASR dataset.

        Args:
            data_dir: Path to afrispeech_data directory
            split: Dataset split ("train", "dev", "test")
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds (skip longer)
            min_duration: Minimum audio duration in seconds (skip shorter)
            accent_filter: Optional filter for specific accent
            num_examples: Maximum number of examples to load (None for all)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration

        # Load metadata
        metadata_path = self.data_dir / split / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Filter samples
        self.samples = []
        skipped_duration = 0
        skipped_missing = 0
        skipped_accent = 0

        for entry in self.metadata:
            # Check if we've reached the limit
            if num_examples and len(self.samples) >= num_examples:
                break

            # Check duration
            duration = entry.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                skipped_duration += 1
                continue

            # Check accent filter
            if accent_filter and entry.get("accent", "").lower() != accent_filter.lower():
                skipped_accent += 1
                continue

            # Check audio file exists
            audio_path = entry.get("audio_path", "")
            if not audio_path:
                skipped_missing += 1
                continue

            full_audio_path = self.data_dir / audio_path
            if not full_audio_path.exists():
                skipped_missing += 1
                continue

            # Add valid sample
            self.samples.append({
                "audio_path": str(full_audio_path),
                "transcript": entry.get("transcript", ""),
                "accent": entry.get("accent", ""),
                "duration": duration,
                "audio_id": entry.get("audio_id", ""),
            })

        logging.info(
            f"AfriSpeech {split}: loaded {len(self.samples)} samples "
            f"(skipped: {skipped_duration} duration, {skipped_missing} missing, {skipped_accent} accent)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Try to load the sample, skip corrupted files
        max_retries = 10
        for retry in range(max_retries):
            current_index = (index + retry) % len(self.samples)
            sample = self.samples[current_index]

            try:
                # Load audio
                audio = _load_audio(sample["audio_path"], self.sample_rate)

                # Format for GRPO trainer
                # The trainer expects: prompt, audio, solution, and any extra fields
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio_url": sample["audio_path"]},
                                {"type": "text", "text": self.ASR_PROMPT_TEMPLATE},
                            ],
                        }
                    ],
                    "audio": audio,
                    "solution": f"<answer>{sample['transcript']}</answer>",
                    "language": "en",  # AfriSpeech is English with African accents
                    "accent": sample["accent"],
                    "dataset_name": "AfriSpeech",
                }
            except Exception as e:
                logging.warning(f"Failed to load audio {sample['audio_path']}: {e}, trying next sample")
                continue

        raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries starting from index {index}")


class AfriSpeechASRDatasetFromHF(Dataset):
    """
    Load AfriSpeech directly from Hugging Face datasets.

    This is an alternative to the file-based approach that streams
    directly from HuggingFace.
    """

    ASR_PROMPT_TEMPLATE = "Transcribe the following audio. Output the transcription in <answer> </answer>."

    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        accent_filter: str = None,
        num_examples: int = None,
    ):
        """
        Initialize AfriSpeech dataset from HuggingFace.

        Args:
            split: Dataset split ("train", "dev", "test")
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            accent_filter: Optional filter for specific accent
            num_examples: Maximum number of examples to load (None for all)
        """
        super().__init__()
        from datasets import load_dataset

        self.sample_rate = sample_rate

        # Load dataset from HuggingFace
        logging.info(f"Loading AfriSpeech {split} from HuggingFace...")
        dataset = load_dataset("intronhealth/afrispeech-200", split=split)

        # Filter samples
        self.samples = []
        for i, sample in enumerate(dataset):
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
                "duration": duration,
            })

        logging.info(f"AfriSpeech {split}: loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Get audio array (already loaded by datasets)
        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        orig_sr = audio_data["sampling_rate"]

        # Resample if needed
        if orig_sr != self.sample_rate:
            import torchaudio
            import torch

            audio_tensor = torch.tensor(audio_array).unsqueeze(0)
            audio_tensor = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )(audio_tensor)
            audio_array = audio_tensor[0].numpy()

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": ""},  # Will be handled by audio array
                        {"type": "text", "text": self.ASR_PROMPT_TEMPLATE},
                    ],
                }
            ],
            "audio": audio_array,
            "solution": f"<answer>{sample['transcript']}</answer>",
            "language": "en",
            "accent": sample["accent"],
            "dataset_name": "AfriSpeech",
        }
