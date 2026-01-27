"""
ContextASR-Bench Dataset for GRPO training.

Loads the ContextASR-Bench dataset for Contextual Automatic Speech Recognition
training with WER-based rewards. This dataset includes entity hints that help
the model recognize specific names/terms in the audio.

Dataset: MrSupW/ContextASR-Bench
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torchaudio
from torch.utils.data import Dataset


# Context ASR Prompt Templates
# NOTE: The GRPO trainer works best with user-only prompts (no system role).
# We embed the entity context directly in the user prompt.
CONTEXT_ASR_PROMPT_TEMPLATE = (
    "You are a speech transcription system. Output ONLY the exact words spoken. "
    "The following names/terms may appear: {entity_str}. Use correct spelling for these terms. "
    "Output the transcription in <answer> </answer>."
)

# Fallback for samples without entities
NO_CONTEXT_PROMPT_TEMPLATE = (
    "You are a speech transcription system. Output ONLY the exact words spoken. "
    "Output the transcription in <answer> </answer>."
)

# Two-step training: Refinement prompt templates (used for second pass)
# The draft transcription from first pass is included to help refine
REFINEMENT_PROMPT_TEMPLATE = (
    "You are a speech transcription system. A previous transcription attempt produced: \"{draft_transcription}\". "
    "Listen to the audio again and correct any errors. "
    "The following names/terms may appear: {entity_str}. Use correct spelling for these terms. "
    "Output the corrected transcription in <answer> </answer>."
)

REFINEMENT_NO_CONTEXT_PROMPT_TEMPLATE = (
    "You are a speech transcription system. A previous transcription attempt produced: \"{draft_transcription}\". "
    "Listen to the audio again and correct any errors. "
    "Output the corrected transcription in <answer> </answer>."
)


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
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform[0].numpy()


class ContextASRDataset(Dataset):
    """
    Dataset for ContextASR-Bench GRPO training.

    Expected directory structure (from download_contextasr.py):
        contextasr_data/
        ├── ContextASR-Dialogue/
        │   ├── English/
        │   │   ├── metadata.json
        │   │   └── audio/
        │   └── Mandarin/
        └── ContextASR-Speech/
            ├── English/
            └── Mandarin/

    Each item in metadata.json has:
        - uniq_id: unique identifier
        - text: ground truth transcription
        - entity_list: list of entities/terms that may appear
        - audio_path: relative path to audio file
        - duration: audio duration in seconds
    """

    def __init__(
        self,
        data_dir: str,
        config: str = "ContextASR-Dialogue",
        language: str = "English",
        sample_rate: int = 16000,
        max_duration: float = 180.0,  # ContextASR has long audio (99-253s)
        min_duration: float = 0.5,
        max_audio_chunk: float = 30.0,  # Chunk long audio for model input
        num_examples: Optional[int] = None,
        include_no_context: bool = False,
        no_entities: bool = False,
    ):
        """
        Initialize ContextASR dataset.

        Args:
            data_dir: Path to contextasr_data directory
            config: Dataset config ("ContextASR-Dialogue" or "ContextASR-Speech")
            language: Language ("English" or "Mandarin")
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds (skip longer)
            min_duration: Minimum audio duration in seconds (skip shorter)
            max_audio_chunk: Maximum chunk size for model input (30s default)
            num_examples: Maximum number of examples to load (None for all)
            include_no_context: Include samples without entity list
            no_entities: Do not add entities to prompt (use simple prompt)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.config = config
        self.language = language
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_audio_chunk = max_audio_chunk
        self.include_no_context = include_no_context
        self.no_entities = no_entities

        # Load metadata
        metadata_path = self.data_dir / config / language / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Filter samples and create chunks from dialogue timing
        self.samples = []
        skipped_duration = 0
        skipped_missing = 0
        skipped_no_entities = 0
        total_chunks = 0

        for entry in self.metadata:
            # Check if we've reached the limit
            if num_examples and len(self.samples) >= num_examples:
                break

            # Check duration
            duration = entry.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                skipped_duration += 1
                continue

            # Check entity list
            entity_list = entry.get("entity_list", [])
            if not entity_list and not include_no_context:
                skipped_no_entities += 1
                continue

            # Check audio file exists
            audio_path = entry.get("audio_path", "")
            if not audio_path:
                skipped_missing += 1
                continue

            # Handle different audio path formats
            possible_paths = [
                self.data_dir / audio_path,
                self.data_dir / config / language / audio_path,
                self.data_dir / config / language / "audio" / Path(audio_path).name,
            ]

            full_audio_path = None
            for p in possible_paths:
                if p.exists():
                    full_audio_path = p
                    break

            if full_audio_path is None:
                skipped_missing += 1
                continue

            # Create chunks from dialogue timing (if available)
            dialogue = entry.get("dialogue", [])
            full_text = entry.get("text", "")

            if dialogue and len(dialogue) > 0:
                # Group dialogue turns into chunks of max_audio_chunk seconds
                chunks = self._create_dialogue_chunks(dialogue, max_audio_chunk)
                for chunk in chunks:
                    if num_examples and len(self.samples) >= num_examples:
                        break
                    self.samples.append({
                        "audio_path": str(full_audio_path),
                        "transcript": chunk["text"],
                        "entity_list": entity_list,
                        "start_time": chunk["start"],
                        "end_time": chunk["end"],
                        "duration": chunk["end"] - chunk["start"],
                        "uniq_id": f"{entry.get('uniq_id', '')}_{chunk['chunk_id']}",
                    })
                    total_chunks += 1
            else:
                # No dialogue timing - create chunks with proportionally estimated transcript
                chunk_id = 0
                current_time = 0.0
                while current_time < duration:
                    if num_examples and len(self.samples) >= num_examples:
                        break

                    chunk_end = min(current_time + max_audio_chunk, duration)
                    chunk_duration = chunk_end - current_time

                    # Proportionally estimate transcript for this chunk
                    start_ratio = current_time / duration
                    end_ratio = chunk_end / duration
                    start_char = int(len(full_text) * start_ratio)
                    end_char = int(len(full_text) * end_ratio)
                    chunk_text = full_text[start_char:end_char].strip()

                    if chunk_text:  # Only add if there's text
                        self.samples.append({
                            "audio_path": str(full_audio_path),
                            "transcript": chunk_text,
                            "entity_list": entity_list,
                            "start_time": current_time,
                            "end_time": chunk_end,
                            "duration": chunk_duration,
                            "uniq_id": f"{entry.get('uniq_id', '')}_{chunk_id}",
                        })
                        total_chunks += 1

                    current_time = chunk_end
                    chunk_id += 1

        logging.info(
            f"ContextASR {config}/{language}: loaded {len(self.samples)} samples "
            f"({total_chunks} chunks from dialogue, "
            f"skipped: {skipped_duration} duration, {skipped_missing} missing, "
            f"{skipped_no_entities} no_entities)"
        )

    def _create_dialogue_chunks(self, dialogue: List[dict], max_chunk_duration: float) -> List[dict]:
        """Create chunks from dialogue turns, grouping turns to fit within max duration."""
        chunks = []
        current_chunk = {"start": 0.0, "end": 0.0, "text": "", "chunk_id": 0}

        for turn in dialogue:
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            turn_text = turn.get("text", "")

            # If this turn would exceed max duration, save current chunk and start new one
            if current_chunk["text"] and (turn_end - current_chunk["start"]) > max_chunk_duration:
                chunks.append(current_chunk)
                current_chunk = {
                    "start": turn_start,
                    "end": turn_end,
                    "text": turn_text,
                    "chunk_id": len(chunks),
                }
            else:
                # Add turn to current chunk
                if not current_chunk["text"]:
                    current_chunk["start"] = turn_start
                current_chunk["end"] = turn_end
                current_chunk["text"] = (current_chunk["text"] + " " + turn_text).strip()

        # Add final chunk
        if current_chunk["text"]:
            chunks.append(current_chunk)

        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Try to load the sample, skip corrupted files
        max_retries = 10
        for retry in range(max_retries):
            current_index = (index + retry) % len(self.samples)
            sample = self.samples[current_index]

            try:
                # Load full audio
                full_audio = _load_audio(sample["audio_path"], self.sample_rate)

                # Extract chunk based on timing
                start_sample = int(sample["start_time"] * self.sample_rate)
                end_sample = int(sample["end_time"] * self.sample_rate)
                audio = full_audio[start_sample:end_sample]

                # Ensure audio is not empty
                if len(audio) < self.sample_rate * 0.1:  # Less than 0.1 second
                    raise ValueError(f"Audio chunk too short: {len(audio)} samples")

                # Build prompt (with or without entity context)
                entity_list = sample["entity_list"]
                if entity_list and not self.no_entities:
                    entity_str = ", ".join(entity_list)
                    prompt_text = CONTEXT_ASR_PROMPT_TEMPLATE.format(entity_str=entity_str)
                else:
                    prompt_text = NO_CONTEXT_PROMPT_TEMPLATE

                # Format for GRPO trainer (user-only prompt, matching AfriSpeech format)
                # The trainer expects: prompt, audio, solution, and any extra fields
                # Each chunk is an independent sample with its own audio slice and transcript
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio_url": sample["audio_path"]},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    "audio": audio,
                    "solution": f"<answer>{sample['transcript']}</answer>",
                    "language": "en" if self.language == "English" else "zh",
                    "entity_list": entity_list,
                    "dataset_name": "ContextASR",
                    # Chunk metadata for debugging
                    "uniq_id": sample["uniq_id"],
                    "chunk_start": sample["start_time"],
                    "chunk_end": sample["end_time"],
                }
            except Exception as e:
                logging.warning(f"Failed to load audio {sample['audio_path']}: {e}, trying next sample")
                continue

        raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries starting from index {index}")


class ContextASRDatasetFromHF(Dataset):
    """
    Load ContextASR-Bench directly from Hugging Face datasets.

    This is an alternative to the file-based approach that streams
    directly from HuggingFace. Creates chunks from dialogue timing.
    """

    def __init__(
        self,
        config: str = "ContextASR-Dialogue",
        language: str = "English",
        sample_rate: int = 16000,
        max_duration: float = 180.0,  # ContextASR has long audio (99-253s)
        min_duration: float = 0.5,
        max_audio_chunk: float = 30.0,  # Chunk long audio for model input
        num_examples: Optional[int] = None,
        include_no_context: bool = False,
        no_entities: bool = False,
    ):
        """
        Initialize ContextASR dataset from HuggingFace.

        Args:
            config: Dataset config ("ContextASR-Dialogue" or "ContextASR-Speech")
            language: Language split ("English" or "Mandarin")
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            max_audio_chunk: Maximum chunk size for model input
            num_examples: Maximum number of examples to load (None for all)
            include_no_context: Include samples without entity list
            no_entities: Do not add entities to prompt (use simple prompt)
        """
        super().__init__()
        from datasets import load_dataset
        import torch

        self.sample_rate = sample_rate
        self.language = language
        self.include_no_context = include_no_context
        self.max_audio_chunk = max_audio_chunk
        self.no_entities = no_entities

        # Load dataset from HuggingFace
        logging.info(f"Loading ContextASR {config}/{language} from HuggingFace...")
        dataset = load_dataset(
            "MrSupW/ContextASR-Bench",
            config,
            split=language,
            trust_remote_code=True,
        )

        # Filter samples and create chunks
        self.samples = []
        total_chunks = 0

        for sample in dataset:
            if num_examples and len(self.samples) >= num_examples:
                break

            duration = sample.get("duration", 0)
            if duration < min_duration or duration > max_duration:
                continue

            entity_list = list(sample.get("entity_list", []))
            if not entity_list and not include_no_context:
                continue

            audio_data = sample.get("audio", {})
            if not audio_data:
                continue

            # Get audio array and resample if needed
            if isinstance(audio_data, dict):
                audio_array = np.array(audio_data["array"])
                orig_sr = audio_data.get("sampling_rate", 16000)
            else:
                audio_array = np.array(audio_data)
                orig_sr = 16000

            if orig_sr != sample_rate:
                audio_tensor = torch.tensor(audio_array).unsqueeze(0).float()
                audio_array = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=sample_rate
                )(audio_tensor)[0].numpy()

            # Create chunks from dialogue timing
            dialogue = list(sample.get("dialogue", []))
            uniq_id = sample.get("uniq_id", f"sample_{len(self.samples)}")
            if dialogue:
                chunks = self._create_dialogue_chunks(dialogue, max_audio_chunk)
                for chunk in chunks:
                    if num_examples and len(self.samples) >= num_examples:
                        break

                    start_sample = int(chunk["start"] * sample_rate)
                    end_sample = int(chunk["end"] * sample_rate)
                    chunk_audio = audio_array[start_sample:end_sample]

                    if len(chunk_audio) < sample_rate * 0.1:  # Skip very short chunks
                        continue

                    self.samples.append({
                        "audio": chunk_audio,
                        "transcript": chunk["text"],
                        "entity_list": entity_list,
                        # Chunk metadata
                        "uniq_id": f"{uniq_id}_{chunk['chunk_id']}",
                        "chunk_start": chunk["start"],
                        "chunk_end": chunk["end"],
                    })
                    total_chunks += 1
            else:
                # No dialogue timing - use first chunk
                chunk_samples = int(max_audio_chunk * sample_rate)
                chunk_audio = audio_array[:chunk_samples]
                chunk_duration = len(chunk_audio) / sample_rate
                self.samples.append({
                    "audio": chunk_audio,
                    "transcript": sample.get("text", ""),
                    "entity_list": entity_list,
                    # Chunk metadata
                    "uniq_id": uniq_id,
                    "chunk_start": 0.0,
                    "chunk_end": chunk_duration,
                })

        logging.info(f"ContextASR {config}/{language}: loaded {len(self.samples)} samples ({total_chunks} chunks)")

    def _create_dialogue_chunks(self, dialogue: List[Dict], max_chunk_duration: float) -> List[Dict]:
        """Create chunks from dialogue turns, grouping turns to fit within max duration."""
        chunks = []
        current_chunk = {"start": 0.0, "end": 0.0, "text": "", "chunk_id": 0}

        for turn in dialogue:
            turn_start = turn.get("start", 0.0)
            turn_end = turn.get("end", 0.0)
            turn_text = turn.get("text", "")

            if current_chunk["text"] and (turn_end - current_chunk["start"]) > max_chunk_duration:
                chunks.append(current_chunk)
                current_chunk = {
                    "start": turn_start,
                    "end": turn_end,
                    "text": turn_text,
                    "chunk_id": len(chunks),
                }
            else:
                if not current_chunk["text"]:
                    current_chunk["start"] = turn_start
                current_chunk["end"] = turn_end
                current_chunk["text"] = (current_chunk["text"] + " " + turn_text).strip()

        if current_chunk["text"]:
            chunks.append(current_chunk)

        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Audio is already preprocessed and chunked
        audio_array = sample["audio"]

        # Build prompt (with or without entity context)
        entity_list = sample["entity_list"]
        if entity_list and not self.no_entities:
            entity_str = ", ".join(entity_list)
            prompt_text = CONTEXT_ASR_PROMPT_TEMPLATE.format(entity_str=entity_str)
        else:
            prompt_text = NO_CONTEXT_PROMPT_TEMPLATE

        # Format for GRPO trainer (user-only prompt, matching AfriSpeech format)
        # Each chunk is an independent sample with its own audio slice and transcript
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": ""},  # Will be handled by audio array
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "audio": audio_array,
            "solution": f"<answer>{sample['transcript']}</answer>",
            "language": "en" if self.language == "English" else "zh",
            "entity_list": entity_list,
            "dataset_name": "ContextASR",
            # Chunk metadata for debugging
            "uniq_id": sample.get("uniq_id", f"chunk_{index}"),
            "chunk_start": sample.get("chunk_start", 0.0),
            "chunk_end": sample.get("chunk_end", 0.0),
        }
