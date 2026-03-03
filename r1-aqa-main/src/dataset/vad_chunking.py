"""
VAD-based audio chunking.

This module implements Voice Activity Detection (VAD) based chunking.

The approach:
1. Use VAD to detect speech segments
2. Merge short segments to form chunks
3. Ensure no chunk exceeds max_duration (default 30s)
4. Align transcripts to chunk boundaries
"""

import logging
import warnings
import os
import sys
from contextlib import contextmanager

# Suppress NNPACK warnings (harmless - just means CPU optimizations unavailable)
warnings.filterwarnings("ignore", message=".*NNPACK.*")


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide C++ warnings like NNPACK."""
    stderr_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(stderr_fd), 'w') as old_stderr:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr.fileno(), stderr_fd)
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def get_vad_model():
    """
    Load Silero VAD model.

    Returns:
        Tuple of (model, get_speech_timestamps function, read_audio function)
    """
    try:
        import torch
        with suppress_stderr():
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
        get_speech_timestamps = utils[0]
        return model, get_speech_timestamps
    except Exception as e:
        logger.warning(f"Failed to load Silero VAD: {e}. Falling back to energy-based VAD.")
        return None, None


def energy_based_vad(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    energy_threshold: float = 0.01,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
) -> List[Dict[str, float]]:
    """
    Simple energy-based VAD as fallback.

    Args:
        audio: Audio waveform as numpy array
        sample_rate: Sample rate
        frame_duration_ms: Frame duration in milliseconds
        energy_threshold: Minimum energy to consider as speech (relative to max)
        min_speech_duration_ms: Minimum speech segment duration
        min_silence_duration_ms: Minimum silence duration to split segments

    Returns:
        List of speech segments with 'start' and 'end' times in seconds
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        return [{'start': 0.0, 'end': len(audio) / sample_rate}]

    # Compute frame energies
    energies = []
    for i in range(num_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)

    energies = np.array(energies)
    max_energy = np.max(energies) if len(energies) > 0 else 1.0

    # Threshold for speech detection
    threshold = max_energy * energy_threshold

    # Find speech frames
    is_speech = energies > threshold

    # Convert to segments
    segments = []
    in_speech = False
    start_frame = 0

    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)

    silence_count = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            # Start of speech
            in_speech = True
            start_frame = i
            silence_count = 0
        elif not speech and in_speech:
            silence_count += 1
            if silence_count >= min_silence_frames:
                # End of speech segment
                end_frame = i - silence_count
                if end_frame - start_frame >= min_speech_frames:
                    segments.append({
                        'start': start_frame * frame_duration_ms / 1000,
                        'end': end_frame * frame_duration_ms / 1000
                    })
                in_speech = False
                silence_count = 0
        elif speech and in_speech:
            silence_count = 0

    # Handle last segment
    if in_speech:
        end_frame = num_frames
        if end_frame - start_frame >= min_speech_frames:
            segments.append({
                'start': start_frame * frame_duration_ms / 1000,
                'end': end_frame * frame_duration_ms / 1000
            })

    # If no segments found, return the whole audio
    if not segments:
        return [{'start': 0.0, 'end': len(audio) / sample_rate}]

    return segments


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int = 16000,
    vad_model=None,
    get_speech_timestamps=None,
) -> List[Dict[str, float]]:
    """
    Detect speech segments using VAD.

    Args:
        audio: Audio waveform as numpy array (mono, float32)
        sample_rate: Sample rate (should be 16000 for Silero VAD)
        vad_model: Pre-loaded Silero VAD model (optional)
        get_speech_timestamps: Silero's get_speech_timestamps function (optional)

    Returns:
        List of speech segments with 'start' and 'end' times in seconds
    """
    import torch

    # Try Silero VAD first
    if vad_model is not None and get_speech_timestamps is not None:
        try:
            # Silero VAD expects torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Get speech timestamps (suppress NNPACK warnings)
            with suppress_stderr():
                speech_timestamps = get_speech_timestamps(
                    audio_tensor,
                    vad_model,
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    speech_pad_ms=30,
                    return_seconds=True,
                )

            if speech_timestamps:
                return speech_timestamps
        except Exception as e:
            logger.warning(f"Silero VAD failed: {e}. Falling back to energy-based VAD.")

    # Fallback to energy-based VAD
    return energy_based_vad(audio, sample_rate)


def merge_segments_to_chunks(
    segments: List[Dict[str, float]],
    max_chunk_duration: float = 30.0,
    min_chunk_duration: float = 0.5,
    merge_threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Merge VAD segments into chunks that don't exceed max_chunk_duration.

    Following the paper's approach:
    - Merge short resulting chunks
    - Ensure no segment exceeds 30 seconds

    Args:
        segments: List of VAD speech segments
        max_chunk_duration: Maximum chunk duration in seconds
        min_chunk_duration: Minimum chunk duration in seconds
        merge_threshold: Maximum gap between segments to merge (seconds)

    Returns:
        List of merged chunks with 'start', 'end', and 'chunk_id'
    """
    if not segments:
        return []

    chunks = []
    current_chunk = {
        'start': segments[0]['start'],
        'end': segments[0]['end'],
        'chunk_id': 0
    }

    for segment in segments[1:]:
        potential_duration = segment['end'] - current_chunk['start']

        # Merge if combined duration doesn't exceed max
        if potential_duration <= max_chunk_duration:
            current_chunk['end'] = segment['end']
        else:
            # Save current chunk if long enough
            if current_chunk['end'] - current_chunk['start'] >= min_chunk_duration:
                chunks.append(current_chunk)

            # Start new chunk
            current_chunk = {
                'start': segment['start'],
                'end': segment['end'],
                'chunk_id': len(chunks)
            }

    # Add final chunk
    if current_chunk['end'] - current_chunk['start'] >= min_chunk_duration:
        chunks.append(current_chunk)

    # Handle chunks that are too long by splitting them
    final_chunks = []
    for chunk in chunks:
        duration = chunk['end'] - chunk['start']
        if duration > max_chunk_duration:
            # Split into smaller chunks
            num_splits = int(np.ceil(duration / max_chunk_duration))
            split_duration = duration / num_splits
            for i in range(num_splits):
                final_chunks.append({
                    'start': chunk['start'] + i * split_duration,
                    'end': chunk['start'] + (i + 1) * split_duration,
                    'chunk_id': len(final_chunks)
                })
        else:
            chunk['chunk_id'] = len(final_chunks)
            final_chunks.append(chunk)

    return final_chunks


def align_transcript_to_chunks(
    chunks: List[Dict[str, float]],
    full_text: str,
    total_duration: float,
    dialogue: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Align transcript text to VAD-derived chunks.

    If dialogue timing is available, use it for precise alignment.
    Otherwise, estimate text proportionally based on duration.

    Args:
        chunks: List of VAD-derived chunks with 'start', 'end', 'chunk_id'
        full_text: Full transcript text
        total_duration: Total audio duration in seconds
        dialogue: Optional list of dialogue turns with timing info

    Returns:
        List of chunks with added 'text' field
    """
    if not chunks:
        return []

    aligned_chunks = []

    if dialogue and len(dialogue) > 0:
        # Use dialogue timing for alignment
        for chunk in chunks:
            chunk_text_parts = []

            for turn in dialogue:
                turn_start = turn.get('start', 0.0)
                turn_end = turn.get('end', 0.0)
                turn_text = turn.get('text', '')

                # Check for overlap between chunk and dialogue turn
                overlap_start = max(chunk['start'], turn_start)
                overlap_end = min(chunk['end'], turn_end)

                if overlap_end > overlap_start:
                    # There's overlap
                    turn_duration = turn_end - turn_start
                    if turn_duration > 0:
                        # Calculate what portion of the turn falls in this chunk
                        overlap_ratio = (overlap_end - overlap_start) / turn_duration

                        if overlap_ratio > 0.5:
                            # More than half of the turn is in this chunk
                            chunk_text_parts.append(turn_text)
                        elif overlap_ratio > 0.1:
                            # Partial overlap - estimate text portion
                            words = turn_text.split()
                            num_words = int(len(words) * overlap_ratio)
                            if turn_start >= chunk['start']:
                                # Turn starts in this chunk
                                chunk_text_parts.append(' '.join(words[:max(1, num_words)]))
                            else:
                                # Turn ends in this chunk
                                chunk_text_parts.append(' '.join(words[-max(1, num_words):]))

            aligned_chunks.append({
                **chunk,
                'text': ' '.join(chunk_text_parts).strip()
            })
    else:
        # Proportional text alignment based on duration
        for chunk in chunks:
            start_ratio = chunk['start'] / total_duration if total_duration > 0 else 0
            end_ratio = chunk['end'] / total_duration if total_duration > 0 else 1

            start_char = int(len(full_text) * start_ratio)
            end_char = int(len(full_text) * end_ratio)

            # Try to align to word boundaries
            # Find nearest space before start
            while start_char > 0 and full_text[start_char] != ' ':
                start_char -= 1
            # Find nearest space after end
            while end_char < len(full_text) and full_text[end_char] != ' ':
                end_char += 1

            chunk_text = full_text[start_char:end_char].strip()

            aligned_chunks.append({
                **chunk,
                'text': chunk_text
            })

    return aligned_chunks


def create_vad_chunks(
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_chunk_duration: float = 30.0,
    min_chunk_duration: float = 0.5,
    full_text: str = "",
    dialogue: Optional[List[Dict]] = None,
    vad_model=None,
    get_speech_timestamps=None,
) -> List[Dict]:
    """
    Main function to create VAD-based chunks from audio.

    This implements the following approach:
    1. Segment with VAD
    2. Merge short chunks
    3. Ensure max 30s chunks
    4. Align transcripts

    Args:
        audio: Audio waveform as numpy array
        sample_rate: Sample rate
        max_chunk_duration: Maximum chunk duration (default 30s as in paper)
        min_chunk_duration: Minimum chunk duration
        full_text: Full transcript for alignment
        dialogue: Optional dialogue timing info
        vad_model: Pre-loaded VAD model
        get_speech_timestamps: VAD timestamp function

    Returns:
        List of chunks with 'start', 'end', 'chunk_id', 'text'
    """
    total_duration = len(audio) / sample_rate

    # Step 1: Detect speech segments using VAD
    segments = detect_speech_segments(
        audio,
        sample_rate,
        vad_model,
        get_speech_timestamps
    )

    logger.debug(f"VAD detected {len(segments)} speech segments")

    # Step 2: Merge segments into chunks
    chunks = merge_segments_to_chunks(
        segments,
        max_chunk_duration=max_chunk_duration,
        min_chunk_duration=min_chunk_duration,
    )

    logger.debug(f"Merged into {len(chunks)} chunks")

    # Step 3: Align transcript to chunks
    aligned_chunks = align_transcript_to_chunks(
        chunks,
        full_text,
        total_duration,
        dialogue,
    )

    # Filter out chunks with empty text
    aligned_chunks = [c for c in aligned_chunks if c.get('text', '').strip()]

    # Re-number chunk IDs
    for i, chunk in enumerate(aligned_chunks):
        chunk['chunk_id'] = i

    return aligned_chunks


def _get_dist_info():
    """Get rank and world_size from torch.distributed if initialized."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except ImportError:
        pass
    return 0, 1


def parallel_vad_chunk_files(file_items, vad_chunker, sample_rate=16000, load_audio_fn=None):
    """
    Distribute VAD processing across all distributed training ranks.

    In multi-GPU training (torchrun), each rank processes its shard of audio
    files, then results are gathered via all_gather_object so every rank
    gets the complete set of VAD chunks.

    Falls back to single-process mode if torch.distributed is not initialized.

    Args:
        file_items: List of dicts, each with:
            - index: int, original position for reordering after gather
            - audio_path: str, path to audio file
            - full_text: str, full transcript
            - dialogue: list or None, dialogue timing info
        vad_chunker: VADChunker instance (model loaded once per rank)
        sample_rate: int, audio sample rate
        load_audio_fn: callable(audio_path, sample_rate) -> np.ndarray

    Returns:
        List of dicts sorted by original index, each with:
            - index: int, original position
            - chunks: list of chunk dicts, or None on failure
    """
    rank, world_size = _get_dist_info()

    # Each rank processes its shard
    my_indices = list(range(rank, len(file_items), world_size))
    my_results = []

    for count, i in enumerate(my_indices):
        item = file_items[i]
        if count % 50 == 0:
            logger.info(
                f"[Rank {rank}] VAD progress: {count}/{len(my_indices)} files "
                f"({100 * count / max(1, len(my_indices)):.0f}%)"
            )
        try:
            audio = load_audio_fn(item["audio_path"], sample_rate)
            chunks = vad_chunker.chunk_audio(
                audio=audio,
                sample_rate=sample_rate,
                full_text=item.get("full_text", ""),
                dialogue=item.get("dialogue"),
            )
        except Exception as e:
            logger.warning(f"[Rank {rank}] VAD failed for {item.get('audio_path', '?')}: {e}")
            chunks = None
        my_results.append({"index": item["index"], "chunks": chunks})

    logger.info(f"[Rank {rank}] VAD processing done: {len(my_results)} files")

    if world_size <= 1:
        my_results.sort(key=lambda x: x["index"])
        return my_results

    # Multi-process: gather results from all ranks
    import torch.distributed as dist

    gathered = [None] * world_size
    dist.all_gather_object(gathered, my_results)

    # Flatten and sort by original index
    all_results = []
    for rank_results in gathered:
        all_results.extend(rank_results)
    all_results.sort(key=lambda x: x["index"])

    if rank == 0:
        total_failures = sum(1 for r in all_results if r["chunks"] is None)
        logger.info(
            f"Distributed VAD complete: {len(all_results)} files processed "
            f"across {world_size} ranks ({total_failures} failures)"
        )

    return all_results


class VADChunker:
    """
    Reusable VAD chunker class that loads the model once.

    Usage:
        chunker = VADChunker()
        chunks = chunker.chunk_audio(audio, sample_rate, full_text, dialogue)
    """

    def __init__(self, max_chunk_duration: float = 30.0, min_chunk_duration: float = 0.5):
        """
        Initialize VAD chunker.

        Args:
            max_chunk_duration: Maximum chunk duration in seconds
            min_chunk_duration: Minimum chunk duration in seconds
        """
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.vad_model = None
        self.get_speech_timestamps = None
        self._load_vad_model()

    def _load_vad_model(self):
        """Load VAD model on initialization."""
        try:
            self.vad_model, self.get_speech_timestamps = get_vad_model()
            if self.vad_model is not None:
                logger.info("Loaded Silero VAD model")
            else:
                logger.info("Using energy-based VAD (Silero not available)")
        except Exception as e:
            logger.warning(f"Failed to load VAD model: {e}")

    def chunk_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        full_text: str = "",
        dialogue: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Chunk audio using VAD.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate
            full_text: Full transcript
            dialogue: Optional dialogue timing info

        Returns:
            List of chunks with timing and text
        """
        return create_vad_chunks(
            audio=audio,
            sample_rate=sample_rate,
            max_chunk_duration=self.max_chunk_duration,
            min_chunk_duration=self.min_chunk_duration,
            full_text=full_text,
            dialogue=dialogue,
            vad_model=self.vad_model,
            get_speech_timestamps=self.get_speech_timestamps,
        )
