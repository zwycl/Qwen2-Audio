"""
Download AfriSpeech-200 Dataset

Downloads the AfriSpeech-200 dataset from Hugging Face including audio files.
Total size: ~120GB, ~200 hours of speech from 120 African accents.

Dataset: https://huggingface.co/datasets/intronhealth/afrispeech-200

Usage:
    # Download everything
    python download_afrispeech.py

    # Download only test split
    python download_afrispeech.py --split test

    # Download specific accent
    python download_afrispeech.py --accent yoruba

    # Specify output directory
    python download_afrispeech.py --output-dir ./data/afrispeech
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import HfApi, hf_hub_download
import pandas as pd


DATASET_ID = "intronhealth/afrispeech-200"
DEFAULT_OUTPUT_DIR = "./afrispeech_data"


def download_dataset(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    split: str = None,
    accent: str = None,
    max_samples: int = None,
    save_audio: bool = True,
):
    """
    Download the AfriSpeech-200 dataset with audio files.

    Args:
        output_dir: Directory to save the dataset
        split: Optional split filter ("train", "dev", "test")
        accent: Optional accent filter (e.g., "yoruba", "igbo", "swahili")
        max_samples: Maximum samples to download per split (for testing)
        save_audio: Whether to save audio files locally
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"]
    if split:
        splits = [split]

    print(f"Downloading AfriSpeech-200 to {output_path.absolute()}")
    print(f"Splits: {splits}")
    if accent:
        print(f"Accent filter: {accent}")
    if max_samples:
        print(f"Max samples per split: {max_samples}")
    print("-" * 60)

    api = HfApi()
    all_files = api.list_repo_files(DATASET_ID, repo_type="dataset")

    total_samples = 0
    total_duration = 0

    for split_name in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print("="*60)

        try:
            # Download transcript CSV for this split
            transcript_file = f"transcripts/{split_name}.csv"
            print(f"Downloading transcripts: {transcript_file}")

            local_csv = hf_hub_download(
                repo_id=DATASET_ID,
                filename=transcript_file,
                repo_type="dataset"
            )
            df = pd.read_csv(local_csv)
            print(f"Loaded {len(df)} entries from {transcript_file}")

            # Filter by accent if specified
            if accent:
                df = df[df["accent"].str.lower() == accent.lower()]
                print(f"Filtered to accent '{accent}': {len(df)} samples")

            # Limit samples if specified
            if max_samples and len(df) > max_samples:
                df = df.head(max_samples)
                print(f"Limited to {max_samples} samples")

            # Create directories
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            audio_dir = split_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            # Build audio_id to row mapping (using the actual filename from audio_paths)
            audio_id_to_row = {}
            for _, row in df.iterrows():
                audio_path = row.get("audio_paths", "")
                if audio_path:
                    audio_filename = Path(audio_path).stem  # Get filename without extension
                    audio_id_to_row[audio_filename] = row

            # Find and download audio tar files for this split
            if save_audio:
                # Get unique accents in this dataset
                accents_in_df = df["accent"].str.lower().str.replace(" ", "-").str.replace("(", "").str.replace(")", "").unique()

                # Find relevant tar files
                tar_files = []
                for f in all_files:
                    if f.endswith('.tar.gz') and f'/{split_name}/' in f:
                        # Check if this accent is in our filtered data
                        tar_accent = f.split('/')[1]  # e.g., audio/yoruba/test/...
                        if accent is None or tar_accent.lower() == accent.lower():
                            tar_files.append(f)

                print(f"Found {len(tar_files)} audio tar files to process")

                # Process each tar file
                extracted_count = 0
                for tar_path in tqdm(tar_files, desc="Downloading audio files"):
                    try:
                        local_tar = hf_hub_download(
                            repo_id=DATASET_ID,
                            filename=tar_path,
                            repo_type="dataset"
                        )

                        # Extract matching audio files
                        with tarfile.open(local_tar, 'r:gz') as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('.wav'):
                                    # Extract audio_id from path (the hash before .wav)
                                    audio_id = Path(member.name).stem

                                    if audio_id in audio_id_to_row:
                                        # Extract this file
                                        audio_filename = f"{audio_id}.wav"
                                        target_path = audio_dir / audio_filename

                                        if not target_path.exists():
                                            # Extract to target location
                                            f = tar.extractfile(member)
                                            if f:
                                                with open(target_path, 'wb') as out:
                                                    out.write(f.read())
                                                extracted_count += 1

                                        if max_samples and extracted_count >= max_samples:
                                            break

                    except Exception as e:
                        print(f"Error processing {tar_path}: {e}")
                        continue

                    if max_samples and extracted_count >= max_samples:
                        break

                print(f"Extracted {extracted_count} audio files")

            # Build metadata
            metadata = []
            for _, row in df.iterrows():
                original_audio_path = row.get("audio_paths", "")
                audio_file_id = Path(original_audio_path).stem if original_audio_path else row.get("audio_ids", "")
                audio_filename = f"{audio_file_id}.wav"
                audio_file_path = audio_dir / audio_filename

                entry = {
                    "audio_id": row.get("audio_ids", ""),
                    "audio_file_id": audio_file_id,
                    "speaker_id": row.get("user_ids", ""),
                    "transcript": row.get("transcript", ""),
                    "duration": row.get("duration", 0),
                    "accent": row.get("accent", ""),
                    "gender": row.get("gender", ""),
                    "age_group": row.get("age_group", ""),
                    "domain": row.get("domain", ""),
                    "country": row.get("country", ""),
                    "audio_path": f"{split_name}/audio/{audio_filename}" if (save_audio and audio_file_path.exists()) else "",
                }
                metadata.append(entry)
                total_duration += entry["duration"]

            # Save metadata
            metadata_file = split_dir / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(metadata)} entries to: {metadata_file}")
            total_samples += len(metadata)

        except Exception as e:
            print(f"Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create index file
    index_file = output_path / "index.json"
    index = {
        "dataset": DATASET_ID,
        "splits": splits,
        "accent_filter": accent,
        "total_samples": total_samples,
        "total_duration_hours": total_duration / 3600,
        "base_path": str(output_path.absolute()),
    }
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Dataset saved to: {output_path.absolute()}")
    print(f"Index file: {index_file}")
    print("=" * 60)


def list_accents():
    """List all available accents in the dataset."""
    print("Loading dataset to list accents...")

    # Download accents.json which contains stats for all accents
    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="accents.json",
        repo_type="dataset"
    )

    with open(local_path) as f:
        accents_data = json.load(f)

    # Filter out 'all' which is the aggregate
    accents = [k for k in accents_data.keys() if k != 'all']

    print(f"\nAvailable accents ({len(accents)}):")
    for accent in sorted(accents):
        stats = accents_data[accent]
        total_clips = stats.get('num_clips', 0)
        duration_hrs = stats.get('duration', 0) / 3600
        print(f"  - {accent}: {total_clips} clips, {duration_hrs:.1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AfriSpeech-200 dataset")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "dev", "test"],
        default=None,
        help="Download only specific split"
    )
    parser.add_argument(
        "--accent", "-a",
        type=str,
        default=None,
        help="Download only specific accent (e.g., yoruba, igbo, swahili)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum samples to download per split (for testing)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (metadata only)"
    )
    parser.add_argument(
        "--list-accents",
        action="store_true",
        help="List available accents and exit"
    )

    args = parser.parse_args()

    if args.list_accents:
        list_accents()
    else:
        download_dataset(
            output_dir=args.output_dir,
            split=args.split,
            accent=args.accent,
            max_samples=args.max_samples,
            save_audio=not args.no_audio,
        )
