"""
Download ContextASR-Bench Dataset

Downloads the ContextASR-Bench dataset including audio files from Hugging Face.
Audio files are stored in tar archives that will be extracted.
Total size: ~97GB

Usage:
    # Download everything
    python download_contextasr.py

    # Download only English
    python download_contextasr.py --language English

    # Download only Dialogue config
    python download_contextasr.py --config ContextASR-Dialogue

    # Specify output directory
    python download_contextasr.py --output-dir ./data/contextasr
"""

import argparse
import os
import json
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from datasets import load_dataset


DATASET_ID = "MrSupW/ContextASR-Bench"
DEFAULT_OUTPUT_DIR = "./contextasr_data"


def download_and_extract_tar(repo_id: str, tar_path: str, output_dir: Path) -> int:
    """Download and extract a tar file from the repository."""
    try:
        # Download tar file
        local_tar = hf_hub_download(
            repo_id=repo_id,
            filename=tar_path,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        # Extract tar file
        extract_dir = output_dir / Path(tar_path).parent
        extract_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0
        with tarfile.open(local_tar, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Extract to the audio directory
                    tar.extract(member, extract_dir)
                    extracted_count += 1

        # Optionally remove tar file after extraction to save space
        # os.remove(local_tar)

        return extracted_count
    except Exception as e:
        print(f"Error with {tar_path}: {e}")
        return 0


def download_dataset(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    config: str = None,
    language: str = None,
    num_workers: int = 2,
    keep_tar: bool = True,
):
    """
    Download the ContextASR-Bench dataset with audio files.

    Args:
        output_dir: Directory to save the dataset
        config: Optional config filter ("ContextASR-Dialogue" or "ContextASR-Speech")
        language: Optional language filter ("English" or "Mandarin")
        num_workers: Number of parallel download workers
        keep_tar: Keep tar files after extraction
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs = ["ContextASR-Dialogue", "ContextASR-Speech"]
    languages = ["English", "Mandarin"]

    if config:
        configs = [config]
    if language:
        languages = [language]

    print(f"Downloading ContextASR-Bench to {output_path.absolute()}")
    print(f"Configs: {configs}")
    print(f"Languages: {languages}")
    print(f"Workers: {num_workers}")
    print("-" * 60)

    # Get list of tar files
    all_files = list_repo_files(DATASET_ID, repo_type="dataset")
    tar_files = [f for f in all_files if f.endswith('.tar')]

    for cfg in configs:
        for lang in languages:
            print(f"\n{'='*60}")
            print(f"Processing {cfg}/{lang}...")
            print("="*60)

            # Filter tar files for this config/language
            prefix = f"audio/{cfg}/{lang}/"
            config_tars = [f for f in tar_files if f.startswith(prefix)]

            if not config_tars:
                print(f"No tar files found for {cfg}/{lang}")
                continue

            print(f"Found {len(config_tars)} tar archives to download")

            # Download and extract tar files
            total_extracted = 0
            for tar_file in tqdm(config_tars, desc="Downloading & extracting"):
                count = download_and_extract_tar(DATASET_ID, tar_file, output_path)
                total_extracted += count

            print(f"Extracted {total_extracted} audio files")

            # Load and save metadata
            try:
                print("Loading metadata from HuggingFace...")
                ds = load_dataset(
                    DATASET_ID,
                    cfg,
                    split=lang,
                    streaming=True,
                )

                metadata_dir = output_path / cfg / lang
                metadata_dir.mkdir(parents=True, exist_ok=True)
                metadata_file = metadata_dir / "metadata.json"

                metadata = []
                for item in tqdm(ds, desc="Saving metadata"):
                    audio_path = item.get("audio", "")
                    entry = {
                        "uniq_id": item.get("uniq_id"),
                        "language": item.get("language"),
                        "duration": item.get("duration"),
                        "text": item.get("text"),
                        "entity_list": list(item.get("entity_list", [])),
                        "audio_path": audio_path,
                    }

                    if cfg == "ContextASR-Dialogue":
                        entry["dialogue"] = list(item.get("dialogue", []))
                        entry["movie_name"] = item.get("movie_name", "")
                    else:
                        entry["domain_label"] = item.get("domain_label", "")

                    metadata.append(entry)

                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                print(f"Saved {len(metadata)} entries to: {metadata_file}")

            except Exception as e:
                print(f"Error saving metadata: {e}")
                import traceback
                traceback.print_exc()

    # Create index file
    index_file = output_path / "index.json"
    index = {
        "dataset": DATASET_ID,
        "configs": configs,
        "languages": languages,
        "base_path": str(output_path.absolute()),
    }
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Dataset saved to: {output_path.absolute()}")
    print(f"Index file: {index_file}")
    print("=" * 60)


def download_full_snapshot(output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Download the entire dataset using snapshot_download.
    Note: This downloads tar files - you'll need to extract them manually.
    """
    print(f"Downloading full dataset snapshot to {output_dir}...")
    print("This will download ~97GB of data (tar archives)...")
    print("Note: You'll need to extract the tar files manually after download.")

    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )

    # Extract all tar files
    print("\nExtracting tar archives...")
    for tar_path in Path(path).rglob("*.tar"):
        print(f"Extracting {tar_path}...")
        extract_dir = tar_path.parent
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)

    print(f"Download and extraction complete! Dataset saved to: {path}")
    return path


def list_available_files():
    """List all files available in the repository."""
    print(f"Listing files in {DATASET_ID}...")
    files = list_repo_files(DATASET_ID, repo_type="dataset")

    tar_files = [f for f in files if f.endswith('.tar')]
    other_files = [f for f in files if not f.endswith('.tar')]

    print(f"\nMetadata/other files ({len(other_files)}):")
    for f in other_files:
        print(f"  {f}")

    print(f"\nTar archives ({len(tar_files)}):")
    for f in tar_files:
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ContextASR-Bench dataset")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=["ContextASR-Dialogue", "ContextASR-Speech"],
        default=None,
        help="Download only specific config"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["English", "Mandarin"],
        default=None,
        help="Download only specific language"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel download workers (default: 2)"
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Use snapshot_download to download entire dataset at once"
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List available files in the repository without downloading"
    )

    args = parser.parse_args()

    if args.list_files:
        list_available_files()
    elif args.snapshot:
        download_full_snapshot(args.output_dir)
    else:
        download_dataset(
            output_dir=args.output_dir,
            config=args.config,
            language=args.language,
            num_workers=args.workers,
        )
