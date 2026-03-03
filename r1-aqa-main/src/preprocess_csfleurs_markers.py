#!/usr/bin/env python3
"""
Preprocess CS-FLEURS test sets to add ** markers around English words.

Uses FastText's language ID model for fast word-level language detection.
This makes test sets compatible with CS-WER computation (like xtts_train).

Usage:
    # Install fasttext first: pip install fasttext

    # Process all test subsets
    python preprocess_csfleurs_markers.py --data_dir /path/to/csfleurs_data

    # Process specific subset
    python preprocess_csfleurs_markers.py --data_dir /path/to/csfleurs_data --subset xtts_test1

    # Dry run (preview without saving)
    python preprocess_csfleurs_markers.py --data_dir /path/to/csfleurs_data --dry_run --num_examples 10
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# FastText language ID model
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("Warning: fasttext not installed. Run: pip install fasttext")


# Model URL and local path
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_PATH = os.path.expanduser("~/.cache/fasttext/lid.176.ftz")


def download_fasttext_model():
    """Download FastText language ID model if not present."""
    if os.path.exists(FASTTEXT_MODEL_PATH):
        return FASTTEXT_MODEL_PATH

    print(f"Downloading FastText language ID model...")
    os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, FASTTEXT_MODEL_PATH)
    print(f"Model saved to: {FASTTEXT_MODEL_PATH}")
    return FASTTEXT_MODEL_PATH


def load_fasttext_model():
    """Load FastText language ID model."""
    if not FASTTEXT_AVAILABLE:
        raise RuntimeError("fasttext not installed. Run: pip install fasttext")

    model_path = download_fasttext_model()
    # Suppress FastText warning about loading model
    fasttext.FastText.eprint = lambda x: None
    model = fasttext.load_model(model_path)
    return model


# Map CS-FLEURS language codes to FastText language codes
LANG_CODE_TO_FASTTEXT = {
    "ara": "ar", "cmn": "zh", "zho": "zh", "hin": "hi", "spa": "es",
    "fra": "fr", "deu": "de", "por": "pt", "rus": "ru", "jpn": "ja",
    "kor": "ko", "vie": "vi", "tha": "th", "ind": "id", "tur": "tr",
    "pol": "pl", "nld": "nl", "ita": "it", "eng": "en", "hun": "hu",
    "ces": "cs", "slk": "sk", "tel": "te",
}


def detect_word_language(word: str, model, lang_pair: str = "eng-eng") -> str:
    """
    Detect if a word belongs to the primary or secondary language.

    Args:
        word: Word to check
        model: FastText model
        lang_pair: Language pair like "ara-eng"

    Returns:
        "primary", "secondary", or "unknown"
    """
    clean = re.sub(r'[^\w]', '', word)
    if not clean or len(clean) < 2:
        return "unknown"

    # Parse language pair
    parts = lang_pair.lower().split("-")
    primary_code = LANG_CODE_TO_FASTTEXT.get(parts[0], parts[0])
    secondary_code = LANG_CODE_TO_FASTTEXT.get(parts[1], parts[1]) if len(parts) > 1 else "en"

    # Get top predictions
    predictions = model.predict(clean, k=10)

    # Find best match among the two allowed languages
    primary_score = 0.0
    secondary_score = 0.0

    for label, score in zip(predictions[0], predictions[1]):
        lang = label.replace('__label__', '')
        if lang == primary_code:
            primary_score = max(primary_score, score)
        elif lang == secondary_code:
            secondary_score = max(secondary_score, score)

    # Decide based on which language has higher score
    if secondary_score > primary_score:
        return "secondary"
    elif primary_score > 0:
        return "primary"
    else:
        return "unknown"


def split_mixed_script_token(token: str) -> List[str]:
    """
    Split a token that contains mixed scripts (e.g., Arabic prefix + Latin word).

    Example: "الkeyboard" -> ["ال", "keyboard"]
    """
    if not token:
        return [token]

    result = []
    current = []
    current_is_latin = None

    for char in token:
        # Check if character is Latin (A-Z, a-z, extended Latin)
        is_latin = ('\u0041' <= char <= '\u007A') or ('\u00C0' <= char <= '\u024F')

        if current_is_latin is None:
            current_is_latin = is_latin
            current.append(char)
        elif is_latin == current_is_latin:
            current.append(char)
        else:
            # Script change - save current and start new
            if current:
                result.append(''.join(current))
            current = [char]
            current_is_latin = is_latin

    if current:
        result.append(''.join(current))

    return result if result else [token]


def add_markers_to_text(text: str, model, lang_pair: str = "eng-eng") -> str:
    """
    Add ** markers around secondary language words/spans in text.

    Uses the language pair metadata to restrict detection to only those two languages.
    Also handles mixed-script tokens (e.g., "الkeyboard" -> "ال**keyboard**").

    Args:
        text: Original text (mixed language)
        model: FastText model
        lang_pair: Language pair like "ara-eng"

    Returns:
        Text with **secondary language spans** marked
    """
    if not text.strip():
        return text

    # Already has markers - skip
    if "**" in text:
        return text

    # Tokenize preserving whitespace
    raw_tokens = re.findall(r'\S+|\s+', text)

    # Expand mixed-script tokens
    tokens = []
    for tok in raw_tokens:
        if tok.isspace():
            tokens.append(tok)
        else:
            parts = split_mixed_script_token(tok)
            tokens.extend(parts)

    result = []
    secondary_span = []

    for token in tokens:
        if token.isspace():
            if secondary_span:
                secondary_span.append(token)
            else:
                result.append(token)
            continue

        word_lang = detect_word_language(token, model, lang_pair)

        if word_lang == "secondary":
            secondary_span.append(token)
        else:
            # Close any open secondary language span
            if secondary_span:
                span_text = ''.join(secondary_span).strip()
                if span_text:
                    result.append(f"**{span_text}**")
                # Add trailing whitespace if any
                trailing_ws = ''.join(t for t in secondary_span if t.isspace())
                if trailing_ws:
                    result.append(' ')
                secondary_span = []
            result.append(token)

    # Close final secondary span
    if secondary_span:
        span_text = ''.join(secondary_span).strip()
        if span_text:
            result.append(f"**{span_text}**")

    return ''.join(result)


def process_metadata_file(
    metadata_path: Path,
    model,
    output_path: Path = None,
    dry_run: bool = False,
    num_examples: int = None,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process a metadata.jsonl file and add markers.

    Uses language pair metadata to constrain detection to only the two
    languages in the pair (e.g., ara-eng only considers Arabic or English).

    Args:
        metadata_path: Path to metadata.jsonl
        model: FastText model
        output_path: Output path (defaults to same file with .marked.jsonl suffix)
        dry_run: If True, don't write output
        num_examples: Limit number of examples (for testing)
        verbose: Print examples

    Returns:
        Tuple of (total_processed, num_with_markers)
    """
    if output_path is None:
        output_path = metadata_path.with_suffix('.marked.jsonl')

    print(f"\nProcessing: {metadata_path}")

    processed = []
    num_with_markers = 0

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_examples and i >= num_examples:
                break

            entry = json.loads(line.strip())

            # Get the text field (could be 'text', 'transcript', or 'sentence')
            text_key = None
            for key in ['text', 'transcript', 'sentence']:
                if key in entry:
                    text_key = key
                    break

            if text_key is None:
                processed.append(entry)
                continue

            original_text = entry[text_key]

            # Get language pair for constrained detection
            lang_pair = entry.get('language', 'eng-eng')

            # Add markers using language-pair-constrained detection
            marked_text = add_markers_to_text(original_text, model, lang_pair)

            # Store both original and marked versions
            entry[f'{text_key}_original'] = original_text
            entry[text_key] = marked_text

            if "**" in marked_text:
                num_with_markers += 1

            if verbose and i < 5:
                print(f"\n  [{i}] Original: {original_text[:100]}...")
                print(f"      Marked:   {marked_text[:100]}...")

            processed.append(entry)

    total = len(processed)
    print(f"  Processed: {total} entries, {num_with_markers} with markers ({100*num_with_markers/total:.1f}%)")

    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in processed:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"  Saved to: {output_path}")
    else:
        print(f"  [DRY RUN] Would save to: {output_path}")

    return total, num_with_markers


def main():
    parser = argparse.ArgumentParser(description="Add ** markers to CS-FLEURS test sets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/Qwen2-Audio/csfleurs_data",
        help="Path to CS-FLEURS data directory",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["xtts_test1", "xtts_test2", "read_test", "mms_test", "all"],
        help="Subset to process (default: all test sets)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview without saving",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Limit number of examples per subset (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print example transformations",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite original metadata.jsonl (default: create .marked.jsonl)",
    )

    args = parser.parse_args()

    # Map subset names to paths
    subset_paths = {
        "xtts_test1": "xtts/test1/metadata.jsonl",
        "xtts_test2": "xtts/test2/metadata.jsonl",
        "read_test": "read/test/metadata.jsonl",
        "mms_test": "mms/test/metadata.jsonl",
    }

    # Determine which subsets to process
    if args.subset is None or args.subset == "all":
        subsets_to_process = list(subset_paths.keys())
    else:
        subsets_to_process = [args.subset]

    # Load model
    print("Loading FastText language ID model...")
    model = load_fasttext_model()
    print("Model loaded.")

    # Process each subset
    total_processed = 0
    total_with_markers = 0

    for subset in subsets_to_process:
        metadata_path = Path(args.data_dir) / subset_paths[subset]

        if not metadata_path.exists():
            print(f"\nSkipping {subset}: {metadata_path} not found")
            continue

        if args.in_place:
            output_path = metadata_path
        else:
            output_path = None  # Will use .marked.jsonl suffix

        processed, with_markers = process_metadata_file(
            metadata_path,
            model,
            output_path=output_path,
            dry_run=args.dry_run,
            num_examples=args.num_examples,
            verbose=args.verbose,
        )

        total_processed += processed
        total_with_markers += with_markers

    print(f"\n{'='*60}")
    print(f"Total: {total_processed} entries, {total_with_markers} with markers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
