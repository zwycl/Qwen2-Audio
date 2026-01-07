"""
Contextual ASR Gradio App with Qwen2-Audio

A Gradio web interface demonstrating contextual ASR using the Qwen2-Audio model
with examples from the ContextASR-Bench dataset.

Usage:
    python context_asr_gradio.py
    python context_asr_gradio.py --checkpoint Qwen/Qwen2-Audio-7B-Instruct --server-port 7860

    # With local dataset (faster, works offline)
    python context_asr_gradio.py --data-dir ./contextasr_data
"""

import difflib
import gradio as gr
import librosa
import torch
import numpy as np
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

DEFAULT_CKPT_PATH = "Qwen/Qwen2-Audio-7B-Instruct"
DEFAULT_DATA_DIR = "./contextasr_data"

# Audio chunking settings
MAX_AUDIO_SECONDS = 30  # Qwen2-Audio works best under 30s
OVERLAP_SECONDS = 1  # Overlap between chunks to avoid cutting words

# Global variables for model and processor
model = None
processor = None
dataset_cache = {}
local_data_dir = None


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Run with CPU only"
    )
    parser.add_argument(
        "--server-port", type=int, default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0",
        help="Server name"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--data-dir", "-d", type=str, default=None,
        help="Path to local ContextASR-Bench data (downloaded with download_contextasr.py)"
    )
    return parser.parse_args()


def load_model(checkpoint: str, cpu_only: bool = False):
    """Load the Qwen2-Audio model and processor."""
    global model, processor

    device_map = "cpu" if cpu_only else "auto"
    print(f"Loading model from {checkpoint} with device_map={device_map}...")

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    print("Model loaded successfully!")


def load_examples_from_local(config: str, split: str, num_examples: int = 50):
    """Load examples from locally downloaded dataset."""
    if local_data_dir is None:
        return None

    data_path = Path(local_data_dir) / config / split
    metadata_file = data_path / "metadata.json"

    if not metadata_file.exists():
        print(f"Local metadata not found: {metadata_file}")
        return None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        examples = []
        audio_found = 0
        for i, item in enumerate(metadata):
            if i >= num_examples:
                break

            rel_audio_path = item.get("audio_path", "")
            full_audio_path = Path(local_data_dir) / rel_audio_path

            audio_exists = full_audio_path.exists()
            if audio_exists:
                audio_found += 1

            example = {
                "id": item.get("uniq_id", f"example_{i}"),
                "language": item.get("language", "English"),
                "duration": item.get("duration", 0),
                "text": item.get("text", ""),
                "entity_list": item.get("entity_list", []),
                "audio_path": str(full_audio_path) if audio_exists else None,
                "audio": None,
            }

            if config == "ContextASR-Dialogue":
                example["dialogue"] = item.get("dialogue", [])
                example["movie_name"] = item.get("movie_name", "")
            else:
                example["domain_label"] = item.get("domain_label", "")

            examples.append(example)

        print(f"Loaded {len(examples)} examples from local storage ({audio_found} with audio): {data_path}")
        return examples

    except Exception as e:
        print(f"Error loading local data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_examples_from_hf(config: str, split: str, num_examples: int = 50):
    """Load examples from Hugging Face dataset."""
    cache_key = f"{config}_{split}"

    if cache_key not in dataset_cache:
        print(f"Loading dataset from HuggingFace: {config}/{split}...")
        try:
            from datasets import load_dataset
            ds = load_dataset("MrSupW/ContextASR-Bench", config, split=split, trust_remote_code=True)
            dataset_cache[cache_key] = ds
            print(f"Loaded {len(ds)} examples from HuggingFace")
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            return []

    ds = dataset_cache[cache_key]
    examples = []

    for i, item in enumerate(ds):
        if i >= num_examples:
            break

        example = {
            "id": item.get("uniq_id", f"example_{i}"),
            "language": item.get("language", "English"),
            "duration": item.get("duration", 0),
            "text": item.get("text", ""),
            "entity_list": item.get("entity_list", []),
            "audio": item.get("audio", {}),
            "audio_path": None,
        }

        if config == "ContextASR-Dialogue":
            example["dialogue"] = item.get("dialogue", [])
            example["movie_name"] = item.get("movie_name", "")
        else:
            example["domain_label"] = item.get("domain_label", "")

        examples.append(example)

    return examples


def load_examples(config: str = "ContextASR-Dialogue", split: str = "English", num_examples: int = 50):
    """Load examples from ContextASR-Bench dataset (local first, then HuggingFace)."""
    cache_key = f"{config}_{split}_examples"

    if cache_key in dataset_cache:
        cached = dataset_cache[cache_key]
        return cached[:num_examples] if len(cached) > num_examples else cached

    examples = load_examples_from_local(config, split, num_examples)

    if examples is None:
        examples = load_examples_from_hf(config, split, num_examples)

    if examples:
        dataset_cache[cache_key] = examples

    return examples


def get_audio_from_example(example: dict) -> Optional[tuple]:
    """Extract audio data from a dataset example (local file or HuggingFace)."""
    audio_path = example.get("audio_path")
    if audio_path and Path(audio_path).exists():
        try:
            sr = processor.feature_extractor.sampling_rate if processor else 16000
            audio, _ = librosa.load(audio_path, sr=sr)
            return (sr, audio)
        except Exception as e:
            print(f"Error loading local audio from {audio_path}: {e}")

    audio_data = example.get("audio", {})

    if isinstance(audio_data, dict):
        if "array" in audio_data and "sampling_rate" in audio_data:
            return (audio_data["sampling_rate"], np.array(audio_data["array"]))
        elif "path" in audio_data:
            try:
                sr = processor.feature_extractor.sampling_rate if processor else 16000
                audio, _ = librosa.load(audio_data["path"], sr=sr)
                return (sr, audio)
            except Exception as e:
                print(f"Error loading audio from HF path: {e}")

    return None


def generate_colored_diff(text1: str, text2: str) -> str:
    """Generate HTML with colored diff between two texts."""
    words1 = text1.split()
    words2 = text2.split()

    matcher = difflib.SequenceMatcher(None, words1, words2)

    html_parts = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            html_parts.append(' '.join(words1[i1:i2]))
        elif op == 'replace':
            if i1 < i2:
                html_parts.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{" ".join(words1[i1:i2])}</span>')
            if j1 < j2:
                html_parts.append(f'<span style="background-color: #ccffcc; font-weight: bold;">{" ".join(words2[j1:j2])}</span>')
        elif op == 'delete':
            html_parts.append(f'<span style="background-color: #ffcccc; text-decoration: line-through;">{" ".join(words1[i1:i2])}</span>')
        elif op == 'insert':
            html_parts.append(f'<span style="background-color: #ccffcc; font-weight: bold;">{" ".join(words2[j1:j2])}</span>')

    return ' '.join(html_parts)


def prepare_audio(audio_input):
    """Prepare audio array from various input formats."""
    if audio_input is None:
        return None

    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        if sr != processor.feature_extractor.sampling_rate:
            audio_array = librosa.resample(
                audio_array.astype(np.float32),
                orig_sr=sr,
                target_sr=processor.feature_extractor.sampling_rate
            )
    elif isinstance(audio_input, str):
        audio_array, _ = librosa.load(audio_input, sr=processor.feature_extractor.sampling_rate)
    else:
        return None

    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0

    return audio_array.astype(np.float32)


def chunk_audio(audio_array: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    """
    Split long audio into chunks of MAX_AUDIO_SECONDS with overlap.

    Returns list of audio chunks. If audio is short enough, returns single chunk.
    """
    total_samples = len(audio_array)
    max_samples = int(MAX_AUDIO_SECONDS * sample_rate)
    overlap_samples = int(OVERLAP_SECONDS * sample_rate)

    # If audio is short enough, return as single chunk
    if total_samples <= max_samples:
        return [audio_array]

    chunks = []
    start = 0

    while start < total_samples:
        end = min(start + max_samples, total_samples)
        chunks.append(audio_array[start:end])

        # Move start forward, accounting for overlap
        start = end - overlap_samples

        # Avoid tiny final chunks
        if total_samples - start < sample_rate * 2:  # Less than 2 seconds remaining
            break

    return chunks


def run_transcription_single(audio_array: np.ndarray, system_prompt: str) -> str:
    """Run transcription on a single audio chunk (should be <=30s)."""
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": "Transcribe the speech word-for-word:"},
        ]},
    ]

    try:
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=text,
            audio=[audio_array],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def run_transcription(audio_array: np.ndarray, system_prompt: str) -> str:
    """Run transcription with chunking for long audio."""
    sample_rate = processor.feature_extractor.sampling_rate
    chunks = chunk_audio(audio_array, sample_rate)

    if len(chunks) == 1:
        # Short audio - single transcription
        return run_transcription_single(audio_array, system_prompt)

    # Long audio - transcribe chunks and concatenate
    results = []
    for i, chunk in enumerate(chunks):
        chunk_duration = len(chunk) / sample_rate
        print(f"Transcribing chunk {i+1}/{len(chunks)} ({chunk_duration:.1f}s)")
        result = run_transcription_single(chunk, system_prompt)
        if not result.startswith("Error:"):
            results.append(result)
        else:
            results.append(f"[Chunk {i+1} error: {result}]")

    # Join results with space
    return " ".join(results)


def transcribe_both(
    audio_input,
    entity_hints: str,
) -> tuple[str, str, str]:
    """
    Transcribe audio both with and without entity context.

    Returns:
        Tuple of (without_context, with_context, diff_html)
    """
    if model is None or processor is None:
        return "Error: Model not loaded", "", ""

    if audio_input is None:
        return "Error: No audio provided", "", ""

    audio_array = prepare_audio(audio_input)
    if audio_array is None:
        return "Error: Invalid audio format", "", ""

    # Show audio info
    sample_rate = processor.feature_extractor.sampling_rate
    duration = len(audio_array) / sample_rate
    chunks = chunk_audio(audio_array, sample_rate)
    print(f"Audio: {duration:.1f}s, will process in {len(chunks)} chunk(s)")

    # 1. Transcribe WITHOUT context
    result_without = run_transcription(
        audio_array,
        "You are a speech transcription system. Output ONLY the exact words spoken in the audio, nothing else."
    )

    # 2. Transcribe WITH entity context
    entities = [e.strip() for e in entity_hints.split(",") if e.strip()]
    if entities:
        entity_str = ", ".join(entities)
        result_with = run_transcription(
            audio_array,
            f"You are a speech transcription system. Output ONLY the exact words spoken. The following names/terms may appear: {entity_str}. Use correct spelling for these terms."
        )
    else:
        result_with = "(No entities provided - same as without context)"

    # 3. Generate colored diff
    if entities and not result_with.startswith("(No entities") and not result_with.startswith("Error"):
        diff_html = f"""
        <div style="padding: 10px; font-family: monospace; line-height: 1.6;">
            <p><strong>Legend:</strong>
            <span style="background-color: #ffcccc; text-decoration: line-through;">removed</span>
            <span style="background-color: #ccffcc; font-weight: bold;">added</span></p>
            <hr>
            <p>{generate_colored_diff(result_without, result_with)}</p>
        </div>
        """
    else:
        diff_html = "<p>No diff available (no entities provided or error occurred)</p>"

    return result_without, result_with, diff_html


def load_example_to_ui(
    example_idx: int,
    config: str,
    split: str,
) -> tuple:
    """Load a dataset example into the UI components."""
    examples = load_examples(config, split, num_examples=50)

    if not examples or example_idx is None or example_idx >= len(examples):
        return None, "", ""

    example = examples[example_idx]

    audio_data = get_audio_from_example(example)
    entity_list = example.get("entity_list", [])
    entity_hints = ", ".join(entity_list) if entity_list else ""
    ground_truth = example.get("text", "")

    return audio_data, entity_hints, ground_truth


def format_example_list(config: str, split: str) -> list:
    """Format examples for dropdown display."""
    examples = load_examples(config, split, num_examples=50)
    choices = []
    for i, ex in enumerate(examples):
        label = f"{ex['id']} ({ex['duration']:.1f}s)"
        if config == "ContextASR-Dialogue" and ex.get("movie_name"):
            label += f" - {ex['movie_name'][:30]}"
        elif ex.get("domain_label"):
            label += f" - {ex['domain_label']}"
        choices.append((label, i))
    return choices


def update_example_dropdown(config: str, split: str):
    """Update the example dropdown when config/split changes."""
    choices = format_example_list(config, split)
    return gr.Dropdown(choices=choices, value=0 if choices else None)


def create_app():
    """Create the Gradio app."""

    with gr.Blocks(title="Contextual ASR with Qwen2-Audio", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Contextual ASR with Qwen2-Audio

        This demo shows how **entity hints** from the `entity_list` column can improve ASR accuracy.

        Load an example, then click **Transcribe** to see results **with and without** entity context, plus a colored diff.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Dataset Examples")

                config_dropdown = gr.Dropdown(
                    choices=["ContextASR-Dialogue", "ContextASR-Speech"],
                    value="ContextASR-Dialogue",
                    label="Dataset Configuration"
                )

                split_dropdown = gr.Dropdown(
                    choices=["English", "Mandarin"],
                    value="English",
                    label="Language Split"
                )

                example_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Example",
                    type="index"
                )

                load_example_btn = gr.Button("Load Example", variant="secondary")

                gr.Markdown("### Audio Input")
                audio_input = gr.Audio(
                    label="Audio (upload or record)",
                    sources=["upload", "microphone"],
                    type="numpy"
                )

                gr.Markdown("### Entity Hints")
                entity_hints = gr.Textbox(
                    label="Entity/Vocabulary Hints (comma-separated)",
                    placeholder="Names, terms, etc. that may appear in the audio",
                    lines=3,
                    info="These hints help the model recognize specific names and terms"
                )

                gr.Markdown("### Ground Truth")
                ground_truth_box = gr.Textbox(
                    label="Ground Truth (from dataset)",
                    lines=3,
                    interactive=False
                )

            with gr.Column(scale=1):
                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

                gr.Markdown("### Without Entity Context")
                result_without = gr.Textbox(
                    label="Transcription (No Context)",
                    lines=4,
                    interactive=False
                )

                gr.Markdown("### With Entity Context")
                result_with = gr.Textbox(
                    label="Transcription (With entity hints)",
                    lines=4,
                    interactive=False
                )

                gr.Markdown("### Diff (Without → With)")
                diff_output = gr.HTML(
                    label="Colored Diff",
                    value="<p>Click Transcribe to see diff</p>"
                )

        # Event handlers
        config_dropdown.change(
            fn=update_example_dropdown,
            inputs=[config_dropdown, split_dropdown],
            outputs=[example_dropdown]
        )

        split_dropdown.change(
            fn=update_example_dropdown,
            inputs=[config_dropdown, split_dropdown],
            outputs=[example_dropdown]
        )

        load_example_btn.click(
            fn=load_example_to_ui,
            inputs=[example_dropdown, config_dropdown, split_dropdown],
            outputs=[audio_input, entity_hints, ground_truth_box]
        )

        transcribe_btn.click(
            fn=transcribe_both,
            inputs=[audio_input, entity_hints],
            outputs=[result_without, result_with, diff_output]
        )

        # Initialize example dropdown on load
        app.load(
            fn=update_example_dropdown,
            inputs=[config_dropdown, split_dropdown],
            outputs=[example_dropdown]
        )

        gr.Markdown("""
        ---
        ### How It Works

        1. **Load an example** from the ContextASR-Bench dataset
        2. The **entity_list** (names, terms) is automatically loaded as hints
        3. Click **Transcribe** to run ASR twice:
           - Once **without** any context hints
           - Once **with** entity hints in the system prompt
        4. Compare results with the **colored diff**:
           - <span style="background-color: #ffcccc;">Red strikethrough</span> = removed/changed from no-context version
           - <span style="background-color: #ccffcc;">Green bold</span> = added/corrected in with-context version
        """)

    return app


def set_data_dir(data_dir: str = None):
    """Set the local data directory for the dataset."""
    global local_data_dir

    if data_dir:
        if Path(data_dir).exists():
            local_data_dir = data_dir
            print(f"Using local dataset from: {local_data_dir}")
            return True
        else:
            print(f"Warning: Local data directory not found: {data_dir}")
            print("Will fall back to HuggingFace dataset")
            local_data_dir = None
            return False
    else:
        if Path(DEFAULT_DATA_DIR).exists():
            local_data_dir = DEFAULT_DATA_DIR
            print(f"Found local dataset at default location: {local_data_dir}")
            return True
        else:
            local_data_dir = None
            print("No local dataset found. Will stream from HuggingFace.")
            print("Tip: Run 'python download_contextasr.py' to download the dataset locally")
            return False


if __name__ == "__main__":
    args = get_args()

    # Set local data directory
    set_data_dir(args.data_dir)

    # Load model
    load_model(args.checkpoint, args.cpu_only)

    # Create and launch app
    app = create_app()
    app.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
