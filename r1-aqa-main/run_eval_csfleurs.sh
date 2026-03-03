#!/bin/bash

# ============================================================================
# Evaluation Script for CS-FLEURS Code-Switched Speech Recognition
# ============================================================================
#
# This script evaluates Qwen2-Audio on CS-FLEURS using CER and bCER metrics.
# bCER (Boundary-CER): CER near language switch boundaries (±k chars)
#   - Requires preprocessing with: python src/preprocess_csfleurs_markers.py
#   - Measures model performance at code-switch transition points
#   - Also reports nbCER (non-boundary CER) for comparison
# It evaluates on examples NOT used in training (skips first N examples).
#
# Audio chunking:
#   - Uses VAD (Voice Activity Detection) to segment at speech boundaries
#   - Merges short segments, ensures max 30s chunks
#   - Aligns transcripts to VAD boundaries
#
# Usage:
#   # Single GPU (default)
#   ./run_eval_csfleurs.sh raw 100
#
#   # Multi-GPU (8 GPUs in parallel)
#   ./run_eval_csfleurs.sh raw 100 xtts_train all 1000 8
#
#   # With VAD chunking control:
#   ./run_eval_csfleurs.sh raw 100 xtts_train all 1000 8 true   # VAD chunking (default)
#   ./run_eval_csfleurs.sh raw 100 xtts_train all 1000 8 false  # No chunking
#
#   # Examples:
#   ./run_eval_csfleurs.sh                                    # Raw model, 100 examples, 1 GPU
#   ./run_eval_csfleurs.sh ./outputs/checkpoint-30            # Checkpoint, 1 GPU
#   ./run_eval_csfleurs.sh raw 200 xtts_train ara-eng 1000 8  # 8 GPUs, specific lang pair
#
# Arguments:
#   $1 - Model path or "raw" for base model (default: raw)
#   $2 - Number of examples to evaluate (default: 100)
#   $3 - Dataset subset: read_test, xtts_train, xtts_test1, xtts_test2, mms_test (default: xtts_train)
#   $4 - Language pair (e.g., ara-eng, cmn-eng) or 'all' (default: all)
#   $5 - Number of training examples to skip (default: 1000)
#   $6 - Number of GPUs (default: 1, set to 8 for multi-GPU)
#   $7 - Use VAD chunking: true/false (default: true)
#   $8 - Two-step evaluation: true/false (default: false) - draft then refine
#
# Subsets:
#   read_test:  14 X-English pairs, 17 hours (read speech)
#   xtts_train: 16 X-English pairs, 128 hours (generative TTS) - RECOMMENDED
#   xtts_test1: 16 X-English pairs, 36 hours (generative TTS)
#   xtts_test2: 60 {Arabic, Chinese, Hindi, Spanish}-X pairs, 42 hours
#   mms_test:   45 X-English pairs, 56 hours (concatenative TTS)
#
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Parse arguments
MODEL_PATH="${1:-raw}"
NUM_EXAMPLES="${2:-100}"
SUBSET="${3:-xtts_train}"
LANGUAGE_PAIR="${4:-all}"
SKIP_EXAMPLES="${5:-1000}"
NUM_GPUS="${6:-1}"
USE_VAD="${7:-true}"
TWO_STEP="${8:-false}"

# Handle "raw" as base model
RAW_PROMPT_FLAG=""
if [ "${MODEL_PATH}" = "raw" ]; then
    MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
    MODEL_NAME="raw"
    RAW_PROMPT_FLAG="--raw_model_prompt"
else
    MODEL_NAME=$(basename "${MODEL_PATH}")
fi

# Data directory (local clone of cs-fleurs)
DATA_DIR="/home/ubuntu/Qwen2-Audio/csfleurs_data"

# Output file for detailed results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/eval_results"
if [ "${LANGUAGE_PAIR}" = "all" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/eval_csfleurs_${MODEL_NAME}_${SUBSET}_n${NUM_EXAMPLES}_${TIMESTAMP}.json"
else
    OUTPUT_FILE="${OUTPUT_DIR}/eval_csfleurs_${MODEL_NAME}_${SUBSET}_${LANGUAGE_PAIR}_n${NUM_EXAMPLES}_${TIMESTAMP}.json"
fi

echo "=============================================="
echo "CS-FLEURS Evaluation"
echo "=============================================="
echo "Model:       ${MODEL_PATH}"
echo "Subset:      ${SUBSET}"
echo "Language:    ${LANGUAGE_PAIR}"
echo "Eval size:   ${NUM_EXAMPLES} examples"
echo "Skip:        ${SKIP_EXAMPLES} (training set)"
echo "GPUs:        ${NUM_GPUS}"
echo "VAD chunk:   ${USE_VAD}"
echo "Two-step:    ${TWO_STEP}"
echo "Output:      ${OUTPUT_FILE}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build language pair argument
LANG_ARGS=""
if [ "${LANGUAGE_PAIR}" != "all" ]; then
    LANG_ARGS="--language_pair ${LANGUAGE_PAIR}"
fi

# Build two-step argument
TWO_STEP_ARGS=""
if [ "${TWO_STEP}" = "true" ]; then
    TWO_STEP_ARGS="--two_step"
fi

# Run evaluation
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Running multi-GPU evaluation with ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node=${NUM_GPUS} \
        --master_port=29501 \
        src/evaluate_csfleurs.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --subset "${SUBSET}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        --use_vad_chunking "${USE_VAD}" \
        ${LANG_ARGS} \
        ${RAW_PROMPT_FLAG} \
        ${TWO_STEP_ARGS}
else
    echo "Running single-GPU evaluation..."
    python src/evaluate_csfleurs.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --subset "${SUBSET}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        --use_vad_chunking "${USE_VAD}" \
        --verbose \
        ${LANG_ARGS} \
        ${RAW_PROMPT_FLAG} \
        ${TWO_STEP_ARGS}
fi

echo ""
echo "Results saved to: ${OUTPUT_FILE}"
