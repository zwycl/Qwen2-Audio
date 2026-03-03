#!/bin/bash

# ============================================================================
# Evaluation Script for SwitchLingua Code-Switched Speech Recognition
# ============================================================================
#
# SwitchLingua (NeurIPS 2025): 80+ hours, 11 X-English language pairs.
# Languages: Arabic, Cantonese, French, German, Hindi, Italian, Japanese,
#            Korean, Mandarin, Russian, Spanish
#
# Usage:
#   ./run_eval_switchlingua.sh                              # Raw model, 100 examples
#   ./run_eval_switchlingua.sh raw 5 Italian                # 5 Italian examples
#   ./run_eval_switchlingua.sh raw 100 all 0 4              # 4 GPUs, all languages
#   ./run_eval_switchlingua.sh ./outputs/checkpoint-30 100  # Checkpoint eval
#
# Arguments:
#   $1 - Model path or "raw" for base model (default: raw)
#   $2 - Number of examples to evaluate (default: 100)
#   $3 - Language: Arabic, Cantonese, French, German, Hindi, Italian,
#        Japanese, Korean, Mandarin, Russian, Spanish, or "all" (default: all)
#   $4 - Number of examples to skip (default: 0)
#   $5 - Number of GPUs (default: 1)
#   $6 - Two-step evaluation: true/false (default: false)
#
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Parse arguments
MODEL_PATH="${1:-raw}"
NUM_EXAMPLES="${2:-100}"
LANGUAGE="${3:-all}"
SKIP_EXAMPLES="${4:-0}"
NUM_GPUS="${5:-1}"
TWO_STEP="${6:-false}"

# Handle "raw" as base model
RAW_PROMPT_FLAG=""
if [ "${MODEL_PATH}" = "raw" ]; then
    MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
    MODEL_NAME="raw"
    RAW_PROMPT_FLAG="--raw_model_prompt"
else
    MODEL_NAME=$(basename "${MODEL_PATH}")
fi

# Data directory
DATA_DIR="/home/ubuntu/Qwen2-Audio/SwitchLingua_audio"

# Output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/eval_results"
if [ "${LANGUAGE}" = "all" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/eval_switchlingua_${MODEL_NAME}_all_n${NUM_EXAMPLES}_${TIMESTAMP}.json"
else
    OUTPUT_FILE="${OUTPUT_DIR}/eval_switchlingua_${MODEL_NAME}_${LANGUAGE}_n${NUM_EXAMPLES}_${TIMESTAMP}.json"
fi

echo "=============================================="
echo "SwitchLingua Evaluation"
echo "=============================================="
echo "Model:       ${MODEL_PATH}"
echo "Language:    ${LANGUAGE}"
echo "Eval size:   ${NUM_EXAMPLES} examples"
echo "Skip:        ${SKIP_EXAMPLES}"
echo "GPUs:        ${NUM_GPUS}"
echo "Two-step:    ${TWO_STEP}"
echo "Output:      ${OUTPUT_FILE}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build language argument
LANG_ARGS=""
if [ "${LANGUAGE}" != "all" ]; then
    LANG_ARGS="--language ${LANGUAGE}"
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
        --master_port=29502 \
        src/evaluate_switchlingua.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        ${LANG_ARGS} \
        ${RAW_PROMPT_FLAG} \
        ${TWO_STEP_ARGS}
else
    echo "Running single-GPU evaluation..."
    python src/evaluate_switchlingua.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        --verbose \
        ${LANG_ARGS} \
        ${RAW_PROMPT_FLAG} \
        ${TWO_STEP_ARGS}
fi

echo ""
echo "Results saved to: ${OUTPUT_FILE}"
