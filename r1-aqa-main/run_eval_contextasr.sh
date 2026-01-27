#!/bin/bash

# ============================================================================
# Evaluation Script for ContextASR-Bench
# ============================================================================
#
# This script evaluates Qwen2-Audio on ContextASR-Bench using WER and B-WER metrics.
# It evaluates on examples NOT used in training (skips first N examples).
#
# Usage:
#   # Single GPU (default)
#   ./run_eval_contextasr.sh raw 100
#
#   # Multi-GPU (8 GPUs in parallel)
#   ./run_eval_contextasr.sh raw 100 Dialogue English 1000 8
#
#   # Examples:
#   ./run_eval_contextasr.sh                              # Raw model, 100 examples, 1 GPU
#   ./run_eval_contextasr.sh ./outputs/checkpoint-30      # Checkpoint, 1 GPU
#   ./run_eval_contextasr.sh raw 200 Dialogue English 1000 8  # 8 GPUs
#
# Arguments:
#   $1 - Model path or "raw" for base model (default: raw)
#   $2 - Number of examples to evaluate (default: 100)
#   $3 - Dataset config short name: Dialogue or Speech (default: Dialogue)
#   $4 - Language: English or Mandarin (default: English)
#   $5 - Number of training examples to skip (default: 1000)
#   $6 - Number of GPUs (default: 1, set to 8 for multi-GPU)
#
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Parse arguments
MODEL_PATH="${1:-raw}"
NUM_EXAMPLES="${2:-100}"
DATASET_CONFIG_SHORT="${3:-Dialogue}"
LANGUAGE="${4:-English}"
SKIP_EXAMPLES="${5:-1000}"
NUM_GPUS="${6:-1}"

# Expand config name
DATASET_CONFIG="ContextASR-${DATASET_CONFIG_SHORT}"

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
DATA_DIR="/home/ubuntu/Qwen2-Audio/contextasr_data"

# Output file for detailed results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./outputs/eval_results"
OUTPUT_FILE="${OUTPUT_DIR}/eval_${MODEL_NAME}_${DATASET_CONFIG}_${LANGUAGE}_n${NUM_EXAMPLES}_${TIMESTAMP}.json"

echo "=============================================="
echo "ContextASR Evaluation"
echo "=============================================="
echo "Model:       ${MODEL_PATH}"
echo "Config:      ${DATASET_CONFIG}"
echo "Language:    ${LANGUAGE}"
echo "Eval size:   ${NUM_EXAMPLES} examples"
echo "Skip:        ${SKIP_EXAMPLES} (training set)"
echo "GPUs:        ${NUM_GPUS}"
echo "Output:      ${OUTPUT_FILE}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Running multi-GPU evaluation with ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        src/evaluate_contextasr.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --dataset_config "${DATASET_CONFIG}" \
        --language "${LANGUAGE}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        ${RAW_PROMPT_FLAG}
else
    echo "Running single-GPU evaluation..."
    python src/evaluate_contextasr.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --dataset_config "${DATASET_CONFIG}" \
        --language "${LANGUAGE}" \
        --skip_examples "${SKIP_EXAMPLES}" \
        --num_examples "${NUM_EXAMPLES}" \
        --output_file "${OUTPUT_FILE}" \
        --verbose \
        ${RAW_PROMPT_FLAG}
fi

echo ""
echo "Results saved to: ${OUTPUT_FILE}"
