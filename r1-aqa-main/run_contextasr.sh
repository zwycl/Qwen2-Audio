#!/bin/bash

# ============================================================================
# GRPO Training Script for ContextASR-Bench
# ============================================================================
#
# This script trains Qwen2-Audio on ContextASR-Bench using GRPO with WER reward.
# The key feature is contextual ASR with entity hints in the prompt.
#
# Usage:
#   ./run_contextasr.sh                    # Default: Dialogue config, English
#   ./run_contextasr.sh Speech English     # Speech config, English
#   ./run_contextasr.sh Dialogue Mandarin  # Dialogue config, Mandarin
#
# Prerequisites:
#   1. Download ContextASR-Bench data:
#      python download_contextasr.py --output-dir ./contextasr_data
#
#   2. Install requirements:
#      pip install -r requirements.txt
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Configuration
DATASET_CONFIG="${1:-ContextASR-Dialogue}"  # ContextASR-Dialogue or ContextASR-Speech
LANGUAGE="${2:-English}"                     # English or Mandarin
NUM_EXAMPLES="${3:-3000}"                    # Number of training examples

# Model
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Data paths
DATA_DIR="/home/ubuntu/Qwen2-Audio/contextasr_data"
OUT_DIR="./outputs/contextasr_${DATASET_CONFIG}_${LANGUAGE}"

# Training hyperparameters
NUM_GPUS=8
NUM_GENERATIONS=8
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
NUM_EPOCHS=2
SAVE_STEPS=100

# WandB settings
USE_WANDB="true"
RUN_NAME="ContextASR-${DATASET_CONFIG}-${LANGUAGE}-GRPO"

echo "=============================================="
echo "ContextASR GRPO Training"
echo "=============================================="
echo "Config:      ${DATASET_CONFIG}"
echo "Language:    ${LANGUAGE}"
echo "Examples:    ${NUM_EXAMPLES}"
echo "Output:      ${OUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node-rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=32779 \
    src/train_contextasr.py \
    --config_path configs/zero2.json \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --dataset_config ${DATASET_CONFIG} \
    --language ${LANGUAGE} \
    --out_dir ${OUT_DIR} \
    --num_examples ${NUM_EXAMPLES} \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_steps ${SAVE_STEPS} \
    --use_wandb ${USE_WANDB} \
    --run_name ${RUN_NAME}
