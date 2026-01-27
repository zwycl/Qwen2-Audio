#!/bin/bash

# ============================================================================
# LoRA Fine-tuning Script for ContextASR-Bench
# ============================================================================
#
# This script fine-tunes Qwen2-Audio on ContextASR-Bench using LoRA (PEFT).
#
# Usage:
#   ./run_lora_contextasr.sh                          # Default: Speech, English, 2000 examples
#   ./run_lora_contextasr.sh Dialogue English         # Dialogue config, English
#   ./run_lora_contextasr.sh Speech Mandarin 1000     # Speech config, Mandarin, 1000 examples
#
# Prerequisites:
#   1. Download ContextASR-Bench data:
#      python download_contextasr.py --output-dir ./contextasr_data
#
#   2. Install requirements:
#      pip install peft
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Configuration
DATASET_CONFIG_SHORT="${1:-Speech}"          # Dialogue or Speech
LANGUAGE="${2:-English}"                     # English or Mandarin
NUM_EXAMPLES="${3:-2000}"                    # Number of training examples

# Expand to full config name
DATASET_CONFIG="ContextASR-${DATASET_CONFIG_SHORT}"

# Model
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Data paths
DATA_DIR="/home/ubuntu/Qwen2-Audio/contextasr_data"

# Training hyperparameters
NUM_GPUS=8
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
NUM_EPOCHS=4
SAVE_STEPS=15
SEED=42

# Output directory
OUT_DIR="./outputs/lora_contextasr_${DATASET_CONFIG}_${LANGUAGE}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}"

echo "=============================================="
echo "ContextASR LoRA Fine-tuning"
echo "=============================================="
echo "Config:      ${DATASET_CONFIG}"
echo "Language:    ${LANGUAGE}"
echo "Examples:    ${NUM_EXAMPLES}"
echo "Epochs:      ${NUM_EPOCHS}"
echo "Output:      ${OUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node-rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=32779 \
    train_lora_contextasr.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUT_DIR} \
    --config ${DATASET_CONFIG} \
    --language ${LANGUAGE} \
    --max_train_samples ${NUM_EXAMPLES} \
    --num_train_epochs ${NUM_EPOCHS} \
    --seed ${SEED} \
    --attn_implementation sdpa \
    --deepspeed configs/lora_zero2.json \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --save_steps ${SAVE_STEPS} \
    --bf16
