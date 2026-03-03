#!/bin/bash

# ============================================================================
# LoRA Fine-tuning Script for CS-FLEURS Code-Switched Speech Recognition
# ============================================================================
#
# This script fine-tunes Qwen2-Audio on CS-FLEURS using LoRA (PEFT).
# Standard SFT training - faster than GRPO since no generation loop.
#
# Usage:
#   ./run_lora_csfleurs.sh                              # Default: xtts_train, 1000 examples
#   ./run_lora_csfleurs.sh xtts_train                   # Specific subset
#   ./run_lora_csfleurs.sh xtts_train all 1000          # All language pairs, 1000 examples
#   ./run_lora_csfleurs.sh xtts_train ara-eng 500       # Specific language pair
#   ./run_lora_csfleurs.sh xtts_train all 1000 4        # 4 epochs
#   ./run_lora_csfleurs.sh xtts_train all 1000 4 50 200 67  # With seed 67
#
# Subsets:
#   read_test:  14 X-English pairs, 17 hours (read speech)
#   xtts_train: 16 X-English pairs, 128 hours (generative TTS) - RECOMMENDED
#   xtts_test1: 16 X-English pairs, 36 hours (generative TTS)
#   xtts_test2: 60 {Arabic, Chinese, Hindi, Spanish}-X pairs, 42 hours
#   mms_test:   45 X-English pairs, 56 hours (concatenative TTS)
#
# Prerequisites:
#   1. Clone dataset: git clone https://huggingface.co/datasets/byan/cs-fleurs csfleurs_data
#   2. pip install peft
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Configuration
SUBSET="${1:-xtts_train}"                    # Dataset subset
LANGUAGE_PAIR="${2:-all}"                    # Language pair (e.g., ara-eng, cmn-eng) or 'all'
NUM_EXAMPLES="${3:-1000}"                    # Number of training examples
NUM_EPOCHS="${4:-4}"                         # Number of epochs
EVAL_STEPS="${5:-50}"                        # Evaluate every N steps
MAX_EVAL_SAMPLES="${6:-200}"                 # Max validation samples for eval
SEED="${7:-42}"                              # Random seed

# Model
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Data paths
DATA_DIR="/home/ubuntu/Qwen2-Audio/csfleurs_data"

# Training hyperparameters
NUM_GPUS=8
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
SAVE_STEPS=50

# Output directory
if [ "${LANGUAGE_PAIR}" = "all" ]; then
    OUT_DIR="./outputs/lora_csfleurs_${SUBSET}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}_s${SEED}"
else
    OUT_DIR="./outputs/lora_csfleurs_${SUBSET}_${LANGUAGE_PAIR}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}_s${SEED}"
fi

# Build language pair argument
LANG_ARGS=""
if [ "${LANGUAGE_PAIR}" != "all" ]; then
    LANG_ARGS="--language_pair ${LANGUAGE_PAIR}"
fi

echo "=============================================="
echo "CS-FLEURS LoRA Fine-tuning"
echo "=============================================="
echo "Subset:      ${SUBSET}"
echo "Language:    ${LANGUAGE_PAIR}"
echo "Examples:    ${NUM_EXAMPLES}"
echo "Epochs:      ${NUM_EPOCHS}"
echo "Seed:        ${SEED}"
echo "Output:      ${OUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "=============================================="

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node-rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=32782 \
    train_lora_csfleurs.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUT_DIR} \
    --subset ${SUBSET} \
    --max_train_samples ${NUM_EXAMPLES} \
    --num_train_epochs ${NUM_EPOCHS} \
    --seed ${SEED} \
    --attn_implementation sdpa \
    --deepspeed configs/lora_zero2.json \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --save_strategy epoch \
    --save_total_limit 1 \
    --eval_strategy no \
    --bf16 \
    --lr_scheduler_type constant \
    ${LANG_ARGS}
