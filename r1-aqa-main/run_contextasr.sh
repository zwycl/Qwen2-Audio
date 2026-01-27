#!/bin/bash

# ============================================================================
# GRPO Training Script for ContextASR-Bench
# ============================================================================
#
# This script trains Qwen2-Audio on ContextASR-Bench using GRPO with WER reward.
# The key feature is contextual ASR with entity hints in the prompt.
#
# Usage:
#   ./run_contextasr.sh                                        # Default: Dialogue, English, wer reward
#   ./run_contextasr.sh Speech English                         # Speech config, English, wer reward
#   ./run_contextasr.sh Dialogue Mandarin                      # Dialogue config, Mandarin, wer reward
#   ./run_contextasr.sh Dialogue English 500 wer               # Dialogue, English, 500 examples, wer reward
#   ./run_contextasr.sh Dialogue English 500 cgpr              # Dialogue, English, 500 examples, CGPR reward
#   ./run_contextasr.sh Dialogue English 500 wer false         # Same but without entities in prompt
#   ./run_contextasr.sh Dialogue English 500 wer false true    # Two-step training (draft + refinement)
#   ./run_contextasr.sh Dialogue English 500 format false true # Format-only baseline (no ASR reward)
#
# Reward types:
#   wer    - Simple negative WER reward (lower WER = higher reward)
#   cgpr   - CGPR (Confidence-Gated Process Rewards) with Tsallis entropy confidence for entity-focused training
#   format - Format-only baseline (no ASR reward, just <answer> tag compliance)
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
DATASET_CONFIG_SHORT="${1:-Dialogue}"        # Dialogue or Speech
LANGUAGE="${2:-English}"                     # English or Mandarin
NUM_EXAMPLES="${3:-1000}"                    # Number of training examples
REWARD_TYPE="${4:-wer}"                      # Reward type: wer or cgpr
NO_ENTITIES="${5:-false}"                    # If true, don't add entities to prompt
TWO_STEP="${6:-false}"                       # If true, use two-step training (draft + refinement)

# Expand to full config name
DATASET_CONFIG="ContextASR-${DATASET_CONFIG_SHORT}"

# CGPR hyperparameters (only used if REWARD_TYPE=cgpr)
CGPR_ALPHA=0.1                               # Coefficient for correct entity reward
CGPR_BETA=0.3                                # Coefficient for incorrect entity penalty
CGPR_LAMBDA=4.0                              # Weight for B-WER in terminal reward (only if USE_BWER=true)
CGPR_USE_BWER=false                          # Include B-WER in terminal reward (default false, redundant with dense)

# Model
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Data paths
DATA_DIR="/home/ubuntu/Qwen2-Audio/contextasr_data"

# Training hyperparameters
NUM_GPUS=8
NUM_GENERATIONS=8
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
NUM_EPOCHS=4
SAVE_STEPS=15

# Output directory (uses NUM_EPOCHS, so must come after)
OUT_DIR="./outputs/contextasr_${DATASET_CONFIG}_${LANGUAGE}_${REWARD_TYPE}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}"
if [ "${NO_ENTITIES}" = "true" ]; then
    OUT_DIR="${OUT_DIR}_noentities"
fi
if [ "${TWO_STEP}" = "true" ]; then
    OUT_DIR="${OUT_DIR}_twostep"
fi

# WandB settings
USE_WANDB="true"
RUN_NAME="ContextASR-${DATASET_CONFIG}-${LANGUAGE}-${REWARD_TYPE}-n${NUM_EXAMPLES}-e${NUM_EPOCHS}-GRPO"

# Debug settings
export DEBUG_MODE="true"
export LOG_PATH="./outputs/cgpr_debug_${RUN_NAME}.log"

echo "=============================================="
echo "ContextASR GRPO Training"
echo "=============================================="
echo "Config:      ${DATASET_CONFIG}"
echo "Language:    ${LANGUAGE}"
echo "Examples:    ${NUM_EXAMPLES}"
echo "Reward:      ${REWARD_TYPE}"
echo "No entities: ${NO_ENTITIES}"
echo "Two-step:    ${TWO_STEP}"
if [ "${REWARD_TYPE}" = "cgpr" ]; then
    echo "  alpha:     ${CGPR_ALPHA}"
    echo "  beta:      ${CGPR_BETA}"
    echo "  use_bwer:  ${CGPR_USE_BWER}"
    if [ "${CGPR_USE_BWER}" = "true" ]; then
        echo "  lambda:    ${CGPR_LAMBDA}"
    fi
fi
if [ "${TWO_STEP}" = "true" ]; then
    echo "  Pass 1: Draft transcription"
    echo "  Pass 2: Refinement using draft + context"
fi
echo "Output:      ${OUT_DIR}"
echo "GPUs:        ${NUM_GPUS}"
echo "=============================================="

# Build reward arguments
REWARD_ARGS=""
if [ "${REWARD_TYPE}" = "cgpr" ]; then
    REWARD_ARGS="--use_cgpr_reward --cgpr_alpha ${CGPR_ALPHA} --cgpr_beta ${CGPR_BETA} --cgpr_lambda_entity ${CGPR_LAMBDA}"
    if [ "${CGPR_USE_BWER}" = "true" ]; then
        REWARD_ARGS="${REWARD_ARGS} --cgpr_use_bwer"
    fi
elif [ "${REWARD_TYPE}" = "format" ]; then
    REWARD_ARGS="--use_format_only_reward"
fi

# Build entity arguments
ENTITY_ARGS=""
if [ "${NO_ENTITIES}" = "true" ]; then
    ENTITY_ARGS="--no_entities"
fi

# Build two-step arguments
TWO_STEP_ARGS=""
if [ "${TWO_STEP}" = "true" ]; then
    TWO_STEP_ARGS="--two_step_training"
fi

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
    --run_name ${RUN_NAME} \
    ${REWARD_ARGS} \
    ${ENTITY_ARGS} \
    ${TWO_STEP_ARGS}