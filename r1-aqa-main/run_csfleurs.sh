#!/bin/bash

# ============================================================================
# GRPO Training Script for CS-FLEURS Code-Switched Speech Recognition
# ============================================================================
#
# This script trains Qwen2-Audio on CS-FLEURS using GRPO with WER reward.
# CS-FLEURS contains 113 unique code-switched language pairs across 52 languages.
#
# Audio chunking:
#   - Uses VAD (Voice Activity Detection) to segment at speech boundaries
#   - Merges short segments, ensures max 30s chunks
#   - Aligns transcripts to VAD boundaries
#
# Usage:
#   ./run_csfleurs.sh                                          # Default: VAD chunking enabled
#   ./run_csfleurs.sh xtts_train                               # Specific subset
#   ./run_csfleurs.sh xtts_train ara-eng                       # Specific language pair
#   ./run_csfleurs.sh xtts_train ara-eng 1000                  # With sample limit
#   ./run_csfleurs.sh xtts_train all 1000 false                # 1000 samples, no two-step
#   ./run_csfleurs.sh xtts_train all 1000 true                 # two-step training
#   ./run_csfleurs.sh xtts_train all 1000 true cer             # two-step, CER reward
#   ./run_csfleurs.sh xtts_train all 1000 true cgpr            # two-step, CGPR reward (RECOMMENDED)
#   ./run_csfleurs.sh xtts_train all 1000 true cgpr true       # VAD chunking (default)
#   ./run_csfleurs.sh xtts_train all 1000 true cgpr false      # No VAD chunking
#
# Reward types:
#   wer    - Word Error Rate (default)
#   cer    - Character Error Rate (good for code-switched/Chinese)
#   mixed  - Combined WER + CER (50/50 by default)
#   cgpr   - Confidence-Gated Process Rewards (dense rewards on code-switched entities)
#   cgpr_plus - CGPR + anti-translation contrastive penalty + script contamination penalty
#   format - Format-only baseline (no ASR reward, just <answer> tag compliance)
#
# Subsets:
#   read_test:  14 X-English pairs, 17 hours (read speech)
#   xtts_train: 16 X-English pairs, 128 hours (generative TTS) - RECOMMENDED
#   xtts_test1: 16 X-English pairs, 36 hours (generative TTS)
#   xtts_test2: 60 {Arabic, Chinese, Hindi, Spanish}-X pairs, 42 hours
#   mms_test:   45 X-English pairs, 56 hours (concatenative TTS)
#
# Prerequisites:
#   pip install -r requirements.txt
#   # Clone dataset: git clone https://huggingface.co/datasets/byan/cs-fleurs csfleurs_data
#
# Language pairs (use codes like ara-eng, cmn-eng, hin-eng, spa-eng, etc.)
# ============================================================================

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Configuration
SUBSET="${1:-xtts_train}"                    # Dataset subset
LANGUAGE_PAIR="${2:-all}"                    # Language pair (e.g., ara-eng, cmn-eng) or 'all'
NUM_EXAMPLES="${3:-1000}"                    # Number of training examples
TWO_STEP="${4:-false}"                       # Two-step training
REWARD_TYPE="${5:-cer}"                      # Reward type: wer, cer, or mixed
USE_VAD="false"                              # VAD chunking disabled
EVAL_STEPS="${6:-50}"                        # Evaluate every N steps
MAX_EVAL_SAMPLES="${7:-200}"                 # Max validation samples for eval
SEED="${8:-42}"                              # Random seed
RESUME_CHECKPOINT="${9:-}"                   # Resume from checkpoint (optional, last arg)

# Data path (local clone of cs-fleurs)
DATA_DIR="/home/ubuntu/Qwen2-Audio/csfleurs_data"

# Model
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Training hyperparameters
NUM_GPUS=8
NUM_GENERATIONS=8
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-6
# Two-step does 2 GRPO updates per step, so halve epochs to equalize gradient updates
if [ "${TWO_STEP}" = "true" ]; then
    NUM_EPOCHS=4
else
    NUM_EPOCHS=8
fi
SAVE_STEPS=50
MAX_AUDIO_DURATION=30.0

# Output directory
if [ "${LANGUAGE_PAIR}" = "all" ]; then
    OUT_DIR="./outputs/csfleurs_${SUBSET}_${REWARD_TYPE}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}"
else
    OUT_DIR="./outputs/csfleurs_${SUBSET}_${LANGUAGE_PAIR}_${REWARD_TYPE}_n${NUM_EXAMPLES}_e${NUM_EPOCHS}"
fi
if [ "${TWO_STEP}" = "true" ]; then
    OUT_DIR="${OUT_DIR}_twostep"
fi
if [ "${USE_VAD}" = "true" ]; then
    OUT_DIR="${OUT_DIR}_vad"
else
    OUT_DIR="${OUT_DIR}_novad"
fi
OUT_DIR="${OUT_DIR}_s${SEED}"

# WandB settings
USE_WANDB="true"
if [ "${USE_VAD}" = "true" ]; then
    VAD_TAG="vad"
else
    VAD_TAG="novad"
fi
if [ "${LANGUAGE_PAIR}" = "all" ]; then
    RUN_NAME="CSFleurs-${SUBSET}-${REWARD_TYPE}-n${NUM_EXAMPLES}-e${NUM_EPOCHS}-${VAD_TAG}-s${SEED}-GRPO"
else
    RUN_NAME="CSFleurs-${SUBSET}-${LANGUAGE_PAIR}-${REWARD_TYPE}-n${NUM_EXAMPLES}-e${NUM_EPOCHS}-${VAD_TAG}-s${SEED}-GRPO"
fi

echo "=============================================="
echo "CS-FLEURS GRPO Training"
echo "=============================================="
echo "Subset:       ${SUBSET}"
echo "Language:     ${LANGUAGE_PAIR}"
echo "Examples:     ${NUM_EXAMPLES}"
echo "Epochs:       ${NUM_EPOCHS}"
echo "Reward:       ${REWARD_TYPE}"
echo "Two-step:     ${TWO_STEP}"
echo "VAD chunk:    ${USE_VAD}"
echo "Seed:         ${SEED}"
echo "Data dir:     ${DATA_DIR}"
if [ "${TWO_STEP}" = "true" ]; then
    echo "  Pass 1: Draft transcription"
    echo "  Pass 2: Refinement using draft"
fi
echo "Output:       ${OUT_DIR}"
echo "GPUs:         ${NUM_GPUS}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "Resume from:  ${RESUME_CHECKPOINT}"
fi
echo "=============================================="

# Build language pair argument
LANG_ARGS=""
if [ "${LANGUAGE_PAIR}" != "all" ]; then
    LANG_ARGS="--language_pair ${LANGUAGE_PAIR}"
fi

# Build two-step argument
TWO_STEP_ARGS=""
if [ "${TWO_STEP}" = "true" ]; then
    TWO_STEP_ARGS="--two_step_training"
fi

# Build VAD chunking argument
VAD_ARGS=""
if [ "${USE_VAD}" = "false" ]; then
    VAD_ARGS="--use_vad_chunking False"
fi

# Run training
if [ -n "${RESUME_CHECKPOINT}" ]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=32780 \
        src/train_csfleurs.py \
        --config_path configs/zero2.json \
        --model_name_or_path ${MODEL_NAME} \
        --data_dir ${DATA_DIR} \
        --subset ${SUBSET} \
        --out_dir ${OUT_DIR} \
        --num_examples ${NUM_EXAMPLES} \
        --max_audio_duration ${MAX_AUDIO_DURATION} \
        --num_generations ${NUM_GENERATIONS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --save_steps ${SAVE_STEPS} \
        --run_name ${RUN_NAME} \
        --reward_type ${REWARD_TYPE} \
        --cgpr_beta_translation 0.05 \
        --cgpr_beta_script 0.05 \
        --eval_steps ${EVAL_STEPS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --seed ${SEED} \
        --resume_from_checkpoint "${RESUME_CHECKPOINT}" \
        ${LANG_ARGS} \
        ${TWO_STEP_ARGS} \
        ${VAD_ARGS}
else
    torchrun --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=32780 \
        src/train_csfleurs.py \
        --config_path configs/zero2.json \
        --model_name_or_path ${MODEL_NAME} \
        --data_dir ${DATA_DIR} \
        --subset ${SUBSET} \
        --out_dir ${OUT_DIR} \
        --num_examples ${NUM_EXAMPLES} \
        --max_audio_duration ${MAX_AUDIO_DURATION} \
        --num_generations ${NUM_GENERATIONS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --num_train_epochs ${NUM_EPOCHS} \
        --save_steps ${SAVE_STEPS} \
        --run_name ${RUN_NAME} \
        --reward_type ${REWARD_TYPE} \
        --cgpr_beta_translation 0.05 \
        --cgpr_beta_script 0.05 \
        --eval_steps ${EVAL_STEPS} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --seed ${SEED} \
        ${LANG_ARGS} \
        ${TWO_STEP_ARGS} \
        ${VAD_ARGS}
fi
