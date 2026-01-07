#!/bin/bash

cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node-rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=32778 \
    src/train_afrispeech_asr.py \
    --config_path configs/zero2.json \
    --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --data_dir /home/ubuntu/Qwen2-Audio/afrispeech_data \
    --out_dir ./outputs/afrispeech_asr_wandb \
    --num_examples 3000 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --use_wandb true \
    --run_name "AfriSpeech-ASR-GRPO"
