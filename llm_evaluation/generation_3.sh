#!/bin/bash

# Environment variables for evaluation
export MODEL_BASE_PATH=${MODEL_BASE_PATH:-"/mnt/raid6/jaewook133/models"}
export RESULTS_BASE_PATH=${RESULTS_BASE_PATH:-"./results"}
export CUDA_DEVICE=${CUDA_DEVICE_3:-"3"}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "1. generation.py" \
    --model_path "$MODEL_BASE_PATH/Qwen2.5-1.5B_translate_enko/" \
    --output_path "$RESULTS_BASE_PATH/qwen2.5-1.5B.json"
