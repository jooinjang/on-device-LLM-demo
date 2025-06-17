#!/bin/bash

# Environment variables for evaluation
export MODEL_BASE_PATH=${MODEL_BASE_PATH:-"/mnt/raid6/jaewook133/models"}
export RESULTS_BASE_PATH=${RESULTS_BASE_PATH:-"./results"}
export CUDA_DEVICE=${CUDA_DEVICE_2:-"2"}

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "1. generation.py" \
    --model_path "$MODEL_BASE_PATH/Llama-3.2-3B-Inst_translate_en-ko/" \
    --output_path "$RESULTS_BASE_PATH/llama3.2-3B.json"
