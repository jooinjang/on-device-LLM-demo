export WANDB_NAME="OLM-translate_en-ko-unsloth"
EPOCH=3
LR=3e-05
CUDA_VISIBLE_DEVICES=1 python train_unsloth.py \
    --proctitle "jaewook133/Llama-3.1" \
    --base_model "unsloth/Meta-Llama-3.1-8B-bnb-4bit" \
    --ckpt_dir "YOUR_MODEL_PATH/Llama3.1-8B_translate_enko_unsloth/ckpt" \
    --output_dir "YOUR_MODEL_PATH/Llama3.1-8B_translate_enko_unsloth/" \
    --output_gguf_dir "YOUR_MODEL_PATH/Llama3.1-8B_translate_enko_unsloth/gguf" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "linear"