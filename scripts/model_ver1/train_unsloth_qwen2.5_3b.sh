export WANDB_NAME="OLM-translate_en-ko-unsloth"
EPOCH=3
LR=3e-05
CUDA_VISIBLE_DEVICES=3 python train_unsloth.py \
    --proctitle "jaewook133/Qwen2.5" \
    --base_model "unsloth/Qwen2.5-3B-bnb-4bit" \
    --ckpt_dir "YOUR_MODEL_PATH/Qwen2.5-3B_translate_enko/ckpt" \
    --output_dir "YOUR_MODEL_PATH/Qwen2.5-3B_translate_enko/" \
    --output_gguf_dir "YOUR_MODEL_PATH/Qwen2.5-3B_translate_enko/gguf" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "linear"