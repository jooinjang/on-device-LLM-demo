EPOCH=1
LR=2e-04
CUDA_VISIBLE_DEVICES=3 python train_unsloth.py \
    --base_model "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" \
    --ckpt_dir "YOUR_MODEL_PATH/Qwen2.5-7B-Inst_translate_en-ko/ckpt" \
    --output_dir "YOUR_MODEL_PATH/Qwen2.5-7B-Inst_translate_en-ko/" \
    --output_gguf_dir "YOUR_MODEL_PATH/Qwen2.5-7B-Inst_translate_en-ko/gguf" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --logging_steps 1 \
    --lr_scheduler_type "linear"