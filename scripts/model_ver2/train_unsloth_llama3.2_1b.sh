EPOCH=1
LR=2e-04
CUDA_VISIBLE_DEVICES=5 python train_unsloth.py \
    --base_model "unsloth/Llama-3.2-1B-Instruct" \
    --ckpt_dir "YOUR_MODEL_PATH/Llama-3.2-1B-Inst_16bit_translate_en-ko/ckpt" \
    --output_dir "YOUR_MODEL_PATH/Llama-3.2-1B-Inst_16bit_translate_en-ko/" \
    --output_gguf_dir "YOUR_MODEL_PATH/Llama-3.2-1B-Inst_16bit_translate_en-ko/gguf" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --load_in_4bit False

# --base_model "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" \