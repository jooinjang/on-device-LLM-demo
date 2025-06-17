export WANDB_NAME="OLM-translate_en-ko"
EPOCH=1
LR=5e-05
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=23456 --nproc_per_node=4 train.py \
    --base_model "meta-llama/Llama-3.2-3B" \
    --output_dir "YOUR_MODEL_PATH/llama3.2-3B_translate_enko_ver1" \
    --ds_config_file "ds_config.json" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --val_set_size 5000 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \