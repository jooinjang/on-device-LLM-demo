import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from setproctitle import setproctitle
import fire


def train(
    debug: bool = False,
    # model/data parameters
    base_model: str = "meta-llama/Llama-3.2-3B",
    bf16: bool = False,
    output_dir: str = "./output",
    ds_config_file: str = None,
    max_examples: int = None,
    # training hyperparameters
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    val_set_size: int = 2000,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 500,
    save_steps: int = 500,
    lr_scheduler_type: str = "cosine",
    # wandb parameters
    use_wandb: bool = False,
    wandb_project: str = "OLM_translation_en-ko",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
):
    setproctitle("jaewook133/LLaMA-3.2")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LLM for OLM with params:\n"
            f"{debug=}\n"
            f"{max_examples=}\n"
            f"{ds_config_file=}\n"
            f"base_model: {base_model}\n"
            f"{bf16=}\n"
            f"output_dir: {output_dir}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"val_set_size: {val_set_size}\n"
            f"{use_wandb=}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(base_model, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    dataset = load_dataset("Jooinjang/my_translate_300k_en-ko", split="train")

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    if val_set_size > 0:
        dataset = dataset.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_dataset, val_dataset = dataset["train"], dataset["test"]
    else:
        train_dataset = dataset
        val_dataset = None

    max_length = 2048

    # 토큰화 함수
    def tokenize_function_causal(examples):
        src_texts = examples["en"]
        tgt_texts = examples["ko"]

        # -----------------------------
        # 1) 우선 Prompt만 구성 (타깃 없이)
        # -----------------------------
        PROMPT = (
            "You are a professional English-to-Korean translator. "
            "Your task is to produce an accurate and natural-sounding Korean translation "
            "of the given English text while preserving the original meaning, tone, and context. "
            "Please do not add, remove, or alter any information.\n\n"
            "### English Text:\n"
            "{}\n\n"
            "### Korean Translation:\n"
            "{}"
        )

        # (1-A) 프롬프트 “영문만” 삽입 (번역문 자리엔 빈 문자열)
        prompt_only_texts = [PROMPT.format(en, "") for en in src_texts]

        # (1-B) 프롬프트만 토큰화 → prompt_length를 구하기 위해
        prompt_only_tokenized = tokenizer(
            prompt_only_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # 특별 토큰은 한꺼번에 토큰화할 때만 붙이도록
            return_tensors="pt",
        )

        # 각 샘플마다 실제 프롬프트 길이를 저장
        prompt_lengths = []
        pad_token_id = tokenizer.pad_token_id
        for input_ids_1sample in prompt_only_tokenized["input_ids"]:
            # 실제 토큰들 중에서 패딩(0 또는 tokenizer.pad_token_id) 아닌 부분만 세도 됨
            # 여기서는 단순히 (패딩 포함) 전체 길이에서부터, 맨 뒤 쪽 패딩 제외한 길이를 구하는 식
            # 일단은 len()을 그대로 써도 되지만, 정확히 “잘린 길이”를 알고 싶다면
            # (input_ids_1sample != pad_token_id).sum() 방식으로 계산 가능.
            prompt_lengths.append((input_ids_1sample != pad_token_id).sum())

        # -----------------------------
        # 2) 프롬프트 + 타깃(ko) 함께 토큰화
        # -----------------------------
        full_texts = [PROMPT.format(en, ko) for en, ko in zip(src_texts, tgt_texts)]
        tokenized = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]  # [batch_size, max_length]
        attention_mask = tokenized["attention_mask"]  # [batch_size, max_length]
        labels = input_ids.clone()  # 라벨은 일단 동일하게 복사

        # -----------------------------
        # 3) 프롬프트 구간 라벨을 -100 처리
        # -----------------------------
        for i, p_len in enumerate(prompt_lengths):
            # 만약 p_len이 max_length보다 작으면 해당 범위까지만 -100
            # (실제로는 모델이 pad_token_id를 쓰거나, 잘릴 수 있으니 최소화 처리)
            end_idx = min(p_len, max_length)
            labels[i, :end_idx] = -100  # 프롬프트 구간은 loss 계산하지 않음

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # 데이터 토큰화
    train_dataset = train_dataset.map(tokenize_function_causal, batched=True)
    if val_dataset:
        val_dataset = val_dataset.map(tokenize_function_causal, batched=True)

    # DataCollator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="max_length", max_length=max_length
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        deepspeed=ds_config_file if ds_config_file else None,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 학습 실행
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed. Model & Tokenizer saved to", output_dir)


if __name__ == "__main__":
    fire.Fire(train)
