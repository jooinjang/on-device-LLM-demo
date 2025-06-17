# *-- Data Handling --*
import os
import numpy as np

# *-- LLM model Training --*
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from unsloth import is_bf16_supported
import fire

# *-- Logging --*
from setproctitle import setproctitle


SYSTEM_PROMPT = (
    "You are a highly capable bilingual translation assistant, skilled in both Korean and English."
    "your task is to accurately translate the given text while preserving meaning, tone, and style. Follow these guidelines carefully:\n\n"
    "1. If the given text is in Korean, translate it into English. If the given text is in English, translate it into Korean.\n"
    "2. Preserve the original meaning, Maintain the tone and style of the given text.\n"
    "3. **Proper nouns** (e.g. names, places, organizations) should remain in their original form unless widely accepted translations exist."
)


def train(
    base_model: str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    ckpt_dir: str = "./ckpt",
    output_dir: str = "./output",
    output_gguf_dir: str = "./output/gguf",
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    lr_scheduler_type: str = "cosine",
    proctitle: str = "jaewook133 - {}",
    load_in_4bit: bool = True,
):
    setproctitle(proctitle.format(base_model))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # Unsloth support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    if "llama-3.1" in base_model.lower() or "llama-3.2" in base_model.lower():
        chat_template = "llama-3.1"
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen2.5" in base_model.lower():
        chat_template = "qwen2.5"
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
    elif "deepseek" in base_model.lower():
        chat_template = "zephyr"
        instruction_part = "<|User|>"
        response_part = "<|Assistant|>"
    else:
        print("Undefined Base Model: {}".format(base_model))
        return

    tokenizer = get_chat_template(tokenizer, chat_template)

    def formatting_prompt_func(examples):
        samples = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in samples
        ]
        return {
            "text": texts,
        }

    # 데이터셋을 messages로 묶기
    dataset = load_dataset("Jooinjang/translation_241230_enko")["train"]
    messages = []
    for i in dataset:
        message = []
        message.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        )
        message.append(
            {
                "role": "user",
                "content": "*** Input Text:\n" + i["input_text"],
            }
        )
        message.append(
            {
                "role": "assistant",
                "content": "*** Translation:\n" + i["output_text"],
            }
        )
        messages.append(message)

    message_ds = Dataset.from_dict({"messages": messages})

    # Chat Template 적용
    dataset = message_ds.map(
        formatting_prompt_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir=ckpt_dir,
            seed=3407,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part,
        response_part,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="q4_k_m")
    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="q8_0")
    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="f16")


if __name__ == "__main__":
    fire.Fire(train)
