# *-- Data Handling --*
import os
import numpy as np

# *-- LLM model Training --*
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from unsloth import is_bf16_supported
import fire

# *-- Saving model --*
from transformers import AutoTokenizer, AutoModelForCausalLM

# *-- Logging --*
from setproctitle import setproctitle


def train(
    base_model: str = "unsloth/Llama-3.2-1B-bnb-4bit",
    ckpt_dir: str = "./ckpt",
    output_dir: str = "./output",
    output_gguf_dir: str = "./output_q",
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-4,
    warmup_steps: int = 20,
    logging_steps: int = 10,
    lr_scheduler_type: str = "cosine",
    proctitle: str = "jaewook133/LLaMA-3.2",
):
    setproctitle(proctitle)

    dataset = load_dataset("Jooinjang/my_translate_300k_en-ko", split="train")

    max_seq_length = 2048

    # load model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "o_proj",
            "gate_proj",
        ],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3047,
        loftq_config=None,
    )
    print(model.print_trainable_parameters())

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

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompt(examples):
        inputs = examples["en"]
        outputs = examples["ko"]
        texts = []
        for input_, output in zip(inputs, outputs):
            text = PROMPT.format(input_, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    training_data = dataset.map(formatting_prompt, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            output_dir=ckpt_dir,
            seed=0,
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # save quantized model checkpoint
    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="q4_k_m")
    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="q8_0")
    model.save_pretrained_gguf(output_gguf_dir, tokenizer, quantization_method="f16")

    print("Training completed. Model & Tokenizer saved to", output_dir)


if __name__ == "__main__":
    fire.Fire(train)
