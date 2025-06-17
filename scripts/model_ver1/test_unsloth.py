from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
import torch

MODEL_PATH = "YOUR_MODEL_PATH/Llama3.1-8B_translate_enko_unsloth/" # replace with your model path
print("MODEL_PATH:", MODEL_PATH)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=4096, load_in_4bit=True, device_map="auto"
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

with open("temp.txt", "r") as f:
    temp = f.readlines()

en = "".join(temp)

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

inputs = tokenizer([PROMPT.format(en, "")], return_tensors="pt").to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
answer = tokenizer.batch_decode(outputs)
answer = answer[0].split("information.\n\n")[-1]
print("\n\nTranslated(Korean):", answer)
