from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


MODEL_PATH = [
    "/mnt/raid6/jaewook133/models/Llama-3.1-8B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Llama-3.2-1B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Llama-3.2-3B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Qwen2.5-0.5B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Qwen2.5-1.5B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Qwen2.5-3B-Inst_translate_en-ko",
    "/mnt/raid6/jaewook133/models/Qwen2.5-7B-Inst_translate_en-ko",
]


print("===== { MODEL LIST } =====")
for idx, model_path in enumerate(MODEL_PATH):
    print(f"({idx}): {model_path}")

num = input("\nselect model: ")
print("\nSelected Model:", MODEL_PATH[int(num)])

selected = MODEL_PATH[int(num)]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=selected, max_seq_length=4096, load_in_4bit=True, device_map="auto"
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

if "llama-3.1" in selected.lower() or "llama-3.2" in selected.lower():
    chat_template = "llama-3.1"
elif "qwen2.5" in selected.lower():
    chat_template = "qwen2.5"

tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

with open("temp.txt", "r") as f:
    temp = f.readlines()

input_text = "".join(temp)

SYSTEM_PROMPT = (
    "You are a highly capable bilingual translation assistant, skilled in both Korean and English."
    "your task is to accurately translate the given text while preserving meaning, tone, and style. Follow these guidelines carefully:\n\n"
    "1. If the given text is in Korean, translate it into English. If the given text is in English, translate it into Korean.\n"
    "2. Preserve the original meaning, Maintain the tone and style of the given text.\n"
    "3. **Proper nouns** (e.g. names, places, organizations) should remain in their original form unless widely accepted translations exist."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "*** Input Text:\n" + input_text},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_token=True, return_tensors="pt"
).to("cuda")

# outputs = model.generate(
#     input_ids=inputs, max_new_tokens=4096, use_cache=True, temperature=0.7
# )

# print("Model Output:\n", tokenizer.batch_decode(outputs))

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=4096,
    use_cache=True,
    temperature=0.9,
)
