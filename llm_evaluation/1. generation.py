from datasets import load_dataset
import openai
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
import fire
import json
import os


SYSTEM_PROMPT = (
    "You are a highly capable bilingual translation assistant, skilled in both Korean and English."
    "your task is to accurately translate the given text while preserving meaning, tone, and style. Follow these guidelines carefully:\n\n"
    "1. If the given text is in Korean, translate it into English. If the given text is in English, translate it into Korean.\n"
    "2. Preserve the original meaning, Maintain the tone and style of the given text.\n"
    "3. **Proper nouns** (e.g. names, places, organizations) should remain in their original form unless widely accepted translations exist."
)


def main(
    model_path: str,
    output_path: str,
    dataset_name: str = None,
):
    # 환경변수에서 설정값 가져오기
    if dataset_name is None:
        dataset_name = os.getenv('EVALUATION_DATASET', 'Jooinjang/translation_241230_enko')
    
    device_map = os.getenv('DEVICE_MAP', 'auto')
    max_seq_length = int(os.getenv('MAX_SEQ_LENGTH', '4096'))
    max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', '4096'))
    temperature = float(os.getenv('TEMPERATURE', '0.9'))
    
    # 1) 모델/토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, 
        max_seq_length=max_seq_length, 
        load_in_4bit=True, 
        device_map=device_map
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    # 2) 모델 유형에 맞춰 Chat 템플릿 선택
    if "llama-3.1" in model_path.lower() or "llama-3.2" in model_path.lower():
        chat_template = "llama-3.1"
    elif "qwen2.5" in model_path.lower():
        chat_template = "qwen2.5"
    else:
        # 필요 시 기본값 지정
        chat_template = "llama-3.1"

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # 3) 데이터셋 로드
    dataset = load_dataset(dataset_name, split="test")

    # 4) 번역 결과를 저장할 리스트
    translation_results = []

    # 'input_text' 컬럼만 사용 (dataset["input_text"]가 리스트 형태로 반환된다고 가정)
    texts = dataset["input_text"]

    # 5) 배치 대신, 각 문장별로 번역 수행
    for t in tqdm(texts, desc="Translating"):
        # 시스템 메시지 + 유저 메시지를 하나씩 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "*** Input Text:\n" + t},
        ]

        # 6) 토크나이징
        # apply_chat_template는 내부적으로 chat 형식(prompt)을 맞춰 토크나이징해주는 함수
        inputs = tokenizer.apply_chat_template(
            [messages],  # 한 문장만 리스트로 감싸서 전달
            tokenize=True,
            add_generation_token=True,
            return_tensors="pt",
        ).to("cuda")

        # 7) 모델 생성
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
        )

        # 8) 생성 결과 디코딩
        # outputs에는 하나의 시퀀스만 들어있으므로 outputs[0] 디코딩
        decoded = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

        translation_results.append({"input": t, "result": decoded})

    # 9) 결과 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translation_results, f, ensure_ascii=False)

    print(f"Done! Results saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
