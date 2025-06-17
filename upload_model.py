from unsloth import FastLanguageModel
import fire


def upload(
    model_path: str,
    hf_path: str,
):
    MODEL_PATH = model_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_length=2048, load_in_4bit=True, dtype=None
    )

    model.push_to_hub_gguf(
        hf_path,
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
    )
    print("q4_k_m done")

    model.push_to_hub_gguf(
        hf_path,
        tokenizer=tokenizer,
        quantization_method="q8_0",
    )
    print("q8_0 done")

    model.push_to_hub_gguf(
        hf_path,
        tokenizer=tokenizer,
        quantization_method="f16",
    )
    print("f16 done")


if __name__ == "__main__":
    fire.Fire(upload)
