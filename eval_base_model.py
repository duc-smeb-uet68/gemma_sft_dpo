import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

BASE_MODEL = "google/gemma-2b-it"
OUTPUT_FILE = "base_model_results.jsonl"
DATASET_SPLIT = "test"

print(">>> Bắt đầu tải model gốc và tokenizer...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model.eval()

print(">>> Tải model gốc thành công!")

print(f">>> Tải tập '{DATASET_SPLIT}' của dataset gsm8k...")
dataset = load_dataset("gsm8k", "main", split=DATASET_SPLIT)


def format_prompt(question):
    """Format prompt giống hệt như khi fine-tune"""
    return f"<start_of_turn>user\nSolve the following math problem step-by-step.\n{question}<end_of_turn>\n<start_of_turn>model\n"


print(f">>> Bắt đầu chạy inference model GỐC và lưu kết quả vào '{OUTPUT_FILE}'...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in tqdm(dataset, desc="Processing samples"):
        question = sample["question"]
        ground_truth_answer = sample["answer"]

        # Định dạng prompt và tokenize
        prompt_text = format_prompt(question)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Chạy model để sinh câu trả lời
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )

        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = full_generated_text.split("<start_of_turn>model\n")[-1]

        result_entry = {
            "question": question,
            "model_answer": model_answer,
            "ground_truth_answer": ground_truth_answer
        }

        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print(f"\n✅ Hoàn thành! Kết quả lưu trong file: {OUTPUT_FILE}")