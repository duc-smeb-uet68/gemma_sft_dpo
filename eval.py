import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


BASE_MODEL = "google/gemma-2b-it"
DPO_ADAPTER = "./gemma-2b-gsm8k-dpo"
OUTPUT_FILE = "dpo_model_results.jsonl"
DATASET_SPLIT = "test"

print(">>> Bắt đầu tải model và tokenizer...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


model = PeftModel.from_pretrained(base_model, DPO_ADAPTER)
model = model.merge_and_unload()

model.eval()

print(">>> Tải model thành công!")

print(f">>> Tải tập '{DATASET_SPLIT}' của dataset gsm8k...")
dataset = load_dataset("gsm8k", "main", split=DATASET_SPLIT)


def format_prompt(question):
    return f"<start_of_turn>user\nSolve the following math problem step-by-step.\n{question}<end_of_turn>\n<start_of_turn>model\n"


print(f">>> Bắt đầu chạy inference và lưu kết quả vào file '{OUTPUT_FILE}'...")

# Mở file để ghi (chế độ 'w' - write)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in tqdm(dataset, desc="Processing samples"):
        question = sample["question"]
        ground_truth_answer = sample["answer"]


        prompt_text = format_prompt(question)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Chạy model để sinh câu trả lời
        # torch.no_grad() để không tính toán gradient, tiết kiệm bộ nhớ và tăng tốc
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Giới hạn độ dài câu trả lời
                do_sample=False,  # Sử dụng greedy decoding để kết quả nhất quán
                pad_token_id=tokenizer.eos_token_id  # Quan trọng để tránh warning
            )
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        model_answer = full_generated_text.split("<start_of_turn>model\n")[-1]

        result_entry = {
            "question": question,
            "model_answer": model_answer,
            "ground_truth_answer": ground_truth_answer
        }

        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print(f"\n finish, result in file: {OUTPUT_FILE}")