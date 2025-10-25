import torch
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "google/gemma-2b-it"
SFT_ADAPTER = "./gemma-2b-gsm8k-finetuned/checkpoint-935"
OUTPUT_FILE = "sft_model_results.jsonl"
DATASET_SPLIT = "test"
BATCH_SIZE = 2
MAX_NEW_TOKENS = 512

print(">>> Bắt đầu tải model và tokenizer...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype= torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load SFT adapter và merge
print(f">>> Tải SFT adapter từ: {SFT_ADAPTER}")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
model = model.merge_and_unload()  # Merge adapter vào base model

model.eval()

print(">>> Tải model SFT thành công!")

try:
    print(">>> Áp dụng torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")
    print(">>> torch.compile() áp dụng thành công!")
except Exception as e:
    print(f"⚠️ Không thể áp dụng torch.compile(): {e}. Tiếp tục mà không compile.")


print(f">>> Tải tập '{DATASET_SPLIT}' của dataset gsm8k...")
dataset = load_dataset("gsm8k", "main", split=DATASET_SPLIT)


def format_prompt(question):
    return f"<start_of_turn>user\nSolve the following math problem step-by-step.\n{question}<end_of_turn>\n<start_of_turn>model\n"

def prepare_data(batch):
    # Tạo prompt cho từng câu hỏi trong batch
    prompts = [format_prompt(q) for q in batch['question']]
    # Tokenize cả batch prompts
    tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False) # Không nên truncate prompt
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'questions': batch['question'], # Giữ lại để ghi vào file output
        'ground_truth_answers': batch['answer'] # Giữ lại để ghi vào file output
    }

# Sử dụng DataLoader để tạo batch
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: prepare_data(dict(zip(dataset.column_names, zip(*[sample.values() for sample in batch])))))
total_batches = len(dataloader)

print(f">>> Bắt đầu chạy inference model SFT và lưu kết quả vào '{OUTPUT_FILE}'...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    # Sử dụng tqdm(dataloader) để có thanh tiến trình
    for batch in tqdm(dataloader, total=total_batches, desc="Processing batches"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        # Chạy model để sinh câu trả lời cho cả batch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id # Sử dụng pad_token_id đã thiết lập
            )

        # Decode kết quả cho từng sample trong batch
        # outputs chứa cả prompt và phần generated. Cần loại bỏ prompt.
        prompt_lengths = [len(ids) for ids in input_ids] # Độ dài prompt của từng sample (sau khi padding)
        generated_texts_full = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Tách lấy phần trả lời của model cho từng sample
        model_answers = []
        original_questions = batch['questions']
        ground_truth_answers_batch = batch['ground_truth_answers']

        for i, full_text in enumerate(generated_texts_full):
            # Tái tạo prompt gốc để tách chính xác hơn
            original_prompt = format_prompt(original_questions[i])
            # Tìm vị trí kết thúc của prompt trong text được sinh ra (loại bỏ padding bên trái)
            # Cách đơn giản hơn: Tách bằng thẻ đặc biệt
            parts = full_text.split("<start_of_turn>model\n")
            if len(parts) > 1:
                model_answer = parts[-1].strip()
            else: # Trường hợp generate không thành công hoặc prompt bị lỗi
                model_answer = ""
            model_answers.append(model_answer)

            # Tạo và ghi kết quả cho từng sample
            result_entry = {
                "question": original_questions[i],
                "model_answer": model_answer,
                "ground_truth_answer": ground_truth_answers_batch[i]
            }
            f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

print(f"\n✅ Hoàn thành! Kết quả lưu trong file: {OUTPUT_FILE}")