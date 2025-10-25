import re, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "google/gemma-2b-it"
SFT_ADAPTER = "/home/lhduc205/dpo_gemma/gemma-2b-gsm8k-finetuned/checkpoint-935"
OUT_FILE = "dpo_gsm8k_pairs.jsonl"
NUM_SAMPLES = 4   # sinh 4 hoàn thành để chọn cặp
MAX_ITEMS = 2000  # giới hạn tạo (bạn có thể tăng dần)


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

model = PeftModel.from_pretrained(model, SFT_ADAPTER)
model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

model.eval()
model.config.use_cache = False

def format_prompt(q):
    return f"<start_of_turn>user\nSolve the following math problem step-by-step.\n{q}<end_of_turn>\n<start_of_turn>model\n"

# helper: trích số cuối cùng trong text (heuristic)
def extract_last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

ds = load_dataset("gsm8k", "main", split="train", token = 'HF_TOKEN')

count = 0
BATCH_SIZE = 8

prompts_batch = []
samples_batch = []

with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for i, sample in enumerate(ds):
        if i >= MAX_ITEMS: break

        # Thêm prompt và sample vào lô hiện tại
        prompts_batch.append(format_prompt(sample["question"]))
        samples_batch.append(sample)

        # Khi lô đã đủ lớn hoặc đã đến cuối dataset thì xử lý
        if len(prompts_batch) == BATCH_SIZE or i == MAX_ITEMS - 1:
            # Generate nhiều câu trả lời cho cả lô
            gens = []
            for seed in range(NUM_SAMPLES):
                torch.manual_seed(seed + 1337 + i)
                inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True).to(model.device)
                outs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    max_new_tokens=200,
                    num_return_sequences=1,
                )
                # Decode kết quả cho cả lô
                texts = tokenizer.batch_decode(outs, skip_special_tokens=True)

                # Sắp xếp lại kết quả theo đúng thứ tự của prompt
                if not gens:
                    gens = [[text] for text in texts]
                else:
                    for idx, text in enumerate(texts):
                        gens[idx].append(text)

            # Xử lý kết quả cho từng sample trong lô
            for j in range(len(prompts_batch)):
                current_gens = gens[j]
                current_sample = samples_batch[j]

                gold = extract_last_number(current_sample["answer"])
                if gold is None:
                    continue

                correct = [g for g in current_gens if extract_last_number(g) == gold]
                wrong = [g for g in current_gens if extract_last_number(g) != gold]

                if len(correct) >= 1 and len(wrong) >= 1:
                    chosen = correct[0]
                    rejected = wrong[0]
                    fout.write(json.dumps({"prompt": prompts_batch[j], "chosen": chosen, "rejected": rejected},
                                          ensure_ascii=False) + "\n")
                    count += 1

            # Reset lại lô
            prompts_batch = []
            samples_batch = []

