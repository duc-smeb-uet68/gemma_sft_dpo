from datasets import load_dataset


from huggingface_hub import login
login(token="HF_TOKEN")

dataset_name = "gsm8k"
dataset = load_dataset(dataset_name, "main")


def create_prompt(sample):

    prompt = f"""<start_of_turn>user
Solve the following math problem step-by-step.
{sample['question']}<end_of_turn>
<start_of_turn>model
{sample['answer']}<end_of_turn>"""
    return {"text": prompt}


formatted_dataset = dataset.map(create_prompt)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_id = "google/gemma-2b-it"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Tải mô hình với cấu hình quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)



peft_model = get_peft_model(model, lora_config)

from transformers import TrainingArguments
from trl import SFTTrainer



# Cấu hình các tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./gemma-2b-gsm8k-finetuned", # Thư mục lưu model
    num_train_epochs=1, # Huấn luyện trong 1 epoch là đủ để thấy kết quả tốt
    per_device_train_batch_size=2, # Batch size nhỏ để vừa VRAM
    gradient_accumulation_steps=4, # Tích lũy gradient để mô phỏng batch size lớn hơn (2*4=8)
    gradient_checkpointing=True, # Kỹ thuật tiết kiệm VRAM bằng cách không lưu toàn bộ activation
    optim="paged_adamw_8bit", # Sử dụng AdamW optimizer được tối ưu hóa cho quantization
    logging_steps=25, # Ghi log sau mỗi 25 bước
    save_strategy="epoch", # Lưu model sau mỗi epoch
    learning_rate=2e-4,
    fp16=True, # Sử dụng mixed precision (tính toán 16-bit)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model=peft_model,
    train_dataset=formatted_dataset["train"],
    peft_config=lora_config, # Độ dài tối đa của chuỗi đầu vào
    args=training_args,
)

trainer.train()