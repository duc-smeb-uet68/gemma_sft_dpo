import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig


BASE_MODEL = "google/gemma-2b-it"
SFT_ADAPTER = "/home/lhduc205/dpo_gemma/gemma-2b-gsm8k-finetuned/checkpoint-935"
DPO_OUTPUT_DIR = "./gemma-2b-gsm8k-dpo"
DPO_DATA = "dpo_gsm8k_pairs.jsonl"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)



model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)


model = PeftModel.from_pretrained(model, SFT_ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DPO_DATA)["train"]


training_args = DPOConfig(
    output_dir=DPO_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    beta=0.1,
    fp16=True,
    # bf16 = True,
    logging_steps=25,
    push_to_hub=True,
    hub_model_id="hongduc05/gemma-2b-gsm8k-dpo",
    save_strategy="epoch",
    hub_strategy="every_save",
    gradient_checkpointing=True,
)


trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(DPO_OUTPUT_DIR)

print("DPO training complete. Model saved at:", DPO_OUTPUT_DIR)
