import json
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import KTOTrainer, KTOConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = "kto_dataset.json"
OUTPUT_DIR = "my_real_estate_model"

def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        formatted_data = []
        for entry in raw_data:
            formatted_data.append({
                "prompt": [{"role": "user", "content": entry["prompt"]}],
                "completion": [{"role": "assistant", "content": entry["completion"]}],
                "label": entry["label"]
            })
            
        return Dataset.from_list(formatted_data)
    except FileNotFoundError:
        print("Train file not found")
        return None

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",  
        torch_dtype=torch.float32 
    )

    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] 
    )

    dataset = load_data()
    if not dataset: 
        return

    kto_config = KTOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=10,
        beta=0.1,
        max_length=512, 
        max_prompt_length=256,
        
        use_cpu=True,    
        fp16=False,     
        bf16=False,       
    )

    print("Training...")
    
    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=dataset,
        processing_class=tokenizer, 
        peft_config=peft_config,
    )

    trainer.train()

    print(f"Model saved: {OUTPUT_DIR}")
train()