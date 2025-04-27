#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised Fine-Tuning Pipeline (Function-Calling ChatML, Unsloth)
"""

import os
import unsloth
import argparse
import logging
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
import wandb

# --------------------------------------------------
# 參數
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Pipeline for Function-Calling ChatML")
    parser.add_argument("--data_file",   type=str, required=True, help="JSONL 檔 (含 messages)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train / Val split ratio")
    parser.add_argument("--model_name",  type=str, default="unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit",
                        help="HuggingFace model 名稱")
    parser.add_argument("--output_base", type=str, default="../model", help="輸出根目錄")
    parser.add_argument("--project_name",type=str, default="function_sft", help="WandB 專案名稱")
    parser.add_argument("--device",      type=str, default="cuda", help="cuda 或 cuda:0")
    return parser.parse_args()

# --------------------------------------------------
# Log
# --------------------------------------------------
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

# --------------------------------------------------
# 資料處理：ChatML ➜ text
# --------------------------------------------------
def load_prepare_dataset(data_file: str, train_ratio: float, tokenizer) -> DatasetDict:
    logging.info("Loading dataset…")
    ds = load_dataset("json", data_files={"full": data_file})["full"]

    # 套用 ChatML template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={
            "role":      "role",
            "content":   "content",
            "user":      "user",
            "assistant": "assistant",
            "system":    "system",   # 若資料裡沒有 system，可以照寫；不會有副作用
        },
    )

    def _to_text(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(m,
                                              tokenize=False,
                                              add_generation_prompt=False)
                for m in batch["messages"]
            ]
        }

    logging.info("Converting messages ➜ text…")
    ds = ds.map(_to_text, batched=True, num_proc=2, remove_columns=ds.column_names)

    split = ds.train_test_split(test_size=1 - train_ratio, seed=42)
    return DatasetDict(train=split["train"], validation=split["test"])

# --------------------------------------------------
# Model & Tokenizer
# --------------------------------------------------
def build_model_and_tokenizer(model_name: str, device: str):
    logging.info(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map=device,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        use_gradient_checkpointing=True,
    )
    return model, tokenizer

# --------------------------------------------------
# Train
# --------------------------------------------------
def train(model, tokenizer, datasets: DatasetDict, args):
    wandb.init(project=args.project_name, name=args.model_name.replace("/", "-"))

    safe_id     = args.model_name.replace("/", "-")
    adapter_dir = os.path.join(args.output_base, safe_id, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        dataset_text_field="text",
        max_seq_length=1536,
        packing=True,
        dataset_num_proc=2,
        args=TrainingArguments(
            output_dir               = adapter_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size =2,
            gradient_accumulation_steps=4,
            warmup_steps             =100,
            num_train_epochs         =1,
            optim                    ="adamw_8bit",
            learning_rate            =2e-4,
            lr_scheduler_type        ="cosine",
            fp16                     = not is_bfloat16_supported(),
            bf16                     = is_bfloat16_supported(),
            logging_steps            =20,
            save_steps               =60,
            report_to                =["wandb"],
        ),
    )

    logging.info("Start training…")
    trainer.train()
    logging.info("Training finished!")

# --------------------------------------------------
# Merge & Save
# --------------------------------------------------
def merge_and_save(model, tokenizer, args):
    safe_id    = args.model_name.replace("/", "-")
    merged_dir = os.path.join(args.output_base, safe_id, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        # save_method = "merged_16bit",   # 也可換成 "merged_4bit" 或 "lora"

        # save_method="merged_4bit_forced",   
        # safe_merge=True,                    # 保留校驗

        save_method="lora",
    )

    logging.info(f"Merged model saved to {merged_dir}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    setup_logging()

    model, tokenizer = build_model_and_tokenizer(args.model_name, args.device)
    datasets         = load_prepare_dataset(args.data_file, args.train_ratio, tokenizer)

    ## For check
    sample = datasets["train"][0]          # 取第一筆
    print("\n=== Preview ===\n", sample)    # 或 print(sample["text"][:400])

    train(model, tokenizer, datasets, args)
    merge_and_save(model, tokenizer, args)

if __name__ == "__main__":
    main()
