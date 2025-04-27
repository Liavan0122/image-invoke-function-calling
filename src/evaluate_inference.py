#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_inference.py — 推論並評估 Generate Image Intent
"""

import argparse
import json
import re
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm   # ← 新增

# 嚴格匹配完整 call(generate_image)
CALL_FULLMATCH = re.compile(
    r'^\s*call\(\s*["\']generate_image["\']\s*,\s*\{.*\}\s*\)\s*$',
    re.S
)

SYSTEM_PROMPT = (
    "你有一個工具 generate_image({prompt:str} → URL)。"
    "只有當使用者明確要求產生圖片時才呼叫；"
    "否則請用自然語言回答。"
)

# 判定 reply 是否為完整 call
def is_generate(reply: str) -> bool:
    return bool(CALL_FULLMATCH.fullmatch(reply.strip()))

# 組 ChatML prompt，單回合
def build_prompt(user_input: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def main():
    parser = argparse.ArgumentParser(
        description="推論並評估 Generate Image Intent 模型"
    )
    parser.add_argument(
        "--base_model", required=True,
        help="4-bit base 模型路徑或 HF 名稱"
    )
    parser.add_argument(
        "--adapter_dir", required=True,
        help="LoRA adapter 目錄"
    )
    parser.add_argument(
        "--test_file", required=True,
        help="測試集 JSONL 檔，包含 messages 欄位"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cuda / cpu / auto"
    )
    args = parser.parse_args()

    # ── 1. 載入模型 ─────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    eos_id = tokenizer.eos_token_id

    # ── 2. 載入 test dataset ──────────────────────────────────
    ds = load_dataset("json", data_files={"test": args.test_file})["test"]

    y_pred, y_true = [], []

    # ── 3. 批次推理 ───────────────────────────────────────────
    for ex in tqdm(ds, desc="Evaluating", total=len(ds)):
        msgs = ex.get("messages", [])
        # 抽取最後 user 訊息
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        user_input = user_msgs[-1]["content"] if user_msgs else ""

        # gold label：最後 assistant content 是否為 call
        last_assist = [m for m in msgs if m.get("role") == "assistant"]
        gold = 1 if last_assist and is_generate(last_assist[-1]["content"]) else 0

        # 顯示處理狀態（不干擾進度條）
        tqdm.write(f"Processing: {user_input}")
        tqdm.write(f"  Gold Label: {'生成' if gold == 1 else '不生成'}")

        # 組 prompt 並推理
        prompt = build_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=40,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                temperature=0.0,
                do_sample=False
            )[0]

        # 切出模型新生成部分並 decode
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[prompt_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw = raw.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()

        pred = 1 if is_generate(raw) else 0
        tqdm.write(
            f"  Predicted: {'生成' if pred == 1 else '不生成'} "
            f"(Model output: {raw})"
        )

        y_pred.append(pred)
        y_true.append(gold)

    # ── 4. 指標計算 ───────────────────────────────────────────
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("\n=== Confusion Matrix ===")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print("\n=== Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
