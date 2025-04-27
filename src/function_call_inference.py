#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_call_infer_lora.py  —  推理 (Base 4-bit + LoRA Δ權重)
更新：使用完整匹配來嚴格判定 call 語句
"""

import argparse
import json
import re
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --------------------------------------------------
# 固定系統提示與嚴格匹配正則
# --------------------------------------------------
SYSTEM_PROMPT = (
    "你有一個工具 generate_image({prompt:str} → URL)。"
    "只有當使用者明確要求產生圖片時才呼叫；"
    "否則請用自然語言回答。"
)
# 完整匹配 call語句，避免任何額外文字
CALL_FULLMATCH = re.compile(
    r'^\s*call\(\s*["\']generate_image["\']\s*,\s*\{.*\}\s*\)\s*$',
    re.S
)

# --------------------------------------------------
# 分流函式
# --------------------------------------------------
def on_generate():
    """
    模型判定要生成圖片時執行
    """
    print("要產圖")


def on_no_generate(reply: str):
    """
    模型判定不生成圖片時執行
    """
    print("不必要產圖")
    print("Assistant 回答：", reply)

# --------------------------------------------------
# 建立 ChatML Prompt
# --------------------------------------------------
def build_prompt(user_input: str) -> str:
    """
    單回合推理 Prompt：只包含 system + user + assistant 開頭
    """
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

# --------------------------------------------------
# 判斷是否為生成呼叫
# --------------------------------------------------
def is_generate(reply: str) -> bool:
    """
    只有當回覆完全是 call(...) 才視為生成意圖
    """
    return bool(CALL_FULLMATCH.fullmatch(reply.strip()))

# --------------------------------------------------
# 主程式
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Llama-3 Function-Calling Inference (Base+LoRA)"
    )
    parser.add_argument(
        "--base_model", default="unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        help="4-bit base 模型路徑或 HF 名稱"
    )
    parser.add_argument(
        "--adapter_dir", required=True,
        help="LoRA Δ 權重資料夾 (save_method='lora')"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cuda / cpu / auto"
    )
    args = parser.parse_args()

    # 載入 4-bit base 模型
    print("► 載入 4-bit base 模型…")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto" if args.device == "auto" else args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 套用 LoRA adapter
    print("► 套用 LoRA adapter…")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    eos_id = tokenizer.eos_token_id

    print("=== 單回合互動模式 (Ctrl+C 離開) ===")
    while True:
        try:
            user_input = input("\n🧑‍💻 User > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue

        # 組 prompt
        prompt_str = build_prompt(user_input)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

        # 推理
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )[0]

        # 切出生成部分並 decode
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[prompt_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 移除可能殘留的 ChatML 標記
        if "<|im_end|>" in raw:
            raw = raw.split("<|im_end|>")[0]
        if "<|im_start|>" in raw:
            raw = raw.split("<|im_start|>")[0].strip()

        # 嚴格判定是否生成
        if is_generate(raw):
            on_generate()
        else:
            on_no_generate(raw)

if __name__ == "__main__":
    main()
