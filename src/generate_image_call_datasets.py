#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根據 seed.jsonl 透過 GPT 改寫，擴充出平衡的 image-function-calling 資料集。
"""

import os
import json
import random
import time
from pathlib import Path

import openai
from tqdm import tqdm   # 進度條；若無可移除

# ==============================
# 全域設定
# ==============================
SEED_PATH   = Path(__file__).resolve().parent.parent / "datas" / "seed.jsonl"

# 產生 train / test 兩個檔案
OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "datas"
TRAIN_PATH  = OUTPUT_DIR / "image_call_train.jsonl"
TEST_PATH   = OUTPUT_DIR / "image_call_test.jsonl"

TARGET_EACH   = 1500        # 正例 / 反例 各多少筆
TEMPERATURE   = 0.8         # GPT 改寫隨機度
TRAIN_RATIO   = 0.8           # train : test = 8 : 2
N_VARIANTS    = 4           # 每次改寫產生幾句
MODEL_NAME    = "gpt-4o-mini"

# ==============================
# 工具函式
# ==============================

def load_seed(seed_path: Path) -> list[dict]:
    """
    讀取 seed.jsonl，回傳 list[dict]。
    """
    with seed_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_seed(seed: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    依照 assistant 內容是否含 call("generate_image"... 分成正 / 反例。
    {"messages":[{"role":"user","content":"我想產一張學生高中生的圖片"},{"role":"assistant","content":"call(\"generate_image\", {\"prompt\": \"學生、高中生、人物照片\"})"}]}
    """
    pos, neg = [], []
    for s in seed:
        if 'call("generate_image"' in s["messages"][1]["content"]:
            pos.append(s)
        else:
            neg.append(s)
    return pos, neg


def augment(sample: dict, n: int = N_VARIANTS) -> list[dict]:
    """
    針對一條 seed，讓 GPT 產生 n 條多樣化同義句，
    並保持「想要生成圖片」的意圖不變。
    """
    user_text      = sample["messages"][0]["content"]
    assistant_text = sample["messages"][1]["content"]

    # ---------- Prompt ----------
    prompt = f"""
請產生 {n} 條與「{user_text}」意思相同、但用不同說法的句子。
僅回傳 JSON 陣列，例如：
["換句話說……", "另一種講法……"]
不要加入任何其他文字。
"""
#     prompt = f"""
# 你是句子改寫器，請同時產生 {n} 條多樣化版本，並滿足：
# 1. **祈使**語氣 1 條
# 2. **疑問**語氣 1 條
# 3. **正式**文體 1 條
# 4. **口語**文體 1 條
# 另外至少 **一條** 中英混寫或帶 Emoji。
# 請務必保留「想要生成圖片」的意圖。

# 僅回傳 JSON 陣列，例如：
# ["換句話說……", "另一種講法……"]
# 不要加入任何其他文字。
# """

    while True:  # 發生 API 錯誤就重試
        try:
            print("\n")
            print(prompt)  # for test
            res = openai.chat.completions.create(
                model      = MODEL_NAME,
                messages   = [{"role": "user", "content": prompt}],
                temperature= TEMPERATURE
            )
            variants = json.loads(res.choices[0].message.content)
            break
        except Exception as e:
            print("API error, retrying:", e)
            time.sleep(3)

    # 封裝成符合 messages 格式
    new_samples = []
    for v in variants:
        new_samples.append({
            "messages": [
                {"role": "user",      "content": v},
                {"role": "assistant", "content": assistant_text}
            ]
        })
    return new_samples


def build_bucket(bucket_name: str, seed_bucket: list[dict]) -> list[dict]:
    """
    不斷對 seed_bucket 做改寫，直到累積 TARGET_EACH 筆。
    """
    cur = seed_bucket.copy()
    idx = 0
    with tqdm(total=TARGET_EACH, desc=f"Augmenting {bucket_name}", ncols=80) as pbar:
        pbar.update(len(cur))             # 先計入原始種子數
        while len(cur) < TARGET_EACH:
            cur += augment(seed_bucket[idx % len(seed_bucket)])
            idx += 1
            pbar.update(n=len(cur) - pbar.n)
    return cur[:TARGET_EACH]              # 截到剛好

def build_bucket_recursive(bucket_name: str,
                           seed_bucket: list[dict],
                           target_each: int = TARGET_EACH) -> list[dict]:
    """
    依序改寫句子，且把「新生成的樣本」也丟回樣本池，直到累積 target_each 筆。

    風險：可能逐輪偏離原意，請務必在最終輸出前抽驗品質。
    """
    # cur：目前已收集的樣本；pool：可被抽樣改寫的母體（持續成長）
    pool = seed_bucket.copy()
    cur  = seed_bucket.copy()

    with tqdm(total=target_each,
              desc=f"Augmenting {bucket_name} (recursive)",
              ncols=80) as pbar:

        pbar.update(len(cur))                     # 先計入原始種子

        idx = 0
        while len(cur) < target_each:
            # 1. 從「會成長的 pool」按 round-robin 取一句來改寫
            base_sample = pool[idx % len(pool)]
            new_samples = augment(base_sample)    # augment() 產生 N_VARIANTS 條

            # 2. 把新句子同時放進 cur 與 pool
            pool.extend(new_samples)
            cur.extend(new_samples)

            # 3. 進度條 & 指標更新
            pbar.update(len(new_samples))
            idx += 1

    return cur[:target_each]                      # 若超出則截斷

# ==============================
# 主流程
# ==============================

def main() -> None:
    # 建議仍用環境變數，這裡為了示範保留你的寫法
    openai.api_key = ""
    if not openai.api_key:
        raise RuntimeError("請先設定 OPENAI_API_KEY！")

    # 1. 讀入種子
    seed = load_seed(SEED_PATH)
    pos_seed, neg_seed = split_seed(seed)

    # 2. 生成正 / 反例（遞迴或原版皆可）
    pos_aug = build_bucket_recursive("pos", pos_seed)
    neg_aug = build_bucket_recursive("neg", neg_seed)

    # 3. 混洗 & 切分
    dataset = pos_aug + neg_aug
    random.shuffle(dataset)

    split_idx    = int(len(dataset) * TRAIN_RATIO)
    train_set    = dataset[:split_idx]
    test_set     = dataset[split_idx:]

    # 4. 輸出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, subset in [(TRAIN_PATH, train_set), (TEST_PATH, test_set)]:
        with path.open("w", encoding="utf-8") as f:
            for item in subset:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    print(f"✅  Done! 產生筆數：{len(dataset)}")
    print(f"   ├── train : {len(train_set)} → {TRAIN_PATH}")
    print(f"   └── test  : {len(test_set)}  → {TEST_PATH}")


if __name__ == "__main__":
    main()
