# 🖼️ Generate-Image Intent Detector  (Llama3.1 8B + LoRA)

> 🔗單函式（call(generate_image)）觸發判別的監督微調專案
>   
> 🔗 Enhancing small LLMs with simplified “function calling” capability
> 
> 🔗輸入任意對話句，模型輸出「是否需要產生圖片」的模擬 function-calling 簡易範例模板
> 
---

## 環境需求
| 套件 | 版本 | 備註 |
| --- | --- | --- |
| Python | 3.10 | |
| transformers | 4.40 ↑ | |
| peft | 0.10 ↑ | |
| datasets | 3 ↑ | |
| bitsandbytes | 0.43 ↑ | 4-bit 量化 |
| tqdm | 4.66 ↑ | 進度條 |

```bash
pip install -r requirements.txt
```
或是
```bash
pip install openai
pip install tqdm
pip install torch torchvision torchaudio
pip install datasets
pip install "unsloth[gpu]"
pip install trl
pip install bitsandbytes accelerate
pip install transformers wandb
```
要注意盡量照順序下載套件，尤其是unsloth這行會有相依性問題如果順序反的話。

## 使用方法 How to Start
### Generate Datasets
```
cd project_root/src/
python generate_image_call_datasets.py
```
產生訓練&測試集在 `project_root/datas/`

### Lora-Finetuning
```
python run_sft.py \
  --data_file ../datas/image_call_train.jsonl \
  --train_ratio 0.9 \
  --model_name unsloth/Llama-3.1-8B-unsloth-bnb-4bit \
  --output_base ../model \
  --project_name function_sft \
  --device cuda:0
```
微調後的模型路徑為 `project_root/model/`

### Inference
#### 單句推論
```
python function_call_inference.py \
    --adapter_dir ../model/unsloth-Llama-3.1-8B-unsloth-bnb-4bit/adapter/checkpoint-56 \
    --base_model unsloth/Llama-3.1-8B-unsloth-bnb-4bit
```
#### 批次推論 & 評估
```
python evaluate_inference.py \
  --base_model unsloth/Llama-3.1-8B-unsloth-bnb-4bit \
  --adapter_dir ../model/unsloth-Llama-3.1-8B-unsloth-bnb-4bit_3000/adapter/checkpoint-135 \
  --test_file ../datas/image_call_test.jsonl
```

## 資料集構建

| 項目       | 數量   | 說明                       |
| ---------- | ------ | -------------------------- |
| Raw seeds pos   | 18      | 自己創的     |
| Raw seeds neg   | 15      | 自己創的     |
| GPT-aug pos     | 1500    | GPT-4 擴充並大略人工複審      |
| GPT-aug neg     | 1500    | GPT-4 擴充並大略人工複審      |
| **合計**   | **3000** | **正例 50 % / 反例 50 %** |

| 項目       | 數量   |
| ---------- | ------ | 
| Training dataset   | 2160    |
| Val dataset   | 240    |
| Testing dataset   | 600      |


- 欄位格式（JSONL）:
```
pos : {"messages": [{"role": "user", "content": "能否提供一幅吉卜力風格的藝術作品"}, {"role": "assistant", "content": "call(\"generate_image\", {\"prompt\": \"吉卜力風格插畫\"})"}]}
neg : {"messages": [{"role": "user", "content": "這張影像的作者是誰？"}, {"role": "assistant", "content": "作者是攝影師山田太郎，其作品多以人文街拍聞名。"}]}
```

- 做法 :
主要根據seed內容去call openai api gpt-4o-mini 產生額外4個相同意思、但不同說法的句子，並且可以把新產生的句子加入 seed 中一起循環新增JSONL。

- 成本 :
0.01$ input
0.05$ output

## 模型與訓練

### Base model
- `unsloth/Llama-3.1-8B-unsloth-bnb-4bit`  
- 4-bit NF4 量化（bitsandbytes）

### LoRA 設定

| 參數    | 值              |
| ------- | --------------- |
| r       | 32               |
| α       | 16              |
| dropout | 0.05             |
| target  | q_proj, k_proj, v_proj, o_proj  |

### 訓練超參

| 項目         | 值                         |
| ------------ | -------------------------- |
| Epochs       | 1                          |
| LR           | 2e-4                       |
| Batch        | 2                           |
| Gradient     | 4                           |
| Max length   | 1536                        |

###
![image](https://github.com/user-attachments/assets/ea9b40b8-9a97-43b4-af76-1d09fda8af3c)


## 評估結果
評估的方式是 inference **原始 Based Model + adapter**，節省硬體成本的方式。

| 指標          | 值      |
|---------------|---------|
| Precision     | 0.9753   |
| Recall        | 0.9049   |
| F1 Score      | 0.9388   |

### Confusion Matrix

|                     | Predicted Positive | Predicted Negative |
|---------------------|--------------------|--------------------|
| **Actual Positive** | TP = 276           | FN = 29            |
| **Actual Negative** | FP = 7             | TN = 288           |

### Model Prediction
```
Processing: 你能否為我畫一幅描繪「雪白的森林冬季」的作品？                                                                                           
  Gold Label: 生成                                                                                                                                   
  Predicted: 生成 (Model output: call("generate_image", {"prompt": "森林中覆滿白雪的雪景插畫"}))    

Processing: 吉卜力動畫所展現的風格特性包括哪些內容？                                                                                                 
  Gold Label: 不生成                                                                                                                                 
  Predicted: 不生成 (Model output: 吉卜力結合了日本古老，常見元素包括……) 
```

### 失敗案例結果 & 分析
```
Processing: 請根據這張照片創作一幅插圖。                                                                                                             
  Gold Label: 生成                                                                                                                                   
  Predicted: 不生成 (Model output: call("generate_image(<reference_photo_path>", {"prompt": "<generate_prompt>"}))  

Processing: 你能為我製作一張圖片嗎？                                                                                                                 
  Gold Label: 不生成                                                                                                                                 
  Predicted: 生成 (Model output: call("generate_image", {"prompt": "依據參數關鍵字製圖"}))

Processing: 請幫我在這張照片上加插圖                                                                                                                 
  Gold Label: 生成                                                                                                                                   
  Predicted: 不生成 (Model output: call("generate_image(<參數列表>)"))
```

1. 函式名稱後面多了 (<reference_photo_path>，導致我們用 fullmatch 嚴格符合 call("generate_image", {...}) 的正規判不出「生成」。因為我只用以下字串比對判斷是否call(generate_image)
```
CALL_FULLMATCH = re.compile(
    r'^\s*call\(\s*["\']generate_image["\']\s*,\s*\{.*\}\s*\)\s*$',
    re.S
)
```
2. 部分「負樣本」本身也很容易被視為生成意圖句子本身確實是在“請求製作一張圖片”，所以模型按直觀把它當成正樣本。這說明在訓練集裡，需要更多類似表達但「不生成」的負樣本來教模型區分何謂「生成新圖」 vs. “討論/生成文字”，所以最主因應該是datasets的問題。

## 反思
1. 由於資料集多元性的侷限，對極度口語或 emoji 句型或是英語性混搭應該仍有漏判。
2. 模型預測的準確度偏高有它的潛在性問題，因為測試集也是從訓練集分割出來的，所以都是從seed衍伸過來地就可能會有 overfitting 的情況。
3. 失敗案例主因來自於 match `call("generate_image", {...})`判斷的函式不夠完善。
4. 與題目要求 `handle_generate_image` 和 `generate_image_intent` 格式不太符合，是做完專案後才發現，希望與題目原意並沒有不同。
5. 採用unsloth加速我訓練的簡易性及程式撰寫方便性、更大的幫助了設備上的限制，GPU不支援8B的大小，memory太小了
6. trl+unsloth是我之前比較熟悉的開發流程
7. 過程中最大的問題是 LLama通常比較適應的簡易資料形式是 Alpaca，但希望可以讓語言模型更了解每個問句中的語意，選擇了ChatML，在程式某些部分就需要特別注意這塊的轉換。

## License

本專案採用 **MIT License with Commons Clause v1.0**，禁止任何商業銷售或收費使用。  
（非商業、個人或學術用途可自由使用／修改／散布）  
如需商業授權或許可，請聯絡 liavanenglish@gmail.com。  
Copyright © 2025 Liavan-0122, CHUN-MING YANG
