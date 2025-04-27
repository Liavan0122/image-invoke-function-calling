# image-invoke-function-calling
自動判斷並輸出是否需要調用Generate Image的二元分類結果


# 🖼️ Generate-Image Intent Detector  (Gemma-1B + LoRA)

> 單函式（call(generate_image)）觸發判別的監督微調專案  
> 🔗 *Enhancing small LLMs with simplified “function calling” capability*

---

## 0. 專案簡述
- **目標**：輸入任意對話句，模型輸出「是否需要產生圖片」的 JSON 模板  
- **特色**  
  1. 使用 **Gemma-3-1B-instruct** 4-bit 量化 + **LoRA** 微調  
  2. ~2 k 句平衡標註資料，F1 ≈ 0.92  
  3. 完整腳本：資料前處理 → SFT → 推論路由器 → 評估

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

## 使用方法
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
| **合計**   | **3000** | **正例 51 % / 反例 49 %** |

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
主要根據seed內容去產生額外4個相同意思、但不同說法的句子，並且可以把新產生的句子加入 seed 中一起循環新增JSONL。

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


## 評估結果

| 指標               | 值      |
|--------------------|---------|
| Precision (正例)    | 0.9753   |
| Recall (正例)       | 0.9049   |
| F1 Score            | 0.9388   |

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

## License

本專案採用 [MIT License](LICENSE) 授權，詳情請見專案根目錄的 `LICENSE` 檔案。  
Copyright © 2025 Liavan-0122, CHUN-MING YANG
