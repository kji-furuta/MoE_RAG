# クラウドGPUでLoRA→GGUF変換ガイド

## 推奨クラウドサービス

### 1. Google Colab Pro+ (推奨)
- **GPU**: A100 40GB/80GB
- **料金**: 月額$49.99
- **メリット**: 簡単、Google Drive連携
- **URL**: https://colab.research.google.com/

### 2. Kaggle (無料枠あり)
- **GPU**: P100 16GB / T4 16GB (無料)、A100 40GB (有料)
- **料金**: 週30時間無料
- **メリット**: 無料枠が大きい
- **URL**: https://www.kaggle.com/

### 3. Paperspace Gradient
- **GPU**: A100 80GB
- **料金**: 時間単位（$3.09/時）
- **メリット**: 最大メモリ
- **URL**: https://www.paperspace.com/

## 事前準備

### 1. LoRAアダプタの準備
```bash
# WSL2でLoRAアダプタを圧縮
cd /home/kjifu/MoE_RAG/outputs
tar -czf lora_20250830_223432.tar.gz lora_20250830_223432/

# サイズ確認（約833MB）
ls -lh lora_20250830_223432.tar.gz
```

### 2. Google Driveにアップロード
1. Google Driveを開く
2. 「マイドライブ」に`lora_20250830_223432.tar.gz`をアップロード
3. フォルダ構成:
   ```
   My Drive/
   └── lora_20250830_223432.tar.gz
   ```

## Google Colab実行手順

### Step 1: Colabノートブック作成
1. https://colab.research.google.com/ にアクセス
2. 「新しいノートブック」作成
3. ランタイム → ランタイムのタイプを変更 → **A100 GPU**を選択

### Step 2: スクリプト実行
```python
# セル1: GPU確認
!nvidia-smi

# セル2: 変換スクリプトアップロード
from google.colab import files
uploaded = files.upload()  # cloud_gpu_conversion.pyをアップロード

# セル3: 実行
!python cloud_gpu_conversion.py
```

### Step 3: 処理時間の目安
- LoRAマージ: 15-20分
- GGUF変換: 10-15分
- 量子化: 10-15分
- **合計: 約40-60分**

## ダウンロードとローカル設定

### 1. ファイルダウンロード
Google Driveまたは直接ダウンロードから取得:
- `deepseek-finetuned-q4_k_m.gguf` (約18-20GB)
- `Modelfile` (1KB)

### 2. WSL2への転送
```bash
# Windowsのダウンロードフォルダから転送
cp /mnt/c/Users/[ユーザー名]/Downloads/deepseek-finetuned-q4_k_m.gguf ~/
cp /mnt/c/Users/[ユーザー名]/Downloads/Modelfile ~/
```

### 3. Dockerコンテナへコピー
```bash
docker cp ~/deepseek-finetuned-q4_k_m.gguf ai-ft-container:/workspace/
docker cp ~/Modelfile ai-ft-container:/workspace/
```

### 4. Ollamaに登録
```bash
docker exec ai-ft-container ollama create deepseek-finetuned -f /workspace/Modelfile
```

### 5. 動作確認
```bash
# Ollamaで直接テスト
docker exec ai-ft-container ollama run deepseek-finetuned "設計速度100km/hの最小曲線半径は？"
# 期待: 460m

# RAGシステムでテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度100km/hの最小曲線半径は？"}'
```

## RAG設定更新

`src/rag/config/rag_config.yaml`:
```yaml
llm:
  use_ollama_fallback: true
  ollama_model: deepseek-finetuned  # 変更
  ollama_host: http://localhost:11434
```

## トラブルシューティング

### メモリ不足エラー
- Colab Pro+にアップグレード
- またはKaggleのA100を使用

### CUDA out of memory
```python
# メモリクリアを追加
import gc
import torch
torch.cuda.empty_cache()
gc.collect()
```

### 変換エラー
```bash
# llama.cppの再インストール
!rm -rf llama.cpp
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && make clean && make LLAMA_CUDA=1
```

## コスト見積もり

| サービス | コスト | 備考 |
|---------|--------|------|
| Google Colab Pro+ | $49.99/月 | 月100時間のA100使用可能 |
| Kaggle | 無料 | 週30時間まで |
| Paperspace | 約$3/時間 | 1回の変換で約$3-5 |

## まとめ

1. **Google Colab Pro+**が最も簡単で確実
2. 処理時間は**約1時間**
3. 生成されるGGUFファイルは**約18-20GB**
4. 一度変換すれば永続的に使用可能

これで、ファインチューニング済みのLoRAモデルをOllamaで使用できるようになります。