# DeepSeekモデル機能復元 - 2025-09-08

## 復元内容

### 1. apply_lora_to_gguf_improved.py
- LoRAアダプターをGGUF形式に適用するスクリプト
- main_unified.pyから呼び出される重要なスクリプト
- 一時ディレクトリを使用してllama.cppを実行
- UIからの実行に最適化

### 2. init_deepseek_model.sh
**新規作成した機能:**
- DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.ggufの自動ダウンロード
- モデルサイズ（約20GB）の確認機能
- 不完全なダウンロードの検出と再ダウンロード
- Ollama用Modelfileの自動生成
- Ollamaへのモデル登録（deepseek-32b-japanese）

### 3. start_web_interface.sh更新
- DeepSeekモデル初期化を自動実行するように更新
- Ollamaモデル初期化後に実行

## モデル情報

**DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf**
- URL: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/
- サイズ: 約20GB
- 形式: GGUF (4ビット量子化)
- 用途: 高性能な日本語処理、RAGシステム

## 使用方法

### 初期化時の自動実行
```bash
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
# DeepSeekモデルが自動的にダウンロード・設定される
```

### 手動でのモデル初期化
```bash
docker exec ai-ft-container bash /workspace/scripts/init_deepseek_model.sh
```

### LoRA適用時の使用
```python
python /workspace/scripts/apply_lora_to_gguf_improved.py \
    --base-model-url https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \
    --base-model-name DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \
    --output-name deepseek-32b-finetuned \
    --lora-adapter /workspace/outputs/lora_xxxxx
```

### Ollamaでの使用
```bash
ollama run deepseek-32b-japanese
```

## ファイル配置
- モデル: `/workspace/models/gguf/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf`
- Modelfile: `/workspace/ollama_models/deepseek-32b-japanese.Modelfile`
- スクリプト: `/workspace/scripts/apply_lora_to_gguf_improved.py`

これで、DeepSeekモデルの初期化とLoRA適用機能が完全に復元されました。