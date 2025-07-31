# Ollama統合ガイド

## 概要

Ollamaを使用することで、大きなモデル（32B、70B）でも20GB GPUで動作可能になります。**ファインチューニング済みモデルもOllamaで使用可能**です。

## ファインチューニング済みモデルのOllama変換

### 1. **自動変換**

ファインチューニング済みモデルをOllama形式に変換するには：

```bash
# 変換スクリプトを実行
python convert_finetuned_to_ollama.py
```

### 2. **API経由での変換**

```bash
curl -X POST "http://localhost:8000/api/convert-to-ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/workspace/outputs/フルファインチューニング_20250723_041920",
    "model_name": "road-engineering-expert"
  }'
```

### 3. **変換後の使用方法**

```bash
# 変換されたモデルを使用
ollama run road-engineering-expert "縦断曲線とは何ですか？"
```

## Ollamaの利点

### 1. **効率的なメモリ管理**
- **GGML/GGUF形式**：最適化されたモデル形式
- **動的メモリ割り当て**：必要な部分のみをメモリに読み込み
- **自動メモリ最適化**：使用されていない部分を自動解放

### 2. **ファインチューニング済みモデルの保持**
- ✅ **学習結果が保持される**
- ✅ **専門分野特化型AIとして動作**
- ✅ **カスタム学習データの効果が反映**

## セットアップ手順

### 1. Ollamaのインストール

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# https://ollama.ai/download からダウンロード
```

### 2. Ollamaの起動

```bash
ollama serve
```

### 3. ファインチューニング済みモデルの変換

```bash
# 変換スクリプトを実行
python convert_finetuned_to_ollama.py
```

### 4. Python依存関係のインストール

```bash
pip install llama-cpp-python requests
```

## 使用方法

### 1. **自動統合**

大きなモデル（32B、70B）を使用する場合、システムは自動的にOllamaを使用します：

```python
# 32Bモデルを使用
response = requests.post("http://localhost:8000/api/generate", json={
    "model_path": "/workspace/outputs/フルファインチューニング_20250723_041920",
    "prompt": "縦断曲線とは何のために設置しますか？",
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9
})
```

### 2. **手動でのOllama使用**

```python
from ollama_integration import OllamaIntegration

# Ollamaクライアントの初期化
ollama = OllamaIntegration()

# ファインチューニング済みモデルでテキスト生成
result = ollama.generate_text(
    model_name="road-engineering-expert",
    prompt="縦断曲線とは何のために設置しますか？",
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

print(result["generated_text"])
```

## メモリ要件比較

| モデルサイズ | Transformers | Ollama |
|-------------|-------------|--------|
| 7B         | 16GB        | 8GB    |
| 14B        | 28GB        | 12GB   |
| 32B        | 64GB        | 20GB   |
| 70B        | 140GB       | 40GB   |

## ファインチューニング済みモデルの変換プロセス

### 1. **GGUF形式への変換**
```bash
# ファインチューニング済みモデルをGGUF形式に変換
python -m llama_cpp.convert /workspace/outputs/フルファインチューニング_20250723_041920 \
  --outfile road-engineering-expert.gguf \
  --outtype q4_k_m
```

### 2. **Ollamaモデルの作成**
```bash
# Modelfileの作成
cat > road-engineering-expert.Modelfile << EOF
FROM road-engineering-expert.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "質問:"
PARAMETER stop "回答:"

TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """あなたは道路工学の専門家です。質問に対して正確で分かりやすい回答を提供してください。"""
EOF

# Ollamaモデルの作成
ollama create road-engineering-expert road-engineering-expert.Modelfile
```

### 3. **使用**
```bash
# ファインチューニング済みモデルを使用
ollama run road-engineering-expert "縦断曲線とは何ですか？"
```

## トラブルシューティング

### 1. 変換エラー

```bash
# 依存関係の確認
pip install llama-cpp-python

# Ollamaの確認
ollama --version

# 変換スクリプトの実行
python convert_finetuned_to_ollama.py
```

### 2. メモリ不足

```bash
# より軽量な量子化を使用
python -m llama_cpp.convert model_path \
  --outfile model.gguf \
  --outtype q2_k  # 2bit量子化（最小メモリ）
```

### 3. モデルが見つからない

```bash
# 利用可能なモデルを確認
ollama list

# モデルを再作成
ollama create road-engineering-expert road-engineering-expert.Modelfile
```

## 注意事項

1. **変換時間**：大きなモデルの変換には時間がかかります（1-2時間）
2. **ディスク容量**：GGUFファイルは数GB〜数十GB必要です
3. **メモリ要件**：変換時には十分なメモリが必要です
4. **学習結果の保持**：ファインチューニングした学習結果は保持されます

## サポート

問題が発生した場合は、以下を確認してください：

1. 変換ログ：`python convert_finetuned_to_ollama.py`
2. Ollamaログ：`ollama logs`
3. システムリソース：`nvidia-smi`、`htop`
4. ディスク容量：`df -h` 