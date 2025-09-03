# GGUF モデルの RAG システム統合ガイド

## 概要
RAGシステムでGGUF形式のモデル（例：DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf）を使用することは可能です。ただし、事前にOllamaへの登録が必要です。

## 手順

### 1. GGUFモデルのOllama登録

既存のスクリプトを使用してGGUFモデルをOllamaに登録します：

```bash
# 基本的な使用方法
python scripts/apply_lora_to_gguf.py \
    --base-model-url "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf" \
    --base-model-name "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf" \
    --output-name "deepseek-32b-rag"
```

または、すでにダウンロード済みのGGUFファイルがある場合：

```bash
# 1. Modelfileを作成
cat > deepseek-32b.modelfile << EOF
FROM /path/to/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf

SYSTEM "あなたは日本の土木設計と道路設計の専門家です。技術的な質問に対して正確で詳細な回答を提供します。"

PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
EOF

# 2. Ollamaに登録
ollama create deepseek-32b-rag -f deepseek-32b.modelfile

# 3. 動作確認
ollama run deepseek-32b-rag "テスト"
```

### 2. RAG設定ファイルの更新

`src/rag/config/rag_config.yaml`を編集：

```yaml
llm:
  # 利用可能なモデルリストに追加
  available_models:
  - deepseek-32b-rag  # 新しく追加
  - deepseek-32b-finetuned
  - llama3.2:3b
  - deepseek-r1:32b
  
  # Ollama設定
  ollama:
    base_url: http://localhost:11434
    model: deepseek-32b-rag  # 使用するモデルを指定
    temperature: 0.6
    top_p: 0.9
  
  # モデル名を更新
  model_name: ollama:deepseek-32b-rag:latest
```

### 3. サーバー再起動

設定を反映させるためにサーバーを再起動：

```bash
# Dockerコンテナの場合
docker restart ai-ft-container

# または手動で再起動
docker exec ai-ft-container pkill -f "uvicorn"
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 4. UIからの使用

1. ブラウザで `http://localhost:8050/rag` にアクセス
2. モデル選択ドロップダウンから "deepseek-32b-rag" を選択
3. クエリを入力して検索実行

## 動作確認

### APIでのテスト

```bash
# RAGクエリのテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "設計速度80km/hの道路の最小曲線半径は？",
       "model": "deepseek-32b-rag",
       "top_k": 5
     }'
```

### Pythonでのテスト

```python
import requests

response = requests.post(
    "http://localhost:8050/rag/query",
    json={
        "query": "道路の設計速度について教えてください",
        "model": "deepseek-32b-rag",
        "top_k": 5
    }
)

print(response.json())
```

## トラブルシューティング

### モデルが選択できない場合

1. Ollamaでモデルが登録されているか確認：
```bash
ollama list
```

2. Ollamaサービスが起動しているか確認：
```bash
curl http://localhost:11434/api/tags
```

3. RAG設定ファイルに正しくモデル名が記載されているか確認

### メモリ不足エラー

GGUFモデルは量子化されていますが、32Bモデルの場合でも相当のメモリが必要です：

- Q4_K_M量子化: 約20GB RAM
- GPUを使用する場合: 約16GB VRAM

メモリが不足する場合は、より小さいモデルを使用するか、量子化レベルを上げてください：
- Q3_K_S: さらに小さいサイズ
- Q2_K: 最小サイズ（精度は低下）

## 利点

1. **高速推論**: GGUFフォーマットは最適化された推論を提供
2. **低メモリ使用**: 量子化により大幅なメモリ削減
3. **Ollama統合**: シンプルなAPI経由でアクセス可能
4. **カスタマイズ可能**: System promptやパラメータを調整可能

## 注意事項

- GGUFモデルは事前にOllamaに登録する必要があります
- 大きなモデルの場合、初回ロード時に時間がかかることがあります
- RAGシステムは`ollama:`プレフィックスでOllamaモデルを識別します