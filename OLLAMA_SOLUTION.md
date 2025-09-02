# ファインチューニング済みモデルをOllamaで使用する解決策

## 現状の問題
1. **LoRAアダプタ**（`lora_20250830_223432`）は直接Ollamaで使用不可
2. **量子化形式の違い**：
   - 現在：BitsAndBytes形式（Python用）
   - 必要：GGUF形式（Ollama用）
3. **メモリ制限**：32Bモデルのマージには64GB+必要（現在48GB）

## ✅ 実行可能な解決策

### 解決策1: ベースモデルのGGUFを使用＋プロンプトエンジニアリング（即座に実行可能）

既にお持ちの`cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese.gguf`を使用して、ファインチューニングで学習した知識をプロンプトで補完：

```bash
# 1. GGUFファイルをDockerコンテナにコピー
docker cp /path/to/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese.gguf ai-ft-container:/workspace/

# 2. Dockerコンテナ内で作業
docker exec -it ai-ft-container bash

# 3. ファインチューニングの知識を含むModelfileを作成
cat > /workspace/DeepSeekFinetuned.Modelfile << 'EOF'
FROM /workspace/cyberagent-DeepSeek-R1-Distill-Qwen-32B-Japanese.gguf

# ファインチューニングで学習した道路設計基準を明示
SYSTEM """あなたは日本の道路設計の専門家です。
以下の正確な設計基準を厳守してください：

【最小曲線半径】
- 設計速度120km/h: 710m
- 設計速度100km/h: 460m
- 設計速度80km/h: 280m
- 設計速度60km/h: 150m
- 設計速度50km/h: 100m
- 設計速度40km/h: 60m
- 設計速度30km/h: 30m
- 設計速度20km/h: 15m

【縦断勾配の標準値】
- 設計速度120km/h: 2%
- 設計速度100km/h: 3%
- 設計速度80km/h: 4%
- 設計速度60km/h: 5%

【その他の基準】
- 横断勾配: 1.5〜2.0%（標準）
- 片勾配: 最大6〜10%（設計速度による）
- 緩和曲線: クロソイド曲線を使用

必ず上記の数値に基づいて正確に回答してください。"""

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
EOF

# 4. Ollamaにモデルを登録
ollama create deepseek-road-expert -f /workspace/DeepSeekFinetuned.Modelfile

# 5. テスト
ollama run deepseek-road-expert "設計速度100km/hの最小曲線半径は？"
# 期待される回答: 460m
```

### 解決策2: より小さいモデルでファインチューニング（推奨）

メモリ制約を回避するため、7Bや14Bモデルを使用：

```python
# 新しいファインチューニングスクリプト
# Qwen2.5-7B-Instructを使用（メモリ使用量: ~14GB）
base_model = "Qwen/Qwen2.5-7B-Instruct"

# 同じデータセットでファインチューニング
# その後GGUF形式に変換可能
```

### 解決策3: クラウドでマージ後ダウンロード

Google Colab Pro+やAWS等でマージ：

```python
# Colab用スクリプト（A100 40GB/80GB使用）
!pip install transformers peft accelerate

from transformers import AutoModelForCausalLM
from peft import PeftModel

# ベースモデルロード
base = AutoModelForCausalLM.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    device_map="auto"
)

# LoRAマージ
model = PeftModel.from_pretrained(base, "/path/to/lora")
merged = model.merge_and_unload()

# 保存してダウンロード
merged.save_pretrained("./merged")
```

## 📝 RAG設定の更新

どの方法を選んでも、最終的にRAG設定を更新：

```yaml
# src/rag/config/rag_config.yaml
llm:
  use_ollama_fallback: true
  ollama_model: deepseek-road-expert  # 作成したモデル名
  ollama_host: http://localhost:11434
```

## 🎯 推奨アクション

1. **即座に使用したい場合**：解決策1（プロンプトエンジニアリング）
2. **品質重視**：解決策2（小規模モデルで再ファインチューニング）
3. **オリジナルモデル重視**：解決策3（クラウドでマージ）

## テストコマンド

```bash
# RAGシステムでテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度100km/hの最小曲線半径は？", "top_k": 3}'

# 正しい回答が返ってくるか確認
# 期待: 460m（現在のllama3.2:3bは30mと誤答）
```