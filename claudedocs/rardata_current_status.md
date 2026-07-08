# RARdata.json 学習済みモデル使用ガイド

**最終更新**: 2025年12月8日 23:30
**ステータス**: ✅ 学習完了・使用可能

---

## 📁 学習済みモデル情報

### モデルの場所
```
outputs/continual_rardata_training_20251208_141953/checkpoint-final/
├── adapter_config.json    # LoRA設定
├── adapter_model.safetensors  # LoRAウェイト
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md
```

### モデル仕様
- **ベースモデル**: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
- **学習方法**: LoRA (Low-Rank Adaptation)
- **学習データ**: RARdata.json (RAR形式)
- **学習可能パラメータ**: 33,554,432 (全体の0.1023%)
- **量子化**: 4-bit (BitsAndBytes nf4)
- **学習エポック**: 3
- **最終Loss**: 2.7630

---

## 🚀 使用方法

### A. REST API経由（推奨）

```bash
# RAGクエリ with 学習済みモデル
curl -X POST "http://localhost:8050/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "設計速度80km/hの道路の最小曲線半径は？",
    "top_k": 5,
    "use_reasoning_model": true
  }'
```

**レスポンス例**:
```json
{
  "answer": "設計速度80km/hの道路の最小曲線半径は280mです。\n\n【引用】\n1. 道路構造令の解説と運用（令和3年）",
  "sources": [...],
  "reasoning_steps": [...],
  "citations": [...]
}
```

---

### B. Web UI経由

#### 1. RAGシステムにアクセス
```
http://localhost:8050/rag
```

#### 2. 検索実行
1. 検索ボックスにクエリを入力:
   ```
   設計速度80km/hの道路の最小曲線半径は？
   ```

2. **Advanced Options** を展開

3. **Use Reasoning Model** を **ON** にする

4. **Search** ボタンをクリック

#### 3. 結果確認
- Chain-of-Thought推論プロセスが表示されます
- 引用元がアコーディオン形式で展開できます
- 回答の根拠となる文書が明示されます

---

### C. Python SDKでの使用

```python
import requests

def query_rag_with_trained_model(query: str):
    """学習済みモデルを使用したRAG検索"""

    response = requests.post(
        "http://localhost:8050/rag/query",
        json={
            "query": query,
            "top_k": 5,
            "use_reasoning_model": True,
            "citation_style": "academic"
        }
    )

    result = response.json()

    print(f"回答: {result['answer']}\n")
    print(f"推論ステップ数: {len(result.get('reasoning_steps', []))}")
    print(f"引用文献数: {len(result['citations'])}")

    return result

# 使用例
result = query_rag_with_trained_model(
    "設計速度80km/hの道路の最小曲線半径は？"
)
```

---

## ❌ よくある誤解

### 誤解1: RARdata.jsonを再度アップロードして学習が必要
**正しい理解**: RARdata.jsonは既に学習済みです。再学習は不要。

### 誤解2: LoRAをGGUFに変換する必要がある
**正しい理解**:
- LoRAアダプターは、ベースモデルと組み合わせて使用します
- GGUF変換が必要な場合は、先にLoRAをマージしてから変換
- 現在のシステムでは、LoRA形式のまま使用可能

### 誤解3: UIでモデルを手動選択する必要がある
**正しい理解**:
- `use_reasoning_model: true` を指定すれば自動的に最新の学習済みモデルが使用されます
- モデルパスを明示的に指定することも可能

---

## 🔄 追加学習が必要な場合

新しい学習データがある場合のみ、以下の手順で追加学習を実施:

### UIでの追加学習

1. **http://localhost:8050/continual** にアクセス

2. **新しいタスクを作成**:
   ```
   Task Name: rardata_task2
   Base Model: outputs/continual_rardata_training_20251208_141953/checkpoint-final
   Training Data: (新しいJSONファイルをアップロード)
   EWC Lambda: 5000
   Epochs: 3
   ```

3. **学習開始** → 完了を待つ

4. **新しいモデルをRAGで使用**

---

## 📊 性能評価

### 推奨評価項目

1. **引用精度**: 正しい文書を引用しているか
2. **推論品質**: Chain-of-Thoughtの論理性
3. **回答正確性**: 専門知識の正確性
4. **応答速度**: レスポンスタイム

### 評価スクリプト例

```python
import json
import requests
from typing import List, Dict

def evaluate_model_performance(test_queries: List[str]):
    """モデル性能を評価"""

    results = []

    for query in test_queries:
        response = requests.post(
            "http://localhost:8050/rag/query",
            json={
                "query": query,
                "top_k": 5,
                "use_reasoning_model": True
            }
        )

        result = response.json()

        # 評価メトリクス
        metrics = {
            "query": query,
            "answer_length": len(result.get("answer", "")),
            "num_citations": len(result.get("citations", [])),
            "num_reasoning_steps": len(result.get("reasoning_steps", [])),
            "has_final_answer": "final_answer" in result.get("answer", "").lower()
        }

        results.append(metrics)

    return results

# テストクエリ
test_queries = [
    "設計速度80km/hの道路の最小曲線半径は？",
    "第3種道路の定義を教えてください",
    "道路構造令における地方部の分類基準は？"
]

# 評価実行
evaluation_results = evaluate_model_performance(test_queries)

# 結果をJSON保存
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
```

---

## 🛠️ トラブルシューティング

### Q1: "Model not found" エラー
**解決策**: モデルパスを確認
```bash
docker exec ai-ft-container ls -lh /workspace/outputs/continual_rardata_training_20251208_141953/checkpoint-final/
```

### Q2: メモリ不足エラー
**解決策**:
- Batch sizeを1に維持
- GPUメモリをクリア:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

### Q3: 推論が遅い
**解決策**:
- vLLM統合を確認（`src/inference/vllm_integration.py`）
- PagedAttentionが有効か確認

### Q4: 引用が表示されない
**解決策**:
- `citation_style` パラメータを確認
- Citation Engineのログを確認:
  ```bash
  docker logs ai-ft-container | grep "citation"
  ```

---

## 📚 関連ドキュメント

- [RAG UI使用ガイド](./rag_ui_usage_guide.md) - Web UIでの詳細な使用方法
- [RAR学習ガイド](./rar_training_guide.md) - RAR形式データの学習方法
- [クイックスタート](./rardata_quickstart.md) - 3ステップでの学習手順

---

## 🎯 次のアクション

1. ✅ **モデルを使ってみる**: REST APIまたはWeb UIで実際にクエリを実行
2. 📊 **性能を評価**: テストクエリで引用精度と推論品質を測定
3. 🔄 **必要に応じて追加学習**: 新しいデータで継続学習
4. 📈 **本番環境への展開**: 評価結果が良好なら本番利用を検討

---

**作成日**: 2025年12月8日 23:30
**対象モデル**: outputs/continual_rardata_training_20251208_141953/checkpoint-final
**次回更新**: 性能評価結果が得られた後
