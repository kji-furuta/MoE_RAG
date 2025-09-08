# RAGシステム ハイブリッド検索・質問応答 使用ガイド

## 概要
MoE_RAGシステムは、ベクトル検索とキーワード検索を組み合わせたハイブリッド検索機能を提供し、土木・道路設計分野に特化した高精度な質問応答を実現します。

## システムアーキテクチャ

### 1. ハイブリッド検索の仕組み
```
クエリ → [ベクトル検索 (70%)] + [キーワード検索 (30%)] → 統合スコアリング → リランキング → 結果
```

#### ベクトル検索（意味的類似性）
- **埋め込みモデル**: `intfloat/multilingual-e5-large` (1024次元)
- **ベクトルDB**: Qdrant
- **重み**: 0.7（70%）
- **特徴**: 文脈や意味を理解した検索

#### キーワード検索（語彙的一致）
- **手法**: TF-IDF + BM25
- **重み**: 0.3（30%）
- **特徴**: 専門用語や数値の正確な一致

### 2. LLMによる回答生成
- **モデル**: GPT-NeoX-20B（LoRA適用済み）またはDeepSeek-32B
- **コンテキスト**: 検索結果上位5件をリランキング後に使用
- **最大トークン**: 4096

## 使用方法

### WebUI経由での使用

#### 1. アクセス
```bash
# ブラウザで開く
http://localhost:8050/rag
```

#### 2. 基本的な質問応答
```
1. RAGタブを選択
2. 「Query」フィールドに質問を入力
   例: "設計速度80km/hの道路の最小曲線半径は？"
3. 「Search」ボタンをクリック
4. 検索結果と生成された回答を確認
```

#### 3. 詳細検索オプション
- **Document Type Filter**: 文書タイプでフィルタリング
  - `standard`: 設計基準書
  - `guideline`: ガイドライン
  - `manual`: マニュアル
  
- **Top-K Results**: 取得する検索結果数（デフォルト: 10）
- **Search Type**: 検索方式の選択
  - `hybrid`: ハイブリッド検索（推奨）
  - `vector`: ベクトル検索のみ
  - `keyword`: キーワード検索のみ

### API経由での使用

#### 1. 基本的な質問応答API
```bash
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "道路の縦断勾配の最大値について教えてください",
       "top_k": 5,
       "filters": {
         "document_type": "standard"
       }
     }'
```

#### レスポンス例
```json
{
  "query": "道路の縦断勾配の最大値について教えてください",
  "answer": "道路の縦断勾配の最大値は、設計速度により異なります。\n- 設計速度120km/h: 最大3%\n- 設計速度100km/h: 最大4%\n- 設計速度80km/h: 最大5%\n- 設計速度60km/h: 最大7%\n特例値として、地形の状況等でやむを得ない場合は、これらの値に2%を加えた値まで許容されます。",
  "sources": [
    {
      "text": "縦断勾配は、設計速度に応じて...",
      "metadata": {
        "document_name": "道路構造令",
        "page": 45,
        "section": "第20条"
      },
      "score": 0.92
    }
  ],
  "metadata": {
    "search_type": "hybrid",
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "response_time": 1.2,
    "model_used": "gpt-neox-20b-finetuned"
  }
}
```

#### 2. ストリーミング検索API（長文回答用）
```bash
curl -X POST "http://localhost:8050/rag/stream-query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "道路設計における安全施設の設置基準を詳しく説明してください",
       "stream": true
     }'
```

#### 3. バッチ検索API（複数質問の一括処理）
```bash
curl -X POST "http://localhost:8050/rag/batch-query" \
     -H "Content-Type: application/json" \
     -d '{
       "queries": [
         "最小曲線半径の計算方法は？",
         "視距の確保について",
         "交差点の設計基準"
       ],
       "top_k": 3
     }'
```

### Python SDKでの使用

```python
from src.rag.core.query_engine import QueryEngine
from src.rag.config.config_loader import ConfigLoader

# 初期化
config = ConfigLoader.load_config()
query_engine = QueryEngine(config)

# 基本的な質問応答
result = query_engine.query(
    query="橋梁の設計荷重について説明してください",
    top_k=5,
    search_type="hybrid"
)

print(f"回答: {result['answer']}")
print(f"ソース: {result['sources'][0]['metadata']['document_name']}")

# メタデータフィルタリング付き検索
result = query_engine.query(
    query="トンネル換気設備の基準",
    filters={
        "document_type": "standard",
        "version": "最新版"
    }
)

# 専門用語のブースト検索
result = query_engine.query(
    query="設計速度と曲線半径の関係",
    boost_terms=["設計速度", "曲線半径", "R値"]
)
```

## 高度な機能

### 1. 数値処理と単位正規化
システムは自動的に数値と単位を認識・正規化します：
- `80キロメートル毎時` → `80km/h`
- `千五百メートル` → `1500m`
- `3パーセント` → `3%`

### 2. 文書バージョン管理
最新の設計基準を優先的に参照：
```python
filters = {
    "version": "最新版",
    "year": {"$gte": 2020}
}
```

### 3. セクション単位の検索
特定の章節に限定した検索：
```python
filters = {
    "chapter": "第3章",
    "section": "道路の構造"
}
```

### 4. 引用付き回答生成
回答に引用元を含める：
```json
{
  "include_citations": true,
  "citation_format": "detailed"
}
```

## パフォーマンス最適化

### 1. キャッシュ機能
- 頻繁な検索結果を自動キャッシュ（TTL: 3600秒）
- 最大1000件のクエリ結果を保持

### 2. バッチ処理
- 最大50件の質問を並列処理
- 4つのワーカーで同時実行

### 3. GPU最適化
- 埋め込み生成にCUDAを使用
- バッチサイズ: 32
- メモリ最適化モード有効

## トラブルシューティング

### 問題: 検索結果が不正確
**解決策**:
1. ハイブリッド検索の重みを調整
```yaml
# src/rag/config/rag_config.yaml
retrieval:
  hybrid_search:
    vector_weight: 0.6  # 意味検索を減らす
    keyword_weight: 0.4  # キーワード検索を増やす
```

2. リランキングモデルを変更
```yaml
retrieval:
  reranking:
    model: deepseek-32b-finetuned  # より高性能なモデルへ
```

### 問題: 回答生成が遅い
**解決策**:
1. モデルの量子化を有効化
2. キャッシュを活用
3. top_kを減らす（10→5）

### 問題: メモリ不足
**解決策**:
```yaml
llm:
  load_in_8bit: true  # 8bit量子化
  max_memory:
    0: 24GB  # GPU0のメモリ制限
```

## 使用例とベストプラクティス

### 1. 設計基準の確認
```
質問: "設計速度100km/hの道路で必要な視距は？"
期待される回答: 停止視距160m、追越視距500m（出典付き）
```

### 2. 計算式の取得
```
質問: "クロソイド曲線のパラメータAの計算式を教えて"
期待される回答: A = √(R × L) の公式と適用条件
```

### 3. 複合的な質問
```
質問: "山岳地帯での道路設計で考慮すべき要素を全て列挙"
期待される回答: 勾配、曲線半径、視距、排水、法面保護等の総合的な回答
```

## モニタリングとログ

### アクセスログ
```bash
tail -f /workspace/logs/rag_queries.log
```

### パフォーマンスメトリクス
```bash
curl http://localhost:8050/rag/metrics
```

### システム情報
```bash
curl http://localhost:8050/rag/system-info
```

## 更新履歴
- 2025-09-08: ハイブリッド検索ガイド作成
- 2025-09-07: GPT-NeoX-20B統合
- 2025-09-01: 初期バージョン