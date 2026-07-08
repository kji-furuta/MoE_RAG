# Webインターフェース統合

## 概要

RAG API（ポート8051）とメインAPI（ポート8050）を統合し、ユーザーが単一のエンドポイントでアクセスできるWebインターフェースを実装しました。

## 解決された問題

**問題**: RAG API（ポート8051）とメインAPI（ポート8050）の分離
**影響**: ユーザーが別々のエンドポイントにアクセスする必要
**原因**: 統合されたWebインターフェースの未実装

## 統合後のアーキテクチャ

### 統合前
```
📍 メインAPI（ポート8050）
  - ファインチューニング機能
  - モデル管理
  - 文字生成

📍 RAG API（ポート8051）
  - 文書検索・質問応答
  - 文書アップロード
  - システム統計
```

### 統合後
```
📍 統合API（ポート8050のみ）
  ├── メイン機能
  │   ├── /（ホーム）
  │   ├── /finetune（ファインチューニング）
  │   ├── /models（モデル管理）
  │   ├── /readme（README）
  │   └── /rag（NEW: RAGシステム）
  │
  ├── API エンドポイント
  │   ├── /api/* （既存のAPI）
  │   └── /rag/* （NEW: RAG API）
  │
  └── RAG統合機能
      ├── /rag/health（ヘルスチェック）
      ├── /rag/query（文書検索・QA）
      ├── /rag/search（簡易検索）
      ├── /rag/documents（文書一覧）
      ├── /rag/upload-document（文書アップロード）
      ├── /rag/statistics（統計情報）
      └── /rag/stream-query（ストリーミング検索）
```

## 技術的実装

### 1. Service Integration アプローチ

**選択理由**: 
- 最もシンプルな統合方法
- プロキシ設定が不要
- 統一されたエラーハンドリング
- 単一プロセスでの実行

### 2. 統合されたコンポーネント

#### RAGApplication クラス
```python
class RAGApplication:
    """RAGアプリケーション統合クラス"""
    
    def __init__(self):
        self.query_engine: Optional[RoadDesignQueryEngine] = None
        self.metadata_manager: Optional[MetadataManager] = None
        self.is_initialized = False
        self.initialization_error = None
        
    async def initialize(self):
        """非同期でシステムを初期化"""
        # RAGシステムコンポーネントの初期化
        
    def check_initialized(self):
        """初期化チェック"""
        # 初期化状態の確認
```

#### 統合されたPydanticモデル
```python
# RAG専用のリクエスト/レスポンスモデル
class QueryRequest(BaseModel): ...
class QueryResponse(BaseModel): ...
class BatchQueryRequest(BaseModel): ...
class SystemInfoResponse(BaseModel): ...
class DocumentUploadResponse(BaseModel): ...
```

#### 統合されたAPI エンドポイント（9個）
1. `GET /rag/health` - ヘルスチェック
2. `GET /rag/system-info` - システム情報
3. `POST /rag/query` - メイン検索・QA機能
4. `POST /rag/batch-query` - バッチ検索
5. `GET /rag/search` - 簡易検索API
6. `GET /rag/documents` - 文書一覧
7. `GET /rag/statistics` - システム統計
8. `POST /rag/upload-document` - 文書アップロード
9. `POST /rag/stream-query` - ストリーミング検索

### 3. 起動フロー

```python
@app.on_event("startup")
async def startup_event():
    """統合アプリ起動時の処理"""
    logger.info("Starting AI Fine-tuning Toolkit with RAG integration...")
    if RAG_AVAILABLE:
        await rag_app.initialize()  # RAGシステム初期化
    else:
        logger.warning("RAG system will not be available")
```

## 利用方法

### 1. サーバー起動
```bash
# 統合されたWebインターフェースを起動
python3 app/main_unified.py

# または
uvicorn app.main_unified:app --host 0.0.0.0 --port 8050
```

### 2. アクセス方法

**Webインターフェース**:
- ホーム: http://localhost:8050/
- ファインチューニング: http://localhost:8050/finetune
- モデル管理: http://localhost:8050/models
- **RAGシステム**: http://localhost:8050/rag （NEW）

**API エンドポイント**:
```bash
# RAGヘルスチェック
curl http://localhost:8050/rag/health

# RAG文書検索
curl -X POST http://localhost:8050/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "道路設計について教えて", "top_k": 5}'

# RAG簡易検索
curl "http://localhost:8050/rag/search?q=交通安全&top_k=3"

# RAG文書一覧
curl http://localhost:8050/rag/documents

# RAGシステム統計
curl http://localhost:8050/rag/statistics
```

### 3. 文書アップロード
```bash
# PDF文書をRAGシステムにアップロード
curl -X POST http://localhost:8050/rag/upload-document \
  -F "file=@document.pdf" \
  -F "title=道路設計指針" \
  -F "category=技術基準"
```

## 統合テスト結果

```
🏁 テスト結果サマリー
============================================================
✅ PASS 統合構造テスト
✅ PASS API互換性テスト  
✅ PASS 起動フローテスト

📊 成功率: 3/3 (100.0%)

🎉 すべてのテストが成功！
💡 Webインターフェース統合が完了しました
🚀 single port (8050) でRAGとメインAPIの両方にアクセス可能
```

### テスト内容
1. **統合構造テスト**: RAGコンポーネントの統合確認
2. **API互換性テスト**: 全エンドポイントの可用性確認
3. **起動フローテスト**: 統合アプリケーションの起動プロセス確認

## メリット

### ユーザー体験の向上
- ✅ **単一アクセスポイント**: ポート8050のみでアクセス
- ✅ **統一されたUI**: 一つのWebインターフェース内でRAGとファインチューニング機能
- ✅ **シンプルな操作**: 複数のポートを管理する必要なし

### 運用面の改善
- ✅ **単一プロセス**: 1つのアプリケーションとして実行
- ✅ **統一ログ**: ログとエラーハンドリングの一元化
- ✅ **簡単デプロイ**: Docker設定の簡素化

### 開発面の利益
- ✅ **保守性向上**: 単一コードベースでの管理
- ✅ **機能追加容易**: 新機能の統合が簡単
- ✅ **テスト効率化**: 統合テストが一箇所で実行可能

## 注意事項

### RAG システムの依存関係
- RAGシステムが利用できない場合でも、メイン機能は正常動作
- `RAG_AVAILABLE` フラグで機能の有効/無効を制御
- 初期化エラーは適切にハンドリングされログ出力

### エラーハンドリング
```python
rag_app.check_initialized()  # RAG初期化チェック
# 初期化されていない場合はHTTPException(503)を発生
```

### バックグラウンドタスク
- 文書アップロード処理は非同期バックグラウンドタスクで実行
- 一時ファイルは処理後に自動削除

## 今後の拡張

### 可能な改善点
1. **WebSocket統合**: リアルタイム機能の拡張
2. **認証統合**: 統一認証システムの実装
3. **キャッシュ共有**: RAGとメイン機能間のキャッシュ共有
4. **監視統合**: 統合されたメトリクスとモニタリング

### API バージョニング
将来的にAPIバージョニングを実装する場合：
```
/v1/rag/*  # バージョン1
/v2/rag/*  # バージョン2
```

## まとめ

Webインターフェース統合により、RAG APIとメインAPIが単一のエンドポイント（ポート8050）に統合され、ユーザー体験と運用効率が大幅に向上しました。

- **統合前**: 2つの分離されたAPI（8050 & 8051）
- **統合後**: 1つの統合API（8050のみ）
- **結果**: 100%の統合テスト成功率

この統合により、AI Fine-tuning Toolkitは真の統合プラットフォームとして機能するようになりました。