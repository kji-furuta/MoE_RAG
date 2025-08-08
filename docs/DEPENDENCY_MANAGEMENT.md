# RAG依存関係管理システム - Phase 1 実装完了

## 📋 概要

Phase 1として、RAGシステムの依存関係管理機能を実装しました。この機能により、RAGシステムが必要とする全ての依存関係を自動的にチェック、管理、インストールすることができます。

## 🎯 実装した機能

### 1. 依存関係マネージャー (`dependency_manager.py`)
- **階層的な依存関係管理**: Core、Infrastructure、Optional の3レベル
- **自動チェック機能**: インストール済みパッケージとバージョンの確認
- **代替パッケージ対応**: プライマリが使えない場合の代替手段
- **キャッシュ機能**: チェック結果の高速化
- **レポート生成**: Text、JSON、Markdown形式でのレポート出力

### 2. CLIツール (`check_rag_dependencies.py`)
```bash
# 依存関係をチェック
python scripts/check_rag_dependencies.py --check

# 不足している依存関係をインストール
python scripts/check_rag_dependencies.py --install

# Markdown形式でレポート生成
python scripts/check_rag_dependencies.py --report markdown > dependencies.md

# ドライラン（実際にインストールせずに確認）
python scripts/check_rag_dependencies.py --install --dry-run
```

### 3. Web API統合 (`main_unified_improved.py`)
新しいエンドポイント：
- `GET /api/dependencies/check` - 依存関係のチェック
- `GET /api/dependencies/report` - レポートの取得
- `POST /api/dependencies/install` - 依存関係のインストール
- `GET /health` - 改善されたヘルスチェック（依存関係情報付き）

### 4. テストスイート
- **単体テスト** (`test_dependency_manager.py`) - pytest対応
- **統合テスト** (`test_dependency_integration.py`) - 実環境での動作確認

## 📊 依存関係の分類

### Core Dependencies (必須)
- `transformers>=4.30.0` - Hugging Face Transformers
- `torch>=2.0.0` - PyTorch
- `sentence_transformers>=2.3.1` - 文埋め込みモデル
- `pydantic>=2.5.0` - データバリデーション

### Infrastructure Dependencies (基盤)
- `qdrant_client>=1.7.3` - ベクトルDB（代替: chromadb, faiss）
- `PyMuPDF>=1.23.16` - PDF処理（代替: pdfplumber）
- `pandas>=2.1.4` - データ分析
- `numpy>=1.24.4` - 数値計算
- `loguru>=0.7.2` - ログ管理

### Optional Dependencies (オプション)
- `easyocr>=1.7.1` - OCR処理
- `spacy>=3.7.2` - 自然言語処理
- `streamlit>=1.29.0` - Web UI
- `plotly>=5.18.0` - 可視化

## 🚀 使用方法

### 1. 基本的な使用

```python
from src.rag.dependencies.dependency_manager import RAGDependencyManager

# マネージャーの初期化
manager = RAGDependencyManager()

# 依存関係をチェック
result = manager.check_all_dependencies()

# レポートを表示
print(manager.get_dependency_report())

# システムが実行可能か確認
if result.can_run:
    print("✅ RAG system can run")
else:
    print("❌ Missing critical dependencies")
```

### 2. インストールの実行

```python
# 不足している依存関係をインストール
install_results = manager.install_missing_dependencies()

# 特定のレベルのみインストール
from src.rag.dependencies.dependency_manager import DependencyLevel
install_results = manager.install_missing_dependencies(
    level=DependencyLevel.CORE
)
```

### 3. Web APIからの使用

```bash
# 依存関係をチェック
curl http://localhost:8050/api/dependencies/check

# Markdownレポートを取得
curl "http://localhost:8050/api/dependencies/report?format=markdown"

# コア依存関係をインストール
curl -X POST "http://localhost:8050/api/dependencies/install?level=core"
```

## 🧪 テストの実行

### 単体テスト
```bash
# pytestを使用
pytest tests/test_dependency_manager.py -v

# または直接実行
python tests/test_dependency_manager.py
```

### 統合テスト
```bash
python scripts/test_dependency_integration.py
```

## 📈 パフォーマンス

- **初回チェック**: 約 1-2 秒
- **キャッシュ使用時**: 約 0.01 秒（100倍高速）
- **キャッシュ有効期限**: 1時間

## 🔧 トラブルシューティング

### 問題: ImportError が発生する
```bash
# 依存関係を確認
python scripts/check_rag_dependencies.py --check

# 不足している依存関係をインストール
python scripts/check_rag_dependencies.py --install
```

### 問題: Qdrantに接続できない
```bash
# 環境変数を設定
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# Docker でQdrantを起動
docker run -p 6333:6333 qdrant/qdrant
```

### 問題: キャッシュをクリアしたい
```python
manager = RAGDependencyManager()
manager._clear_cache()
```

## 📝 設定

環境変数で動作をカスタマイズ：

```bash
# Qdrantの接続設定
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

# デバッグモード
export DEBUG=1

# キャッシュディレクトリ
export RAG_CACHE_DIR=/custom/cache/path
```

## 🎯 次のステップ (Phase 2)

Phase 2では以下を実装予定：

1. **DIコンテナの実装**
   - サービスの依存性注入
   - ライフサイクル管理
   - シングルトン/プロトタイプスコープ

2. **ヘルスチェックシステム**
   - コンポーネント別の監視
   - 自動復旧機能
   - メトリクス収集

3. **設定管理の統一**
   - 環境別設定ファイル
   - 動的設定リロード
   - シークレット管理

## 📚 関連ドキュメント

- [依存関係マネージャーAPI](./api_reference.md)
- [トラブルシューティングガイド](./troubleshooting.md)
- [アーキテクチャ設計書](./architecture.md)

## ✅ チェックリスト

- [x] 依存関係マネージャーの実装
- [x] CLIツールの作成
- [x] Web API統合
- [x] テストスイートの作成
- [x] ドキュメントの作成
- [x] 統合テストの実行
- [ ] 本番環境でのテスト
- [ ] パフォーマンス最適化

## 📞 サポート

問題が発生した場合は、以下の手順で対処してください：

1. エラーログを確認
2. `--verbose` オプションで詳細情報を取得
3. Issueを作成（可能であればログを添付）

---

**Version**: 1.0.0  
**Last Updated**: 2024-12-XX  
**Author**: AI_FT Team
