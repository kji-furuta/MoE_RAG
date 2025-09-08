# App Directory Cleanup - 2025-09-08

## 実施した整理作業

### 削除したファイル
1. **app/__pycache__/** - Pythonキャッシュディレクトリ（不要）
2. **app/dependencies.py** - 未使用のファイル（どこからも参照なし）
3. **app/static/test_moe_status.html** - テスト用HTML（本番不要）
4. **app/static/logo_teikoku.jpg** - 重複ロゴファイル（PNGを保持）

### 保持したファイル（必要性確認済み）
- **main_unified.py** - メインアプリケーション（要リファクタリング）
- **moe_rag_endpoints.py** - MoE-RAGエンドポイント（main_unified.pyから参照）
- **moe_training_endpoints.py** - MoEトレーニングエンドポイント（複数箇所から参照）
- **ollama_integration.py** - Ollama統合（5箇所から参照）
- **memory_optimized_loader.py** - メモリ最適化ローダー
- **model_utils.py** - モデルユーティリティ
- **routers/** - ルーターモジュール（全て使用中）
- **monitoring/** - モニタリング機能（main_unified.pyから参照）
- **continual_learning/** - 継続学習UI
- **static/logo_teikoku.png** - ロゴファイル（PNG形式を保持）
- **static/moe_rag_ui.html** - MoE-RAG UI
- **static/moe_training.html** - MoEトレーニングUI

## 結果
- 不要ファイル4個削除
- ディレクトリ構造の整理完了
- システムの依存関係維持確認済み

## 注意事項
- main_unified.py（248KB）は大きすぎるため、将来的にリファクタリング推奨
- __pycache__は自動生成されるため、.gitignoreに追加推奨