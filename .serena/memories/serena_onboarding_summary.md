# Serena オンボーディングサマリー

## プロジェクト識別情報
- **プロジェクト名**: AI_FT_3 (AI Fine-tuning Toolkit)
- **プロジェクトパス**: `/home/kjifu/AI_FT_7`
- **言語**: Python
- **フレームワーク**: FastAPI, PyTorch, Transformers

## 主要機能概要
1. **ファインチューニングシステム**: 日本語LLMの訓練・カスタマイズ
2. **RAGシステム**: 土木道路設計に特化した検索拡張生成
3. **統合Webインターフェース**: ポート8050で全機能アクセス
4. **モデル管理**: 動的量子化、CPUオフロード、マルチGPU対応

## クリティカルファイル
- **メインサーバー**: `app/main_unified.py`
- **モデルローダー**: `app/memory_optimized_loader.py`
- **RAGエンジン**: `src/rag/core/query_engine.py`
- **LoRA訓練**: `src/training/lora_finetuning.py`

## 開発ワークフロー
1. Docker環境での開発が必須
2. ポート8050で統合APIにアクセス
3. GPU/メモリ監視を常時実施
4. テストスクリプトで検証

## メモリーファイル一覧
- `project_overview_detailed.md` - プロジェクト詳細概要
- `architecture_and_patterns.md` - アーキテクチャパターン
- `command_reference_complete.md` - コマンドリファレンス（目次付き）
- `api_endpoints_reference.md` - APIエンドポイント詳細 [NEW]
- `critical_files_and_symbols.md` - 重要ファイルとシンボル参照 [NEW]
- `debugging_and_monitoring.md` - デバッグと監視ガイド [NEW]
- `model_specifications.md` - モデル仕様と設定詳細 [NEW]
- `development_guidelines.md` - 開発ガイドライン
- `troubleshooting_guide.md` - トラブルシューティング
- `environment_setup_guide.md` - 環境構築ガイド
- `suggested_commands.md` - 推奨コマンド集

## 追加されたオンボーディング情報
1. **APIエンドポイントリファレンス**: 全APIの詳細仕様
2. **重要ファイルとシンボル参照**: コードナビゲーション用
3. **デバッグと監視ガイド**: 問題解決とパフォーマンス監視
4. **モデル仕様書**: サポートモデルと設定詳細

## 次のステップ
- プロジェクトのコード変更時は、シンボルベースの編集ツールを優先使用
- 大規模な変更前にメモリーファイルを参照
- デバッグ時は`debugging_and_monitoring.md`を活用
- 新機能追加時は既存パターンに従う

## 注意事項
- Language Serverエラーが発生する場合があるが、メモリーファイルで代替可能
- Docker環境外での動作は保証されない
- GPU環境が必須（最小8GB VRAM）
- 土木設計関連のRAG機能は専門知識が必要