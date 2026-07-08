# MoE_RAG プロジェクト概要 (2024年版)

## プロジェクト情報
- **名前**: AI Fine-tuning Toolkit (AI_FT_7) / MoE_RAG
- **目的**: 土木工学・道路設計に特化した日本語LLMファインチューニングプラットフォーム（統合RAGシステム付き）
- **リポジトリ**: https://github.com/kji-furuta/MoE_RAG.git
- **開発環境**: WSL2 + Docker (Linux環境)

## 主要機能
1. **LLMファインチューニング**
   - LoRAトレーニング（効率的パラメータチューニング）
   - フルファインチューニング
   - DoRA（重み分解LoRA）
   - EWC（弾性的重み統合）による継続学習
   - マルチGPU分散学習サポート

2. **RAGシステム**
   - Qdrantベースのベクトルストア
   - ハイブリッド検索（ベクトル+キーワード）
   - PDF/OCR対応の文書処理
   - 数値処理・設計基準検証

3. **推論システム**
   - vLLM統合（高速推論）
   - AWQ量子化（メモリ75%削減）
   - Ollama統合（Llama 3.2 3B対応）

## アーキテクチャ
- **統合Webインターフェース**: FastAPI（ポート8050）
- **メインエントリポイント**: `app/main_unified.py`
- **フロントエンド**: Bootstrap + HTML templates
- **バックエンド**: Python 3.12.3 + PyTorch
- **データベース**: Qdrant（ベクトルDB）
- **コンテナ**: Docker 28.3.2

## プロジェクト構造
```
/home/kjifu/MoE_RAG/
├── app/            # Webアプリケーション
├── src/            # コアロジック
│   ├── training/   # 学習システム
│   ├── rag/        # RAGシステム
│   ├── inference/  # 推論システム
│   └── moe/        # MoE実装
├── docker/         # Docker設定
├── scripts/        # ユーティリティスクリプト
├── data/           # データディレクトリ
├── models/         # モデル保存
└── outputs/        # 出力ファイル
```