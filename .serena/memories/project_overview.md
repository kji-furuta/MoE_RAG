# MoE-RAG統合システム プロジェクト概要

## プロジェクト名
**MoE_RAG** - Mixture of Experts + Retrieval-Augmented Generation 統合システム

## プロジェクトの目的
土木工学・道路設計に特化した日本語LLMファインチューニングとRAGシステムの統合プラットフォーム。
8つの専門エキスパート（構造設計、道路設計、地盤工学、水理・排水、材料工学、施工管理、法規・基準、環境・維持管理）による高精度な技術回答を提供。

## 主要機能
1. **MoEシステム**: 専門分野別エキスパートモデルによる高精度回答
2. **RAGシステム**: Qdrantベクトルデータベースによる文書検索・知識拡張
3. **ファインチューニング**: LoRA/QLoRA/フル訓練対応
4. **継続学習**: EWCベースの破滅的忘却防止機構
5. **統合Webインターフェース**: ポート8050で全機能アクセス可能

## システムアーキテクチャ
- **フロントエンド**: HTML/JavaScript/Bootstrap
- **バックエンド**: FastAPI (Python)
- **モデル管理**: Transformers、PEFT、Accelerate
- **ベクトルDB**: Qdrant
- **推論最適化**: vLLM、AWQ量子化、Ollama統合
- **インフラ**: Docker、NVIDIA GPU対応

## プロジェクト構造
```
MoE_RAG/
├── app/               # Webアプリケーション
├── src/              # コアロジック
│   ├── moe/          # MoEアーキテクチャ
│   ├── rag/          # RAGシステム
│   ├── training/     # 訓練モジュール
│   └── inference/    # 推論最適化
├── data/             # データセット
├── scripts/          # ユーティリティスクリプト
├── docker/           # Docker設定
├── config/           # 設定ファイル
└── outputs/          # 学習済みモデル
```

## 最新の実装
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- vLLM統合による高速推論
- AWQ 4ビット量子化
- Ollama統合 (Llama 3.2 3B)
- WebSocketストリーミング応答