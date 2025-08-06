# AI Fine-tuning Toolkit (AI_FT_3) プロジェクト概要

## プロジェクト目的
日本語大規模言語モデル（LLM）のファインチューニングと土木道路設計特化型RAGシステムを統合したWebツールキット。単一ポート（8050）で全機能にアクセス可能。

## 主要機能
1. **ファインチューニング**: LoRA、QLoRA、フルファインチューニング、EWC継続学習
2. **RAGシステム**: ハイブリッド検索、多層リランキング、設計基準チェック機能
3. **統合Webインターフェース**: FastAPI基盤、Bootstrap UI、リアルタイム監視
4. **モデル管理**: 動的量子化、CPUオフロード、マルチGPU対応

## サポートモデル
- Qwen2.5シリーズ (14B, 32B)
- CyberAgent CALM3 (22B)
- Meta Llama 3.1 (32B, 70B)
- Microsoft Phi-3.5 (32B)

## アーキテクチャ
- **メインサーバー**: `app/main_unified.py` (FastAPI)
- **ポート**: 8050 (統合API)
- **データストア**: Qdrant (ベクトルDB)
- **認証**: HF_TOKEN, WANDB_API_KEY (環境変数)

## デプロイメント
Docker Compose環境で全サービスを管理。GPUアクセス必須。