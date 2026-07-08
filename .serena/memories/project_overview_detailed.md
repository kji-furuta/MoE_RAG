# AI Fine-tuning Toolkit (AI_FT_3) - 詳細プロジェクト概要

## プロジェクト概要
日本語大規模言語モデル（LLM）のファインチューニングと土木道路設計に特化したRAGシステムを統合した包括的なAIツールキット。

## 主要機能

### 1. ファインチューニング機能
- **LoRA (Low-Rank Adaptation)**: メモリ効率的な部分的ファインチューニング
- **QLoRA**: 4bit量子化を使用した超効率的ファインチューニング
- **フルファインチューニング**: モデル全体の更新
- **EWC継続学習**: Elastic Weight Consolidationによる継続的な学習

### 2. RAGシステム（土木道路設計特化）
- **ハイブリッド検索**: ベクトル検索とキーワード検索の組み合わせ
- **多層リランキング**: 検索結果の精度向上
- **設計基準チェック**: 道路設計規格との適合性確認
- **数値処理・計算検証**: 設計値の自動検証機能

### 3. 統合Webインターフェース
- **FastAPI基盤**: 高性能な非同期Webフレームワーク
- **Bootstrap UI**: レスポンシブなユーザーインターフェース
- **リアルタイム監視**: WebSocketによる訓練進捗の可視化
- **統合ポート**: 8050番ポートで全機能にアクセス可能

### 4. モデル管理機能
- **動的量子化**: GPU/CPUメモリの効率的利用
- **CPUオフロード**: 大規模モデルのメモリ管理
- **マルチGPU対応**: 分散訓練のサポート
- **Ollama統合**: 量子化モデルの簡易デプロイ

## サポートモデル
### 日本語モデル
- Qwen2.5シリーズ (14B, 17B, 32B)
- CyberAgent CALM3 (22B)
- CyberAgent DeepSeek-R1-Distill-Qwen-32B-Japanese
- ELYZA-japanese-Llama-2 (7B)
- Rinna japanese-gpt2-small
- StabilityAI japanese-stablelm-3b-4e1t-instruct

### 多言語モデル
- Meta Llama 3.1 (32B, 70B)
- Microsoft Phi-3.5 (32B)
- DistilGPT2（テスト用軽量モデル）

## アーキテクチャ概要
- **メインサーバー**: app/main_unified.py (FastAPI統合サーバー)
- **APIポート**: 8050 (全機能統合)
- **データストア**: Qdrant (ベクトルデータベース)
- **コンテナ化**: Docker Compose環境
- **認証**: HF_TOKEN, WANDB_API_KEY (環境変数)

## ユースケース
1. **AIモデルのカスタマイズ**: 特定ドメインへの適応
2. **土木設計支援**: 道路設計基準に準拠した情報検索・検証
3. **継続的学習**: 新しいデータでのモデル更新
4. **マルチモーダルRAG**: テキスト・表・図面の統合処理

## デプロイメント要件
- GPU: NVIDIA GPU (8GB VRAM以上推奨、32Bモデルは20GB以上)
- システムメモリ: 16GB以上
- Docker & Docker Compose
- CUDA 12.6以上