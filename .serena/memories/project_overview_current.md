# MoE-RAG Project Overview (Current State)

## プロジェクト名
**AI Fine-tuning Toolkit (AI_FT_7)** with MoE-RAG Integration

## 概要
土木工学・道路設計分野に特化した日本語LLMファインチューニング＆RAG統合プラットフォーム

## 主要特徴
1. **統合Webインターフェース** - 単一ポート(8050)でファインチューニングとRAG機能を提供
2. **MoE (Mixture of Experts)** - 複数の専門モデルを組み合わせた高度な推論
3. **継続学習** - EWCベースのタスク継続学習機能
4. **ハイブリッドRAG** - ベクトル検索とキーワード検索の組み合わせ
5. **メモリ最適化** - 動的量子化とCPU offloading対応

## 技術スタック
- **フレームワーク**: FastAPI, PyTorch, Transformers
- **ベクトルDB**: Qdrant
- **LLMモデル**: Llama 3.2, CALM3, Swallow
- **推論**: vLLM, Ollama
- **量子化**: AWQ, GPTQ, BitsAndBytes
- **UI**: Bootstrap, JavaScript

## 主要コンポーネント
1. **ファインチューニングシステム**
   - LoRA/DoRA対応
   - マルチGPU訓練
   - EWC継続学習

2. **RAGシステム**
   - 文書処理（PDF, OCR, 表抽出）
   - ハイブリッド検索
   - 数値処理・設計基準検証

3. **推論システム**
   - vLLM高速推論
   - AWQ 4ビット量子化
   - Ollama統合

4. **MoEシステム**
   - エキスパートルーティング
   - データセット管理
   - 訓練・推論統合

## デプロイメント
- **Docker**: 完全コンテナ化環境
- **ポート**: 8050 (Web UI/API)、11434 (Ollama)
- **GPU**: NVIDIA CUDA対応
- **メモリ**: 大規模モデル用最適化実装

## リポジトリ情報
- **GitHub**: https://github.com/kji-furuta/MoE_RAG.git
- **ブランチ**: main (本番)、rag-development-* (開発)
- **CI/CD**: Docker自動ビルド対応

## 現在の状態
- ✅ 統合Webインターフェース稼働中
- ✅ RAGシステム完全動作
- ✅ ファインチューニング機能実装済み
- ✅ 継続学習パイプライン構築済み
- ✅ MoE統合完了
- ✅ Ollama連携実装済み