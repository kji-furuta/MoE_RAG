# MoE-RAG プロジェクト概要 (2025年9月18日最新)

## プロジェクトの目的
**AI Fine-tuning Toolkit with RAG Integration & Continual Learning**
- 日本語LLMファインチューニング統合プラットフォーム
- 土木工学・道路設計特化型RAGシステム
- EWCベース継続学習機能
- DeepSeek-R1-Distill-Qwen-32Bモデル対応
- 単一ポート（8050）での統合Webインターフェース

## GitHubリポジトリ
- URL: https://github.com/kji-furuta/MoE_RAG.git
- ブランチ: rag-development-20250901
- 最新コミット: 26829ba (2025/09/18)

## システムの特徴
1. **統合プラットフォーム**: ファインチューニング、RAG、継続学習を単一UIで管理
2. **メモリ最適化**: 32Bモデルの4bit量子化対応
3. **ハイブリッド検索**: BM25 + ベクトル検索の組み合わせ
4. **Ollama統合**: ローカルLLMとの連携
5. **Docker対応**: 完全コンテナ化環境

## 動作確認済み機能 (2025/09/18テスト)
- ✅ DeepSeek-32B LoRAファインチューニング
- ✅ 継続学習（outputs/lora_20250918_135106使用）
- ✅ GGUF変換とOllama登録
- ✅ RAGシステムのベクトル・キーワード検索
- ✅ PDFアップロードによる文書処理

## 主要な最新修正
1. PeftModelのis_trainable=True対応
2. 量子化モデルでのLoRAパラメータ勾配設定
3. startup_event重複定義の統合
4. 未定義変数エラーの修正
5. CUDAアロケータエラー解決