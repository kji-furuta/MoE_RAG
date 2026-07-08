# Src Directory Cleanup - 2025-09-08

## 実施した整理作業

### 削除したファイル
1. **__pycache__ディレクトリ** - 13個のPythonキャッシュディレクトリを削除
2. **src/inference.py** - 重複ファイル（src/inference/ディレクトリが存在）
3. **src/moe/moe_architecture_optimized.py** - 未使用の最適化版（参照なし）
4. **src/utils/logger.py** - 未使用のロガーユーティリティ（参照なし）
5. **src/utils/helpers.py** - 未使用のヘルパー関数（参照なし）
6. **src/rag/config/rag_config.yaml.bak** - バックアップファイル
7. **.pyc ファイル** - コンパイル済みPythonファイル

### 保持した重要なディレクトリ構造
```
src/
├── __init__.py
├── core/                   # コア機能（memory_manager, quantization_manager）
├── data/                   # データ処理（dataloader）
├── evaluation/             # 評価メトリクス（continual_metrics）
├── inference/              # 推論エンジン
│   ├── awq_quantization.py
│   └── vllm_integration.py
├── models/                 # モデル定義（base_model）※8箇所から参照
├── moe/                    # MoEシステム
│   ├── base_config.py
│   ├── constants.py
│   ├── data_preparation.py
│   ├── exceptions.py
│   ├── lora_to_moe_adapter.py
│   ├── moe_architecture.py  # ※5箇所から参照
│   ├── moe_training.py
│   └── utils.py
├── moe_rag_integration/    # MoE-RAG統合
├── rag/                    # RAGシステム
│   ├── app.py             # ※app/routers/rag.pyから参照
│   ├── auth/              # ※test scriptsから参照
│   ├── config/
│   ├── core/
│   ├── dependencies/
│   ├── document_processing/
│   ├── evaluation/
│   ├── indexing/
│   ├── monitoring/
│   ├── retrieval/
│   ├── specialized/
│   └── utils/
├── training/               # 訓練システム
│   ├── continual_learning_pipeline.py
│   ├── ewc_utils.py
│   ├── full_finetuning.py
│   ├── lora_finetuning.py
│   ├── multi_gpu_training.py
│   └── ...
└── utils/                  # ユーティリティ
    ├── gpu_utils.py
    └── model_discovery.py
```

## 削除による影響
- **影響なし**: 削除したファイルは全て未使用または重複
- **ディスク節約**: 約100MB（主にキャッシュファイル）
- **可読性向上**: 不要ファイルの除去により構造が明確化

## 重要な依存関係（維持確認済み）
- src/models/ → 8箇所から参照（training scripts, examples）
- src/moe/moe_architecture.py → 5箇所から参照（RAG, MoE統合）
- src/rag/app.py → app/routers/rag.pyから参照
- src/rag/auth/ → テストスクリプトから参照

## 整理結果
- **削除ファイル数**: 7個 + 13個のキャッシュディレクトリ
- **コード品質向上**: 未使用コードの除去
- **メンテナンス性向上**: 明確なディレクトリ構造