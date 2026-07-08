# Scripts Directory Cleanup - 2025-09-08

## 整理前の状態
- メインディレクトリ: 65個のPythonスクリプト（混在状態）
- テスト/デモ/デバッグスクリプトが散在
- 重複機能のスクリプトが複数存在

## 実施した整理作業

### 1. 削除したファイル
- **scripts/__pycache__/** - Pythonキャッシュディレクトリ
- **重複スクリプト**:
  - apply_lora_to_gguf.py（旧版）
  - apply_lora_to_gguf_auto.py（旧版） 
  - apply_lora_gpt_neox.py（旧版）
  - apply_lora_gpt_neox_dynamic.py（旧版）
  - cpu_only_quantize.py
  - dequantize_and_convert.py
  - efficient_quantize.py
  - auto_lora_to_ollama.py
- **シェルスクリプト**: メインディレクトリから削除

### 2. ファイル移動と整理

#### test/ディレクトリへ移動（25個）
- test_*.py（テストスクリプト）
- debug_*.py（デバッグスクリプト）
- simple_*.py（簡易デモスクリプト）

#### convert/ディレクトリへ移動
- *ollama*.py（Ollama関連）
- *quantiz*.py（量子化関連）
- *merge*.py（マージ関連）

### 3. ファイル名変更
- apply_lora_to_gguf_improved.py → apply_lora_to_gguf.py
- apply_lora_gpt_neox_full.py → apply_lora_gpt_neox.py

## 整理後の構成

### メインディレクトリ（21個の主要スクリプト）
```
scripts/
├── apply_lora_gpt_neox.py      # LoRA適用（GPT-NeoX）
├── apply_lora_to_gguf.py       # LoRA適用（GGUF）
├── check_available_models.py    # モデル確認
├── check_rag_dependencies.py    # RAG依存関係確認
├── create_sample_continual_datasets.py  # サンプルデータセット作成
├── disk_space_manager.py        # ディスク管理
├── force_update_rag_model.py    # RAGモデル更新
├── generate_deepspeed_configs.py # DeepSpeed設定生成
├── optimize_rag_system.py       # RAG最適化
├── preload_models.py            # モデル事前読み込み
├── prepare_training_data.py     # 訓練データ準備
├── system_diagnosis.py          # システム診断
├── system_status_report.py      # ステータスレポート
├── train_calm3_22b.py          # CALM3訓練
├── train_large_model.py        # 大規模モデル訓練
└── unified_model_processor.py   # 統合モデル処理
```

### サブディレクトリ
- **test/** (38個) - テスト、デバッグ、デモスクリプト
- **convert/** (15個) - 変換、量子化、Ollama関連
- **setup/** - セットアップスクリプト
- **utils/** - ユーティリティ
- **moe/** - MoE関連
- **rag/** - RAG関連
- **continual_learning/** - 継続学習関連

## 削減効果
- **削除**: 44個のスクリプト（重複・不要）
- **整理**: 65個 → 21個（メインディレクトリ）
- **可読性**: 機能別にディレクトリ分類完了