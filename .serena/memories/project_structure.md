# プロジェクト構造詳細

## ディレクトリ構成
```
AI_FT_3/
├── app/                      # Webインターフェース
│   ├── main_unified.py       # 統合FastAPIサーバー（メイン）
│   ├── memory_optimized_loader.py  # モデルローダー
│   ├── static/               # 静的ファイル（CSS/JS）
│   └── continual_learning/   # 継続学習UI
│
├── src/                      # コアロジック
│   ├── training/             # ファインチューニング
│   │   ├── lora_finetuning.py
│   │   ├── full_finetuning.py
│   │   ├── ewc_utils.py     # Elastic Weight Consolidation
│   │   └── multi_gpu_training.py
│   │
│   ├── rag/                  # RAGシステム
│   │   ├── core/             # コアエンジン
│   │   │   └── query_engine.py
│   │   ├── indexing/         # インデックス管理
│   │   │   └── vector_store.py
│   │   ├── retrieval/        # 検索機能
│   │   │   └── hybrid_search.py
│   │   ├── document_processing/  # 文書処理
│   │   └── specialized/     # 特殊機能
│   │
│   ├── data/                 # データ処理
│   ├── utils/                # ユーティリティ
│   └── evaluation/           # 評価メトリクス
│
├── docker/                   # Docker設定
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/                  # ユーティリティスクリプト
│   ├── test/                 # テストスクリプト
│   ├── rag/                  # RAG関連
│   ├── continual_learning/  # 継続学習
│   └── setup/                # セットアップ
│
├── configs/                  # 設定ファイル
│   ├── model_config.yaml
│   ├── ewc_config.yaml
│   └── continual_tasks.yaml
│
├── templates/                # HTMLテンプレート
│   ├── base.html            # ベーステンプレート
│   ├── index.html           # メインページ
│   └── rag.html             # RAGインターフェース
│
├── data/                     # データディレクトリ
│   ├── uploaded/            # アップロードファイル
│   ├── sample_continual.jsonl
│   └── search_history/      # 検索履歴
│
├── outputs/                  # 出力モデル
│   └── lora_adapters/       # LoRAアダプター
│
├── tests/                    # テストコード
├── docs/                     # ドキュメント
├── requirements.txt          # Python依存関係
├── requirements_rag.txt      # RAG専用依存関係
├── pyproject.toml           # プロジェクト設定
├── README.md                # プロジェクト説明
└── CLAUDE.md                # 開発ガイドライン
```

## 重要ファイル
- **app/main_unified.py**: メインエントリーポイント
- **src/rag/core/query_engine.py**: RAGエンジン
- **src/training/lora_finetuning.py**: LoRA訓練
- **docker/docker-compose.yml**: コンテナ設定
- **scripts/start_web_interface.sh**: 起動スクリプト