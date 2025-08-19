# プロジェクト構造

## ディレクトリ構成

```
MoE_RAG/
├── app/                      # Webアプリケーション層
│   ├── main_unified.py       # 統合FastAPIサーバー (ポート8050)
│   ├── moe_rag_api.py        # MoE-RAG API実装
│   ├── moe_rag_endpoints.py  # MoE-RAGエンドポイント定義
│   ├── memory_optimized_loader.py  # モデルローダー（量子化対応）
│   ├── fixed_ewc_trainer.py  # EWC継続学習トレーナー
│   └── static/               # 静的ファイル（HTML/JS/CSS）
│       ├── moe_rag_ui.html  # MoE-RAG UI
│       └── moe_training.html # 訓練管理UI
│
├── src/                      # コアロジック
│   ├── moe/                  # MoEシステム
│   │   ├── moe_architecture.py  # MoEモデル定義
│   │   ├── moe_training.py      # MoE訓練ロジック
│   │   └── data_preparation.py  # データ前処理
│   │
│   ├── moe_rag_integration/  # MoE-RAG統合
│   │   ├── expert_router.py     # エキスパート選択
│   │   ├── hybrid_query_engine.py # ハイブリッド検索
│   │   ├── response_fusion.py   # 応答統合
│   │   └── moe_serving.py       # MoEサービング
│   │
│   ├── rag/                  # RAGシステム
│   │   ├── core/             # コア機能
│   │   │   └── query_engine.py  # クエリ処理エンジン
│   │   ├── indexing/         # インデックス作成
│   │   │   └── vector_store.py  # Qdrantベクトルストア
│   │   ├── retrieval/        # 検索機能
│   │   │   └── hybrid_search.py # ハイブリッド検索
│   │   └── document_processing/ # 文書処理
│   │
│   ├── training/             # 訓練モジュール
│   │   ├── lora_finetuning.py   # LoRA訓練
│   │   ├── full_finetuning.py   # フル訓練
│   │   ├── ewc_utils.py         # EWC実装
│   │   ├── multi_gpu_training.py # マルチGPU
│   │   └── dora/                 # DoRA実装
│   │       └── dora_implementation.py
│   │
│   └── inference/            # 推論最適化
│       ├── vllm_integration.py   # vLLM統合
│       └── awq_quantization.py   # AWQ量子化
│
├── data/                     # データディレクトリ
│   ├── civil_engineering/    # 土木工学データ
│   │   ├── train/           # 訓練データ（8分野）
│   │   └── val/             # 検証データ
│   ├── continual_learning/  # 継続学習データ
│   └── uploaded/            # アップロードファイル
│
├── scripts/                  # ユーティリティスクリプト
│   ├── moe/                 # MoE関連
│   │   ├── train_moe.sh    # MoE訓練スクリプト
│   │   └── run_training.py  # 訓練実行
│   ├── rag/                 # RAG関連
│   │   └── index_documents.py # 文書インデックス作成
│   ├── test/                # テストスクリプト
│   │   └── simple_lora_tutorial.py
│   └── *.sh, *.py           # 各種ユーティリティ
│
├── docker/                   # Docker設定
│   ├── docker-compose.yml   # コンテナ構成
│   └── Dockerfile           # コンテナイメージ定義
│
├── config/                   # 設定ファイル
│   ├── model_config.yaml    # モデル設定
│   └── rag_config.yaml      # RAG設定
│
├── configs/                  # 追加設定
│   └── deepspeed/           # DeepSpeed設定
│
├── templates/               # HTMLテンプレート
│   └── base.html           # ベーステンプレート
│
├── outputs/                 # 出力ディレクトリ
│   ├── trained_models/     # 訓練済みモデル
│   ├── lora_adapters/      # LoRAアダプター
│   └── ewc_data/           # EWCデータ
│
├── tests/                   # テストコード
│   └── test_*.py           # 各種テスト
│
├── docs/                    # ドキュメント
│
├── reports/                 # レポート・ログ
│
├── ollama_models/          # Ollamaモデル
│
└── ルートファイル
    ├── README.md           # プロジェクト説明
    ├── MoE_RAG_README.md   # MoE-RAG詳細
    ├── IMPLEMENTATION_GUIDE.md # 実装ガイド
    ├── CLAUDE.md           # Claude用ガイド
    ├── requirements.txt    # Python依存関係
    ├── pyproject.toml      # プロジェクト設定
    ├── setup.py            # セットアップスクリプト
    └── .env.example        # 環境変数テンプレート
```

## 主要コンポーネントの役割

### app/ - アプリケーション層
- FastAPIベースのWeb API実装
- 静的ファイル配信
- WebSocket対応

### src/ - ビジネスロジック層
- MoE、RAG、訓練の中核実装
- モジュール間の疎結合設計
- 再利用可能なコンポーネント

### data/ - データ管理
- 8つの専門分野別データセット
- JSONLフォーマットの訓練データ
- アップロードファイルの一時保存

### scripts/ - 運用スクリプト
- 自動化されたタスク実行
- テスト・デバッグツール
- システム管理ユーティリティ

### docker/ - コンテナ化
- 開発・本番環境の統一
- GPU対応設定
- ボリュームマウント管理

### config/ - 設定管理
- YAML形式の設定ファイル
- 環境別設定の分離
- デフォルト値の定義