# 技術スタック (2025年9月18日更新)

## フレームワーク・ライブラリ

### Backend
- **FastAPI**: WebフレームワークとAPI
- **Uvicorn**: ASGIサーバー
- **Python 3.11**: メイン言語

### Machine Learning
- **PyTorch 2.0+**: 深層学習フレームワーク
- **Transformers 4.36+**: HuggingFace モデル
- **PEFT**: LoRA/QLoRAファインチューニング
- **BitsAndBytes**: 量子化ライブラリ
- **Accelerate**: 分散訓練

### RAG関連
- **Qdrant**: ベクトルデータベース
- **LangChain**: RAGパイプライン構築
- **Sentence-Transformers**: 埋め込みモデル
- **SpaCy**: 自然言語処理
- **PyPDF2/pdfplumber**: PDF処理
- **pytesseract**: OCR処理

### 推論最適化
- **vLLM**: 高速推論エンジン
- **AWQ**: 量子化技術
- **Ollama**: ローカルLLM管理

### Frontend
- **Jinja2**: テンプレートエンジン
- **Bootstrap 5**: CSSフレームワーク
- **JavaScript (Vanilla)**: クライアントサイド

### インフラ
- **Docker & Docker Compose**: コンテナ化
- **NVIDIA Container Toolkit**: GPU対応
- **PostgreSQL**: メタデータ管理（オプション）

### モニタリング・開発ツール
- **Weights & Biases**: 実験管理
- **tensorboard**: 訓練可視化
- **pytest**: テスティング
- **black/ruff**: コードフォーマッタ

## 主要依存バージョン
```
fastapi==0.104.1
transformers==4.36.2
torch==2.1.2+cu121
peft==0.7.1
qdrant-client==1.7.3
langchain==0.1.0
sentence-transformers==2.2.2
bitsandbytes==0.41.3
accelerate==0.25.0
```

## システム要件
- GPU: NVIDIA RTX 3090以上（24GB VRAM推奨）
- RAM: 32GB以上
- Storage: 100GB以上（モデル保存用）
- OS: Linux (Ubuntu 20.04/22.04推奨)