# 技術スタックと依存関係

## コア技術
- **言語**: Python 3.8+ (推奨3.11)
- **Webフレームワーク**: FastAPI, Uvicorn
- **MLフレームワーク**: PyTorch 2.0+, Transformers 4.30+
- **ファインチューニング**: PEFT, Accelerate, BitsAndBytes
- **ベクトルDB**: Qdrant
- **コンテナ**: Docker, Docker Compose

## 主要ライブラリ
- transformers: LLMモデル管理
- peft: LoRA/QLoRAファインチューニング
- accelerate: マルチGPU訓練
- bitsandbytes: 量子化
- datasets: データセット管理
- wandb: 実験追跡
- tensorboard: 訓練監視

## 開発ツール
- pytest: テスティング
- black: コードフォーマッター (line-length=88)
- flake8: リンター
- isort: インポート整理 (profile=black)

## システム要件
- CUDA 12.6+
- NVIDIA GPU (8GB VRAM以上)
- システムメモリ 16GB+
- Ollama (オプション)