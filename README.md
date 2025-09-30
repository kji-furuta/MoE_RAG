# MoE-RAG: AI Fine-tuning Toolkit with RAG Integration & Continual Learning

<div align="center">

🚀 **日本語LLMファインチューニング + RAGシステム + 継続学習統合プラットフォーム**

[![GitHub](https://img.shields.io/badge/GitHub-MoE__RAG-blue?logo=github)](https://github.com/kji-furuta/MoE_RAG.git)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📋 目次

- [概要](#-概要)
- [最新アップデート](#-最新アップデート-2025-09-30)
- [主要機能](#-主要機能)
- [クイックスタート](#-クイックスタート)
- [システムアーキテクチャ](#-システムアーキテクチャ)
- [使用方法](#-使用方法)
- [ドキュメント](#-ドキュメント)
- [トラブルシューティング](#-トラブルシューティング)
- [開発者ガイド](#-開発者ガイド)
- [ライセンス](#-ライセンス)

---

## 📖 概要

**MoE-RAG**は、Dockerベースの統合Webインターフェースで、日本語大規模言語モデル（LLM）のファインチューニング、土木道路設計特化型RAGシステム、そしてEWCベースの継続学習を同一プラットフォームで実行できる革新的なツールキットです。

### ✨ 特徴

- **🌐 統合Webインターフェース**: 単一ポート（8050）で全機能にアクセス
- **🔥 多様なファインチューニング**: LoRA、DoRA、QLoRA、フルファインチューニング対応
- **🔄 継続学習システム**: EWCによる破滅的忘却防止
- **🏗️ RAGシステム**: 土木道路設計分野に特化したハイブリッド検索
- **🚀 高性能推論**: vLLM、AWQ量子化、Ollama統合
- **🎯 MoEアーキテクチャ**: Mixture of Expertsによる効率的学習

### 🎯 対象ユーザー

- AI研究者・エンジニア
- 土木・道路設計分野の技術者
- LLMファインチューニングを学びたい開発者
- 日本語モデルのカスタマイズが必要な企業

---

## 🎉 最新アップデート (2025-09-30)

### ✅ 全システム完全稼働確認

**3大必須機能の動作検証完了:**

1. **ファインチューニングシステム** ✅
   - LoRA/DoRA/フルファインチューニング正常動作
   - 4bit/8bit量子化対応
   - DeepSeek-R1-Distill-Qwen-32B対応

2. **継続学習システム** ✅
   - EWCベースの継続学習完全稼働
   - Fisher情報行列の計算・保存・適用
   - タスク状態管理と履歴追跡
   - Web UI統合 (`/continual`)

3. **RAGシステム** ✅
   - ハイブリッド検索（ベクトル + BM25）
   - PDFアップロード・インデックス化
   - ファインチューニングモデル統合
   - 技術用語ブースト機能

### 🔧 最近の改善

- **Ollamaモデル自動登録**: ワイルドカード検索でGGUFモデル自動検出
- **タスク状態修正**: 継続学習task_10の状態を完了に修正
- **メモリ最適化**: 32Bモデルの4bit量子化対応
- **統合テスト**: 全機能の動作確認済み

---

## 🌟 主要機能

### 1. ファインチューニングシステム

<details>
<summary>詳細を見る</summary>

#### サポートする学習手法

| 手法 | 説明 | メモリ効率 | 精度 |
|------|------|------------|------|
| **LoRA** | 低ランク適応 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **DoRA** | 重み分解LoRA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **QLoRA** | 量子化LoRA | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **フルファインチューニング** | 全パラメータ更新 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

#### サポートモデル

- **DeepSeek-R1-Distill-Qwen-32B** (推奨) - GGUF対応、Ollama統合
- **CALM3-22B-chat** - 日本語特化
- **Llama 3.2 3B** - 軽量・高速
- その他HuggingFace互換モデル

</details>

### 2. 継続学習システム (NEW)

<details>
<summary>詳細を見る</summary>

#### EWC (Elastic Weight Consolidation)

- **破滅的忘却防止**: Fisher情報行列による重要パラメータ保護
- **複数タスク学習**: 過去の知識を保持しながら新タスク学習
- **タスク管理**: リアルタイム進捗監視とWeb UI統合

#### 技術仕様

```yaml
ewc:
  lambda: 5000  # EWC正則化パラメータ
  fisher_calculation: block_wise  # メモリ効率的計算
  task_history: outputs/ewc_data/task_history.json
  models: outputs/continual_task_*
```

#### 使用例

```bash
# Web UIから実行
http://localhost:8050/continual

# 手動実行
python src/training/continual_learning_pipeline.py \
  --base-model outputs/lora_model \
  --task-name task_02 \
  --data data/continual/task_02_data.jsonl
```

</details>

### 3. RAGシステム（土木道路設計特化）

<details>
<summary>詳細を見る</summary>

#### 主要機能

- **ハイブリッド検索**: ベクトル検索（0.7）+ BM25（0.3）
- **多層リランキング**: Cross-encoder + 技術用語ブースト
- **数値処理**: 設計速度、曲線半径、勾配の自動抽出
- **設計基準チェック**: 道路構造令準拠の適合性検証
- **引用機能**: 正確な出典情報付き回答生成

#### 技術用語ブースト

```python
# 技術用語を含む文書に最大20%のスコアボーナス
boosted_score = hybrid_score * (1.0 + tech_boost)

# 対応技術用語
- 数値と単位: 80km/h、100m、5%
- 設計基準値: 最小曲線半径、最大勾配
- 道路部位: 車道、歩道、中央分離帯
- 専門用語: 設計速度、曲線半径、縦断勾配
```

#### API例

```bash
# RAGクエリ
curl -X POST "http://localhost:8050/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "設計速度80km/hの道路の最小曲線半径は？",
    "top_k": 5,
    "search_type": "hybrid"
  }'

# 文書アップロード
curl -X POST "http://localhost:8050/rag/upload-document" \
  -F "file=@道路設計基準.pdf" \
  -F "title=道路設計基準書"
```

</details>

### 4. 推論最適化

<details>
<summary>詳細を見る</summary>

- **vLLM統合**: PagedAttentionによる3倍高速推論
- **AWQ量子化**: 4bit量子化で75%メモリ削減
- **Ollama統合**: 自動モデル登録・同期
- **GGUF変換**: LoRAアダプターの自動GGUF化

</details>

---

## 🚀 クイックスタート

### 前提条件

- **ハードウェア**: NVIDIA GPU (CUDA対応)、最低8GB VRAM
- **ソフトウェア**: Docker, Docker Compose, Git
- **オプション**: Ollama（ローカルLLM統合用）

### インストール（3ステップ）

```bash
# 1. リポジトリクローン
git clone https://github.com/kji-furuta/MoE_RAG.git
cd MoE_RAG

# 2. Docker環境起動
cd docker && docker-compose up -d --build  # 初回のみ --build

# 3. Webインターフェース起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### アクセス

- **メインダッシュボード**: http://localhost:8050/
- **ファインチューニング**: http://localhost:8050/finetune
- **継続学習**: http://localhost:8050/continual
- **RAGシステム**: http://localhost:8050/rag
- **モデル管理**: http://localhost:8050/models

### Ollama統合（オプション）

```bash
# Ollamaインストール
curl -fsSL https://ollama.com/install.sh | sh

# Ollamaサービス起動（別ターミナル）
ollama serve

# Llama 3.2 3Bモデルダウンロード
ollama pull llama3.2:3b
```

---

## 🏗️ システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│              Web Browser (localhost:8050)               │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/WebSocket
┌─────────────────▼───────────────────────────────────────┐
│              FastAPI Server (Port 8050)                 │
│                 app/main_unified.py                     │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Fine-tune  │  │   RAG    │  │ Continual Learn  │   │
│  │   Routes   │  │  Routes  │  │     Routes       │   │
│  └──────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
└─────────┼──────────────┼─────────────────┼─────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼─────────────┐
│                  Core Services Layer                    │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐    │
│  │  Training   │  │    RAG     │  │     MoE      │    │
│  │   Engine    │  │   Engine   │  │   Manager    │    │
│  └─────────────┘  └────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────┘
          │              │                 │
┌─────────▼──────────────▼─────────────────▼─────────────┐
│              Infrastructure Layer                       │
├──────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌──────────────┐    │
│  │   Models    │  │   Qdrant   │  │    Ollama    │    │
│  │  Storage    │  │   Vector   │  │   Service    │    │
│  │             │  │     DB     │  │ (Port 11434) │    │
│  └─────────────┘  └────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 主要コンポーネント

- **app/main_unified.py**: 統合FastAPIサーバー
- **src/training/**: ファインチューニング・継続学習エンジン
- **src/rag/**: RAGクエリエンジン・ベクトルストア
- **src/inference/**: vLLM・AWQ推論最適化
- **docker/**: コンテナ環境設定

---

## 📚 使用方法

### 1. ファインチューニング

```bash
# Web UIから実行
1. http://localhost:8050/finetune にアクセス
2. データアップロード（JSONL形式）
3. モデル選択（DeepSeek-32B推奨）
4. LoRA/DoRA/QLoRA選択
5. 学習開始ボタンをクリック
```

### 2. 継続学習

```bash
# Web UIから実行
1. http://localhost:8050/continual にアクセス
2. ベースモデル選択
3. タスク名とデータセットアップロード
4. EWCパラメータ設定（λ=5000推奨）
5. 学習開始

# 学習済みモデルは自動保存
outputs/continual_task_*/checkpoint-final/
```

### 3. RAG検索

```bash
# Web UIから実行
1. http://localhost:8050/rag にアクセス
2. PDF文書アップロード
3. 質問を自然言語で入力
   例: "設計速度80km/hの道路の最小曲線半径は?"
4. ハイブリッド検索結果を取得
```

### 4. GGUF変換・Ollama登録

```bash
# LoRAアダプターをGGUF変換
docker exec ai-ft-container python /workspace/scripts/apply_lora_to_gguf_improved.py \
  --output-name "my-finetuned-model"

# Ollamaモデル確認
ollama list

# モデル削除
ollama rm model_name:latest
```

---

## 📖 ドキュメント

### 主要ドキュメント

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: システム設計詳細
- **[API_REFERENCE.md](docs/API_REFERENCE.md)**: API仕様書
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: 開発者ガイド
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: デプロイメント手順
- **[CLAUDE.md](CLAUDE.md)**: Claude Code向けガイド

### Web上のドキュメント

- **利用マニュアル**: http://localhost:8050/manual
- **システム概要**: http://localhost:8050/system-overview

### 技術ドキュメント

<details>
<summary>既存ドキュメント一覧</summary>

- [大規模モデルセットアップ](docs/LARGE_MODEL_SETUP.md)
- [マルチGPU最適化](docs/MULTI_GPU_OPTIMIZATION.md)
- [RAGアーキテクチャ](docs/ROAD_DESIGN_RAG_ARCHITECTURE.md)
- [依存関係管理](docs/DEPENDENCY_MANAGEMENT.md)
- [パフォーマンス最適化](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [監視ガイド（Grafana）](docs/MONITORING_GRAFANA_GUIDE.md)
- [Ollamaモデル同期](docs/OLLAMA_MODEL_SYNC_SOLUTION.md)

</details>

---

## 🔧 トラブルシューティング

### よくある問題

<details>
<summary>Webインターフェースが起動しない</summary>

```bash
# コンテナ状態確認
docker ps -a

# ログ確認
docker logs ai-ft-container --tail 50

# 再起動
docker exec ai-ft-container pkill -f uvicorn
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

</details>

<details>
<summary>Ollama「model not found」エラー</summary>

```bash
# Ollama起動確認
curl http://localhost:11434/api/tags

# モデルダウンロード
ollama pull llama3.2:3b

# Ollama再起動
killall ollama && ollama serve
```

</details>

<details>
<summary>メモリ不足エラー</summary>

```bash
# GPU メモリ確認
nvidia-smi

# バッチサイズ削減（設定ファイル編集）
# batch_size: 4 → 1

# 量子化有効化
# use_8bit: true または use_4bit: true
```

</details>

<details>
<summary>継続学習タスクが失敗と表示される</summary>

```bash
# タスク状態確認
cat data/continual_learning/tasks_state.json

# モデルファイル確認
ls -la outputs/continual_task_*/checkpoint-final/

# サーバー再起動
docker exec ai-ft-container pkill -f uvicorn
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

</details>

詳細なトラブルシューティングは[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)を参照してください。

---

## 👨‍💻 開発者ガイド

### 開発環境セットアップ

```bash
# リポジトリクローン
git clone https://github.com/kji-furuta/MoE_RAG.git
cd MoE_RAG

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

### コーディング規約

- **フォーマッター**: Black（行長88）/ isort（profile "black"）
- **命名規則**:
  - モジュール/関数/変数: `snake_case`
  - クラス: `CapWords`
  - 定数: `UPPER_SNAKE`
- **コミット**: Conventional Commits形式（`feat:`, `fix:`, `docs:`）

### テスト実行

```bash
# 全テスト実行
pytest -q

# 統合テスト除外
pytest -m "not integration" -q

# 特定テスト
python scripts/test_integration.py
python scripts/test_docker_rag.py
python scripts/test_continual_learning_integration.py
```

### プルリクエスト

1. リポジトリをフォーク
2. 機能ブランチ作成: `git checkout -b feature/new-feature`
3. 変更コミット: `git commit -m 'feat: add new feature'`
4. ブランチプッシュ: `git push origin feature/new-feature`
5. プルリクエスト作成

---

## 📊 プロジェクト構造

```
MoE_RAG/
├── app/                      # Webアプリケーション
│   ├── main_unified.py       # 統合FastAPIサーバー
│   ├── static/               # 静的ファイル
│   └── ollama_integration.py # Ollama統合
├── src/                      # コアライブラリ
│   ├── training/             # ファインチューニング・継続学習
│   ├── rag/                  # RAGシステム
│   ├── inference/            # vLLM・AWQ推論
│   ├── moe/                  # MoEアーキテクチャ
│   └── utils/                # ユーティリティ
├── docker/                   # Docker設定
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                  # 運用スクリプト
│   ├── start_web_interface.sh
│   ├── init_ollama_models.sh
│   └── apply_lora_to_gguf_improved.py
├── data/                     # データディレクトリ
│   ├── continual_learning/   # 継続学習データ
│   └── rag_documents/        # RAG文書
├── models/                   # モデルストレージ
├── outputs/                  # 学習済みモデル
│   ├── continual_task_*/     # 継続学習モデル
│   ├── ewc_data/             # Fisher行列
│   └── moe_*/                # MoEモデル
├── templates/                # HTMLテンプレート
├── config/                   # 設定ファイル
├── docs/                     # ドキュメント
└── README.md                 # このファイル
```

---

## 📅 更新履歴

### v4.0.0 (2025-09-30) - 全システム統合完了

- ✅ ファインチューニング・継続学習・RAGシステム完全稼働確認
- ✅ Ollamaモデル自動登録機能（ワイルドカード検索）
- ✅ 継続学習タスク状態管理の改善
- ✅ GGUF変換・Ollama登録の安定化

### v3.2.0 (2025-09-25) - 継続学習システム完全稼働

- ✅ EWCベース継続学習の完全動作確認
- ✅ Fisher情報行列の計算・保存・適用
- ✅ タスク履歴管理と永続化
- ✅ Web UI統合（`/continual`）

### v3.1.0 (2025-09-18) - 量子化モデル対応

- ✅ DeepSeek-32Bの継続学習対応
- ✅ 4bit量子化でのLoRA学習
- ✅ GGUF変換・Ollama登録の改善
- ✅ メモリアロケータエラー解決

<details>
<summary>過去のバージョン</summary>

### v3.0.0 (2025-08-16)
- DoRA実装、vLLM統合、AWQ量子化

### v2.4.0 (2025-08-08)
- フェーズ2完了、依存関係管理、監視機能

### v2.0.0 (2025-07-26)
- RAGシステム統合（単一ポート8050）

### v1.0.0 (2025-07-24)
- 初期リリース

</details>

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

## 🙏 謝辞

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [Qdrant](https://qdrant.tech/)
- [Ollama](https://ollama.com/)
- [vLLM](https://github.com/vllm-project/vllm)

---

## 📞 サポート

- **GitHub Issues**: https://github.com/kji-furuta/MoE_RAG/issues
- **ドキュメント**: http://localhost:8050/manual

---

<div align="center">

**🚀 5分でファインチューニング・RAG・継続学習を開始！**

[クイックスタートに戻る](#-クイックスタート)

</div>