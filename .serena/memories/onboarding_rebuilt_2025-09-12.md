# MoE_RAG Onboarding (Rebuilt • 2025-09-12)

目的: 新規/復帰メンバーが最短で開発・検証・運用を開始できるよう、現行リポジトリ構成・主要コマンド・品質基準を一つに集約します。

## 概要
- 統合FastAPIアプリを単一ポート(8050)で提供。RAG/ファインチューニング/継続学習/MoEを統合。
- ローカル開発/テストと、Dockerスタック運用の双方に対応。
- ベクトルDBはQdrant、Ollama連携(11434)あり。

## ディレクトリ構成(要点)
- `app/` — FastAPI本体。`app/main_unified.py` が統合エントリ。
- `src/` — コア実装
  - `rag/`(query engine, indexing, retrieval, config)
  - `training/`(LoRA/DoRA、継続学習EWC等)
  - `moe_rag_integration/`(MoE統合レイヤ)
  - `inference/`, `utils/`, (存在するため `moe/` も同梱)
- `scripts/` — 運用/支援スクリプト。`start_web_interface.sh` 等。
- `tests/` — pytestスイート(`test_*.py`)。
- `docker/` — Dockerfile と `docker-compose.yml`。
- `config/`, `configs/` — 実行/訓練設定(例: `config/rag_config.yaml`, `config/model_config.yaml`)。
- `data/`, `models/`, `outputs/`, `templates/` — データと成果物。

## ローカル開発(推奨フロー)
1) セットアップ(任意のPython 3.8+)
```bash
python -m venv venv && source venv/bin/activate
pip install -e .[dev]   # 失敗時は: pip install -r requirements.txt
```
2) Lint/Format
```bash
black . && isort . && flake8 src tests
```
3) テスト(軽量)
```bash
pytest -q
# 重い/外部依存を除外:
pytest -m "not integration" -q
```
4) API起動(リロード有効)
```bash
python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```
5) 動作確認
- ブラウザ: http://localhost:8050/
- 代表API: `/rag/health`, `/api/system-info`

## Dockerスタック運用
1) ビルド/起動
```bash
cd docker && docker-compose up -d --build
# 2回目以降: docker-compose up -d
```
2) Webインターフェース起動(コンテナ内)
- README手順に準拠:
```bash
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
# プロダクション相当:
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh production
```
3) ポート/依存
- Web: `8050:8050`
- Ollama: `11434:11434`
- Qdrant: ボリューム `qdrant_storage`

## コーディング規約
- Python 3.8+。Black(88)・isort(profile "black")準拠。
- 命名: モジュール/関数/変数=snake_case、クラス=CapWords、定数=UPPER_SNAKE。
- `src/rag/` と `app/` の公開APIは互換性を維持。新規モジュールはdocstring付与。

## テスト指針
- フレームワーク: pytest。
- 位置: `tests/test_*.py`。
- マーク: 長時間/外部依存は `@pytest.mark.integration`。
- 事前チェック: `pytest -q && flake8 && black --check . && isort --check-only .`

## コミット/PR
- Conventional Commits: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`。
- PRには目的/範囲/テスト計画/ログやスクショ/関連Issue/設定変更の移行ノートを含める。
- `data/`, `models/`, `outputs/` の大容量成果物はコミットしない(小規模サンプルのみ許可)。

## セキュリティと設定
- 秘密情報は `.env`(例: `.env.example`) を使用。鍵はコミット禁止。
- 実行/訓練の既定は `config/` と `src/rag/config/` に整理。上書きルールをREADMEに沿って明記。
- GPU/Dockerパスは `scripts/` と `docker/` から参照されるため、リネーム時は両者を更新。

## 主要コマンド早見表
- セットアップ: `pip install -e .[dev]`
- Lint/Format: `black . && isort . && flake8 src tests`
- テスト: `pytest -q` / `pytest -m "not integration" -q`
- ローカル起動: `python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`
- Docker: `cd docker && docker-compose up -d --build` → `docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh`

## クイック検証(HTTP)
```bash
curl -I http://localhost:8050/
curl http://localhost:8050/rag/health
curl http://localhost:8050/api/system-info
```

## トラブルシューティング
- ポート8050が占有: `lsof -i:8050` → 該当プロセス停止後に再起動。
- Qdrant接続: コンテナ起動順/ボリューム権限を確認(READMEの診断手順参照)。
- モデル/メモリ不足: 量子化/CPU offload設定を有効化。バッチ縮小。
- Ollama応答なし: `docker-compose-ollama.yml` と `11434` の疎通を確認。

## Serenaメモの運用
- 本オンボーディングは「onboarding_rebuilt_2025-09-12」。
- 旧メモ(例: `onboarding_2025`)は参照専用。置換したい場合は依頼してください(安全のため手動削除運用)。

## 次の一歩
- 初回はローカル起動→`/rag/health`の200応答を確認。
- Docker環境で`start_web_interface.sh`まで通し、ダッシュボードのUI遷移を確認。
- 作業前に `black/isort/flake8/pytest` を通すのを習慣化。
