# AI駆動型タスクオートメーションのための設計図（深堀り版）
## MoE_RAGにおけるClaude Code・Vibe Coding・MCP活用の詳細

### 0. コンセプトと狙い
Claude Codeをはじめとするコーディングエージェントは、開発だけでなく実務タスク（照査、運用、ドキュメント化）にも適用可能です。本事例は「AIエージェントがタスクを管理・実行し、必要時のみ人間にエスカレーションする」働き方を、MoE_RAG（日本語LLMのFT/RAG/継続学習統合システム）で具体化したものです。Vibe Codingの反復ループとMCPサーバーのコンテキスト永続化により、継続的改善と運用標準化を実現します。

---

## 1. 全体アーキテクチャ
```
ブラウザ ──(HTTP)──▶ FastAPI (`app/main_unified.py`)
   │                        ├─ BackgroundTasks（学習/索引/変換）
   │                        ├─ RAGエンジン `src/rag/`
   │                        ├─ 継続学習 `src/training/`
   │                        └─ 推論/量子化 `src/inference/`
   │
   └── MCP（Serena/Cipher/GitHub/Context7/Playwright）
         ├─ セッション記憶・知識ベース（Serena, Cipher）
         ├─ 外部連携（GitHub）/ Web操作（Playwright）
         └─ 追加文脈提供（Context7）
```
- インフラ: `docker/docker-compose.yml`、起動: `scripts/start_web_interface.sh`（ポート8050）。
- データ: `data/`, `models/`, `outputs/`、テンプレート: `templates/`。
- テスト: `tests/`（pytest/unittest混在、`@pytest.mark.integration`有）。

---

## 2. Vibe Coding運用モデル（エージェント中心）
- 意図把握 → 粗実装 → 最小検証 → 改善の短サイクル。変更のたびに要件/制約を明示し、差分最小（Black 88, isort）で適用。
- タスク駆動: 「目的→入口（API/スクリプト）→成果物→検証→次アクション」を常に紐づけ。
- 代表パターン:
  - コード生成/修正→`pytest -q`→`uvicorn`で動作検証→ドキュメント更新。
  - RAG索引→検索品質検証（Top-k再現率/回答妥当性）→UI連携確認。
  - 量子化/変換→`llama.cpp`/Ollama統合→応答遅延計測。

---

## 3. MCP連携の詳細（.mcp.json）
- 構成: `serena`（プロジェクト記憶/文脈管理）、`cipher`（ローカル永続メモリ）、`github`（PR/Issue操作）、`context7`（外部文脈付与）、`playwright`（Web操作）。
- 典型フロー:
  1) セッション開始: Serenaで前回コンテキストをロード（例: "architecture_current"）。
  2) タスク実行: Claude Codeがコード/ドキュメント/検証を自動化。
  3) 記録更新: 重要メモリを`write_memory`で永続化（決定/KPI/TODO）。
  4) エスカレーション: 失敗/不確実性時は人間へ通知し承認を取得。
- セキュリティ: `.env`で秘匿、GitHubトークンは環境変数参照、長期メモリは`data/cipher-sessions.db`に保存。

---

## 4. タスク自動化パターン（テンプレ）
1) 照査（コード検証レポート）
- 入力: 変更差分/要件、出力: レポート（品質指標/改善案）。
- 実装: 静的解析→テスト生成→`pytest`実行→Markdown生成。

2) RAG文書インデクシング
- 入力: PDF/HTML群、出力: ベクトル索引・メタデータ。
- 実装: 並列解析→メタ抽出→`src/rag/indexing/`経由で格納。

3) 継続学習（EWC）
- 入力: 新タスク設定、出力: 更新済み重み/評価。
- 実装: Fisher計算→制約付き微調整→ベンチマーク→履歴保存。

4) モデル変換/量子化
- 入力: LoRA/全学習重み、出力: GGUF/Ollama登録。
- 実装: `scripts/convert_to_gguf.py`等→`llama.cpp`→`ollama create`。

---

## 5. API/ジョブ実行フロー
- 非同期API例（抜粋）
```http
POST /api/train            # 学習ジョブ投入（戻り: task_id）
POST /rag/upload-document  # 文書取込（バックグラウンド処理）
GET  /api/task/{id}        # ステータス照会（pending/running/completed/failed）
```
- `BackgroundTasks`で多重処理。識別子で追跡し、完了時に成果（モデル/索引/ログ）へリンク。
- 運用: `uvicorn app.main_unified:app --reload --port 8050`。

---

## 6. 品質・運用KPIと検証
- 品質: テスト成功率、静的解析逸脱ゼロ、RAG回答妥当率（>=90%）、回帰ゼロ。
- 生産性: リードタイム/変更バッチサイズ、AI主導コントリビューション比率、照査自動化率。
- SLO: API p95<800ms（キャッシュ時<500ms）、ジョブ成功率>99%、インデクシング24h内完了。
- 事実例（本リポジトリ）: 開発速度3–5倍、タスク処理時間80%削減、バグ70%削減（報告書本編より）。
- 前提チェック: `pytest -q && flake8 && black --check . && isort --check-only .`。

---

## 7. セキュリティ/ガバナンス
- 秘密管理: `.env`/`.env.example`に準拠。資格情報はコミット禁止。
- 守備範囲: データ境界（`data/`, `outputs/`）はGit対象外。個人情報はRAGに投入しない原則。
- 変更管理: Conventional Commits（`feat:`, `fix:`, `docs:`等）。PR記載: 目的/差分/テスト結果/影響範囲/ロールバック。
- 監査: ログ/成果物のハッシュ化、重要設定変更の承認履歴。

---

## 8. リスクと対策
- コンテキスト制限: Serena/Cipherで選択的永続化＋重要度優先。
- 依存関係揺らぎ: `scripts/check_rag_dependencies.py`で健全性検査、`requirements*.txt`固定。
- 並列実行の競合: タスクIDごとにワークディレクトリ分離、排他制御導入。
- ハルシネーション/誤自動化: 人間ゲート（高リスク操作）、サンドボックス実行、ドライラン必須。

---

## 9. 導入手順とSOP
- 環境構築
```bash
cd docker && docker-compose up -d --build
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```
- ローカル開発
```bash
python -m venv venv && source venv/bin/activate
pip install -e .[dev]
pytest -q
uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
```
- インシデント対応: 影響範囲評価→ロールバック手順→再現テスト→再発防止PR（原因/対策/検証）。

---

## 10. ロードマップ（抜粋）
- 近接: RAG評価自動化（難易度タグ/出典検証）、モデル選択自動化、データ同意管理。
- 中期: マルチエージェント協調、A/Bモデリング、コスト最適化ランチャ。
- 長期: 規制準拠テンプレート、自己修復オーケストレーション、オンデバイス連携。

---

## 付録A: データ/スキーマ（例）
- Task
```json
{"id":"uuid","type":"train|index|convert","priority":"high|med|low","status":"pending|running|completed|failed","artifacts":["/outputs/..."],"metrics":{"latency_ms":1234}}
```
- MCPメモリ命名例
```
project_overview_current / architecture_current / rag_system_architecture /
last_task_summary / kpi_weekly / risk_register
```
- コマンド集
```
pytest -q
black . && isort . && flake8 src tests
python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload
./scripts/docker_build_rag.sh --no-cache
```

---

### 結語
本深堀り版は、エージェント中心のタスク運用を安全・高効率・高品質に進めるための実装要点を体系化しました。Vibe Coding × MCP × RAG/学習基盤の組み合わせにより、現場で“継続的に良くなる”自動化運用を実現します。

```
