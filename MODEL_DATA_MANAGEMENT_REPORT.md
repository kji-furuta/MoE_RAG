# モデル・データ管理レポート

## 📊 現在のモデル管理状況

### 1. ベースモデル ダウンロード状態

#### ✅ ダウンロード済みモデル（ホストマシン）
| モデル名 | パス | サイズ | 状態 |
|----------|------|--------|------|
| **cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese** | `~/.cache/huggingface/hub/` | 62GB | ✅ 完全ダウンロード済み |
| **cyberagent/calm3-22b-chat** | `~/.cache/huggingface/hub/` | 42GB | ✅ 完全ダウンロード済み |
| **Qwen/Qwen2.5-32B-Instruct** | `~/.cache/huggingface/hub/` | 62GB | ✅ 完全ダウンロード済み |
| **intfloat/multilingual-e5-large** | `~/.cache/huggingface/hub/` | 2.2GB | ✅ RAG埋め込み用 |
| **cross-encoder/ms-marco-MiniLM-L-12-v2** | `~/.cache/huggingface/hub/` | 129MB | ✅ リランキング用 |

#### ⚠️ 部分ダウンロード/未完了
| モデル名 | パス | サイズ | 状態 |
|----------|------|--------|------|
| **EleutherAI/gpt-neox-20b** | `~/.cache/huggingface/hub/` | 36KB | ❌ メタデータのみ |
| **microsoft/Orca-2-13b** | `~/.cache/huggingface/hub/` | 20KB | ❌ メタデータのみ |

#### 📝 設定済み但し未ダウンロード
`config/model_config.yaml` に定義されているが未ダウンロード：
- stabilityai/japanese-stablelm-3b-4e1t-instruct (3B)
- rinna/japanese-gpt-neox-3.6b (3.6B)
- line-corporation/japanese-large-lm-3.6b (3.6B)
- elyza/ELYZA-japanese-Llama-3-8B (8B)
- meta-llama/Llama-3.1-17B-Instruct (17B) ※要認証
- microsoft/Phi-3.5-17B-Instruct (17B)
- Qwen/Qwen2.5-17B-Instruct (17B)
- meta-llama/Llama-3.1-32B-Instruct (32B) ※要認証
- microsoft/Phi-3.5-32B-Instruct (32B)
- meta-llama/Llama-3.1-70B-Instruct (70B) ※要認証

### 2. Dockerコンテナ内のモデル状態

#### ⚠️ コンテナ内キャッシュ状況
| モデル | 状態 | 対策 |
|--------|------|------|
| **大規模モデル（22B/32B）** | ❌ コンテナ内未同期 | ボリュームマウント必要 |
| **埋め込みモデル** | ✅ コンテナ内にコピー済み | 正常動作中 |
| **リランキングモデル** | ✅ コンテナ内にコピー済み | 正常動作中 |

#### 🔧 Docker設定の問題点
```yaml
# 現在の設定（docker-compose.yml）
volumes:
  - ./:/workspace  # プロジェクトディレクトリのみ
  # HuggingFaceキャッシュがマウントされていない！
```

**推奨修正：**
```yaml
volumes:
  - ./:/workspace
  - ~/.cache/huggingface:/root/.cache/huggingface  # キャッシュ共有
  - ./outputs:/workspace/outputs  # モデル出力
```

### 3. ファインチューニング済みモデル

#### 📦 LoRAモデル（outputs/）
| タイプ | 数量 | 場所 |
|--------|------|------|
| 継続学習タスク | 30+ | `outputs/continual_task_*` |
| LoRAアダプター | 10+ | `outputs/lora_*` |
| EWCデータ | 複数 | `outputs/ewc_data/` |

#### 🎯 主要な学習済みモデル
- **最新モデル**: `continual_task_100_20250928_043851`
- **ベースモデル**: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
- **タスク数**: 100タスク完了

### 4. ストレージ使用状況

#### 💾 ディスク使用量
```
総容量: 1007GB
使用済: 823GB (87%)
空き: 133GB (13%)
```

#### 📊 モデル別ストレージ消費
| カテゴリ | サイズ | 割合 |
|----------|--------|------|
| ダウンロード済みベースモデル | ~168GB | 20% |
| ファインチューニング済みモデル | ~50GB | 6% |
| RAG埋め込みデータ | ~5GB | <1% |
| その他（データ、ログ等） | ~600GB | 73% |

### 5. データ管理の課題と対策

#### 🔴 重要課題

1. **Dockerコンテナとホストのモデル非同期**
   - 問題: 大規模モデルがコンテナからアクセスできない
   - 対策: HuggingFaceキャッシュをボリュームマウント

2. **ストレージ逼迫（87%使用）**
   - 問題: 空き容量が少ない
   - 対策: 不要な古いモデル・ログの削除

3. **重複モデル**
   - 問題: 同じモデルが複数場所に存在
   - 対策: シンボリックリンクによる統合

#### 🟡 改善推奨事項

1. **モデル管理スクリプト作成**
```python
# scripts/manage_models.py
import os
import shutil
from pathlib import Path

class ModelManager:
    def __init__(self):
        self.cache_dir = Path.home() / ".cache/huggingface/hub"
        self.outputs_dir = Path("outputs")

    def cleanup_old_models(self, days=30):
        """古いモデルを削除"""
        # 30日以上前のモデルを削除
        pass

    def list_all_models(self):
        """全モデルのリスト取得"""
        models = {
            "base_models": self.scan_base_models(),
            "finetuned": self.scan_finetuned_models(),
            "disk_usage": self.calculate_usage()
        }
        return models

    def sync_to_docker(self):
        """Dockerコンテナとモデルを同期"""
        # ボリュームマウント設定の確認と修正
        pass
```

2. **自動クリーンアップCron設定**
```bash
# 毎週日曜日に古いモデルをクリーンアップ
0 2 * * 0 python /workspace/scripts/cleanup_old_models.py
```

3. **モデルレジストリ作成**
```json
{
  "models": {
    "base": {
      "deepseek-r1-32b": {
        "path": "/root/.cache/huggingface/hub/models--cyberagent--DeepSeek-R1-Distill-Qwen-32B-Japanese",
        "size": "62GB",
        "downloaded": true,
        "in_docker": false
      }
    },
    "finetuned": {
      "task_100_latest": {
        "path": "/workspace/outputs/continual_task_100_20250928_043851",
        "base_model": "deepseek-r1-32b",
        "created": "2025-09-28",
        "metrics": {
          "loss": 0.23,
          "accuracy": 0.92
        }
      }
    }
  }
}
```

### 6. 即時対応アクションプラン

#### 📋 優先度高
1. **Docker Compose更新**
   ```bash
   # HuggingFaceキャッシュをマウント
   vim docker/docker-compose.yml
   # volumes セクションに追加
   ```

2. **ストレージ解放**
   ```bash
   # 古い継続学習モデルの削除
   find outputs/continual_task_* -mtime +30 -type d -exec rm -rf {} \;

   # 不要なログファイルの削除
   find . -name "*.log" -mtime +7 -delete
   ```

3. **モデル一覧生成**
   ```bash
   # 現在のモデル状態をJSON出力
   python scripts/generate_model_inventory.py > model_inventory.json
   ```

#### 📋 優先度中
4. **モデル管理UI実装**（前述の提案参照）
5. **自動同期スクリプト作成**
6. **バックアップ戦略策定**

### 7. ベストプラクティス

#### ✅ 推奨事項
- モデルは常にHuggingFaceキャッシュに保存
- ファインチューニング結果は`outputs/`に整理
- 定期的なクリーンアップスクリプト実行
- モデルメタデータのJSON管理
- Dockerコンテナとホストのキャッシュ共有

#### ❌ 避けるべき事項
- モデルの重複ダウンロード
- キャッシュディレクトリの直接削除
- バージョン管理なしのモデル上書き
- メタデータなしのモデル保存

## 📊 サマリー

### 現状
- **ダウンロード済み**: 3つの大規模モデル（166GB）
- **Dockerアクセス**: 埋め込みモデルのみ
- **ストレージ**: 87%使用（要クリーンアップ）

### 必要アクション
1. ✅ Docker Compose修正（HuggingFaceキャッシュマウント）
2. ✅ ストレージクリーンアップ（古いモデル削除）
3. ✅ モデルインベントリ作成
4. ⏳ 管理UI実装
5. ⏳ 自動化スクリプト整備

---

*レポート作成日: 2025年9月28日*
*次回レビュー予定: 2025年10月5日*