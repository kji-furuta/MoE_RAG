# DPO (Direct Preference Optimization) 学習システム

## 概要
DPOは、強化学習（RLHF）を使わずに人間のフィードバックからモデルを直接学習させる手法。
MoE_RAGプロジェクトでは、統合WebインターフェースからDPO学習を実行可能。

## アーキテクチャ

### 主要コンポーネント
1. **Preference収集UI**: `templates/dpo.html`
2. **DPO Router**: `app/dpo/preference_ui.py`
3. **学習エンジン**: `src/training/dpo_trainer.py` (TRL DPOTrainer使用)
4. **統合API**: `app/main_unified.py` - `/api/train` エンドポイントで`training_method: "dpo"`

### データフロー
```
1. ユーザーがPreferenceデータ収集 (UI or API)
   → data/dpo/preference_dataset.jsonl に保存

2. DPO学習実行 (UI or API)
   → /api/train エンドポイントに POST
   → training_method: "dpo" を指定

3. バックグラウンド学習
   → LoRA + DPO統合トレーニング
   → Fisher情報行列によるパラメータ重要度計算

4. 学習済みモデル保存
   → outputs/dpo_adapter/
```

## ファイル構成

### UI関連
- **`templates/dpo.html`**: DPO Preference収集・学習実行UI
  - 行1-70: 統計表示
  - 行71-115: Preferenceデータ収集フォーム
  - 行116-130: データセット一覧
  - 行131-146: ファイルアップロード
  - 行148-242: DPO学習実行セクション（NEW）
  - 行379-482: JavaScript学習ロジック

- **`templates/base.html`**: ナビゲーション
  - 行379-383: DPO Preferenceリンク

### バックエンド
- **`app/dpo/preference_ui.py`**: DPO API Router
  - `/api/dpo/stats`: 統計情報取得
  - `/api/dpo/add-preference`: Preferenceデータ追加
  - `/api/dpo/upload-dataset`: JSONLファイルアップロード
  - `/api/dpo/datasets`: データセット一覧取得
  - `/api/dpo/delete-dataset`: データセット削除

- **`app/main_unified.py`**: 統合APIサーバー
  - 行543-546: `/dpo` UIルート
  - 行175-176: DPO Router組み込み
  - `/api/train`エンドポイント: `training_method: "dpo"`でDPO学習実行

### 学習ロジック
- **`src/training/dpo_trainer.py`**: DPOTrainer実装
  - TRL (Transformer Reinforcement Learning) ライブラリ使用
  - `DPOTrainer`クラスでpreference学習実行

## データ形式

### Preferenceデータ (JSONL)
```json
{
  "prompt": "設計速度80km/hの道路の最小曲線半径は？",
  "chosen": "設計速度80km/hの場合、最小曲線半径は280mです。道路構造令第15条に基づきます。",
  "rejected": "だいたい200mくらいです。",
  "margin": 2.0
}
```

### 学習リクエスト (API)
```json
{
  "model_name": "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
  "training_method": "dpo",
  "training_data": ["data/dpo/preference_dataset.jsonl"],
  "lora_config": {
    "r": 64,
    "lora_alpha": 128,
    "dropout": 0.05
  },
  "training_config": {
    "beta": 0.1,
    "num_epochs": 1,
    "learning_rate": 5e-6,
    "max_prompt_length": 1024,
    "max_length": 2048
  }
}
```

## ハイパーパラメータ

| パラメータ | 推奨値 | 説明 |
|----------|--------|------|
| LoRA r | 64 | LoRAのランク（高いほど表現力↑、メモリ↑） |
| LoRA Alpha | 128 | スケーリング係数（通常2×r） |
| DPO Beta | 0.1 | 選好の強さ（低い→積極的、高い→保守的） |
| 学習率 | 5e-6 | 学習の速度（大きいほど速いが不安定） |
| エポック数 | 1 | データセット全体を学習する回数 |
| Max Length | 2048 | 最大トークン長（長いほどメモリ使用↑） |

## UI操作フロー

### データ収集
1. http://localhost:8050/dpo にアクセス
2. プロンプト入力
3. 好ましい回答（Chosen）入力
4. 好ましくない回答（Rejected）入力
5. 「Preferenceを追加」クリック
6. `data/dpo/preference_dataset.jsonl`に自動保存

### 学習実行
1. 同じページの「🚀 DPO学習を実行」セクション
2. ベースモデル選択（DeepSeek-R1-Distill-Qwen-32B-Japanese）
3. ハイパーパラメータ調整
4. 「DPO学習を開始」クリック
5. リアルタイム進捗監視（3秒ごとポーリング）

## JavaScript実装詳細

### 学習フォーム送信（templates/dpo.html 行383-431）
```javascript
document.getElementById('dpoTrainingForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        model_name: document.getElementById('trainingModelName').value,
        training_method: "dpo",
        training_data: [`data/dpo/${document.getElementById('datasetName').value}`],
        lora_config: { r: ..., lora_alpha: ..., dropout: 0.05 },
        training_config: { beta: ..., num_epochs: ..., learning_rate: ..., max_length: ... }
    };
    
    const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    
    if (response.ok) {
        const result = await response.json();
        currentTaskId = result.task_id;
        startStatusCheck(); // 3秒ごとにステータス確認
    }
});
```

### 進捗監視（templates/dpo.html 行444-482）
```javascript
async function checkTrainingStatus() {
    const response = await fetch(`/api/training-status/${currentTaskId}`);
    const data = await response.json();
    
    // 進捗バー更新
    document.getElementById('trainingProgress').style.width = `${data.progress}%`;
    document.getElementById('trainingMessage').textContent = data.message;
    
    // 完了・失敗処理
    if (data.status === 'completed') {
        clearInterval(statusCheckInterval);
        // 成功アラート表示
    } else if (data.status === 'failed') {
        clearInterval(statusCheckInterval);
        // 失敗アラート表示
    }
}
```

## API エンドポイント

### DPO専用エンドポイント
- `GET /api/dpo/stats`: 統計情報
- `POST /api/dpo/add-preference`: Preferenceデータ追加
- `POST /api/dpo/upload-dataset`: JSONLアップロード
- `GET /api/dpo/datasets`: データセット一覧
- `DELETE /api/dpo/delete-dataset/{filename}`: データセット削除

### 学習エンドポイント
- `POST /api/train`: DPO学習実行（`training_method: "dpo"`）
- `GET /api/training-status/{task_id}`: 学習進捗確認

## 統合機能

### RAGシステムとの統合
- DPO学習済みモデルは`outputs/dpo_adapter/`に保存
- RAGシステムでモデル選択可能
- Ollama形式に変換してRAGで利用可能

### 継続学習との統合
- DPO学習済みモデルを継続学習のベースモデルとして使用可能
- EWC (Elastic Weight Consolidation) と組み合わせて破滅的忘却を防止

## 技術的特徴

### TRL (Transformer Reinforcement Learning)
- Hugging Faceの強化学習ライブラリ
- `DPOTrainer`クラスによる効率的な実装
- LoRAとの統合サポート

### メモリ最適化
- LoRA（Low-Rank Adaptation）による省メモリ学習
- 4bit量子化オプション（QLoRA）
- GPU使用率90-95%を目標

### リアルタイム監視
- WebSocket不使用のポーリング方式（3秒間隔）
- 進捗バーとメッセージの動的更新
- タスクID管理による複数学習の追跡

## トラブルシューティング

### データが保存されない
- `data/dpo/`ディレクトリの存在確認
- ファイル権限の確認
- API呼び出しのレスポンスログ確認

### 学習が開始しない
- ベースモデルのダウンロード状況確認
- GPU メモリ確認（`nvidia-smi`）
- Dockerコンテナログ確認（`docker logs ai-ft-container`）

### 進捗が更新されない
- ブラウザのコンソールログ確認
- `/api/training-status/{task_id}`のレスポンス確認
- JavaScriptポーリング処理の動作確認

## 関連ドキュメント
- README.md: 行205-316（DPO学習詳細）
- README.md: 行741-757（Web操作マニュアル）
- README.md: 行685-702（統合機能一覧）
