# MoE Dataset Management API エンドポイント

## 追加日時
2024-08-22

## エンドポイント一覧

### データセット管理
```
GET /api/moe/dataset/stats/{dataset_name}
```
- **説明**: データセット統計情報の取得
- **パラメータ**: 
  - dataset_name: "civil_engineering" または "road_design"
- **レスポンス**:
  ```json
  {
    "sample_count": 100,
    "expert_distribution": "road_design: 50, structural: 30, ...",
    "last_updated": "2024/08/22 14:30",
    "file_size": 12345
  }
  ```

```
POST /api/moe/dataset/update
```
- **説明**: データセット更新（自動バックアップ付き）
- **リクエスト**: 
  - Form data: file (multipart/form-data)
  - dataset_name: "civil_engineering" または "road_design"
- **レスポンス**:
  ```json
  {
    "status": "success",
    "backup_path": "data/backups/road_design_20240822_143000.jsonl",
    "sample_count": 150,
    "validation_result": "成功",
    "invalid_lines": 0
  }
  ```

```
GET /api/moe/dataset/download/{dataset_name}
```
- **説明**: データセットのダウンロード
- **パラメータ**: 
  - dataset_name: "civil_engineering" または "road_design"
- **レスポンス**: JSONL file stream

### MoEトレーニング管理

```
POST /api/moe/training/start
```
- **説明**: MoEトレーニングの開始
- **リクエスト**:
  ```json
  {
    "training_type": "demo|full|lora|continual",
    "base_model": "cyberagent/open-calm-small",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "warmup_steps": 100,
    "save_steps": 500,
    "dataset": "road_design",
    "experts": ["road_design", "structural"]
  }
  ```

```
GET /api/moe/training/status/{task_id}
```
- **説明**: トレーニングステータスの確認

```
POST /api/moe/training/stop/{task_id}
```
- **説明**: トレーニングの停止

```
GET /api/moe/training/logs/{task_id}?tail=50
```
- **説明**: トレーニングログの取得

```
GET /api/moe/training/gpu-status
```
- **説明**: GPU状態の確認
- **レスポンス**:
  ```json
  {
    "gpus": [
      {
        "id": 0,
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used": 10240,
        "memory_total": 24576,
        "memory_percent": 41.67,
        "temperature": 65,
        "gpu_load": 85
      }
    ],
    "cpu": {
      "percent": 45.2,
      "cores": 16
    },
    "memory": {
      "total": 68719476736,
      "used": 34359738368,
      "percent": 50.0
    }
  }
  ```

```
GET /api/moe/training/history?limit=20
```
- **説明**: トレーニング履歴の取得

```
POST /api/moe/training/deploy/{task_id}
```
- **説明**: 完了したモデルのデプロイ

## アクセス方法
1. メインページ: http://localhost:8050/
2. MoE-RAGタブをクリック
3. 「MoEトレーニング管理画面を新規タブで開く」ボタンをクリック
4. 直接アクセス: http://localhost:8050/static/moe_training.html