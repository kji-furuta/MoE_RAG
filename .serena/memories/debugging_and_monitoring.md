# デバッグと監視ガイド

## ログ管理

### ログファイルの場所
- アプリケーションログ: `logs/app.log`
- 訓練ログ: `logs/training/`
- RAGシステムログ: `logs/rag/`
- Dockerログ: `docker logs ai-ft-container`

### ログレベル設定
```python
# app/main_unified.py でのログ設定
import logging
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## デバッグコマンド

### システム状態確認
```bash
# コンテナ状態
docker ps -a
docker stats ai-ft-container

# ポート使用状況
netstat -tlnp | grep 8050
lsof -i :8050

# プロセス確認
ps aux | grep python
ps aux | grep uvicorn
```

### GPU監視
```bash
# GPU使用状況
nvidia-smi
nvidia-smi -l 1  # 1秒ごとに更新

# GPU メモリ詳細
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# プロセスごとのGPU使用
nvidia-smi pmon -i 0
```

### メモリ監視
```bash
# システムメモリ
free -h
watch -n 1 free -h

# プロセスメモリ
htop
top -p $(pgrep -f uvicorn)

# Python メモリプロファイリング
python -m memory_profiler scripts/your_script.py
```

## エラー診断

### 一般的なエラーと対処法

#### 1. モジュールインポートエラー
```bash
# Python パス確認
docker exec ai-ft-container python -c "import sys; print(sys.path)"

# モジュール確認
docker exec ai-ft-container pip list | grep <module_name>

# 再インストール
docker exec ai-ft-container pip install -r requirements.txt --force-reinstall
```

#### 2. CUDA/GPU エラー
```bash
# CUDA バージョン確認
docker exec ai-ft-container nvidia-smi
docker exec ai-ft-container nvcc --version

# PyTorch CUDA 確認
docker exec ai-ft-container python -c "import torch; print(torch.cuda.is_available())"
docker exec ai-ft-container python -c "import torch; print(torch.cuda.device_count())"
```

#### 3. メモリ不足エラー
```python
# 量子化を有効化
from app.memory_optimized_loader import MemoryOptimizedModelLoader
loader = MemoryOptimizedModelLoader(
    enable_quantization=True,
    quantization_bits=4,  # 4bit or 8bit
    enable_cpu_offload=True
)
```

#### 4. ベクトルストア接続エラー
```bash
# Qdrant 状態確認
curl http://localhost:6333/collections

# 接続テスト
python -c "from qdrant_client import QdrantClient; client = QdrantClient('localhost', port=6333); print(client.get_collections())"
```

## パフォーマンス監視

### 訓練メトリクス
```python
# WandB でのトラッキング
import wandb
wandb.init(project="ai-ft-7", name="experiment-1")
wandb.log({
    "loss": loss,
    "accuracy": accuracy,
    "learning_rate": lr,
    "gpu_memory": torch.cuda.memory_allocated() / 1024**3
})
```

### RAG パフォーマンス
```python
# クエリ時間計測
import time
start = time.time()
results = rag_engine.query(query_text)
elapsed = time.time() - start
logger.info(f"Query time: {elapsed:.2f}s")
```

### API レスポンス監視
```bash
# レスポンスタイム測定
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8050/rag/health"

# curl-format.txt の内容:
time_namelookup:  %{time_namelookup}s\n
time_connect:  %{time_connect}s\n
time_appconnect:  %{time_appconnect}s\n
time_pretransfer:  %{time_pretransfer}s\n
time_redirect:  %{time_redirect}s\n
time_starttransfer:  %{time_starttransfer}s\n
time_total:  %{time_total}s\n
```

## リアルタイムモニタリング

### WebSocket ログストリーミング
```javascript
// 訓練ログのリアルタイム表示
const ws = new WebSocket('ws://localhost:8050/ws/training/task-123');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Epoch: ${data.epoch}, Loss: ${data.loss}`);
};
```

### Grafana/Prometheus 統合
```yaml
# docker-compose.yml に追加
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
```

## デバッグツール

### Python デバッガー
```python
# コード内でブレークポイント
import pdb; pdb.set_trace()

# IPython デバッガー
from IPython import embed; embed()

# VS Code リモートデバッグ
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### プロファイリング
```bash
# CPU プロファイリング
python -m cProfile -o profile.stats scripts/train.py
python -m pstats profile.stats

# メモリプロファイリング
mprof run scripts/train.py
mprof plot

# ライン単位のプロファイリング
kernprof -l -v scripts/train.py
```

## トラブルシューティングチェックリスト

### 起動時の確認事項
1. ✅ Docker コンテナが起動しているか
2. ✅ ポート 8050 が利用可能か
3. ✅ 必要な環境変数が設定されているか
4. ✅ GPU が認識されているか
5. ✅ ログファイルにエラーがないか

### 訓練時の確認事項
1. ✅ データセットが正しくロードされているか
2. ✅ モデルがGPUに載っているか
3. ✅ バッチサイズが適切か
4. ✅ 学習率が適切か
5. ✅ チェックポイントが保存されているか

### RAG使用時の確認事項
1. ✅ Qdrant が起動しているか
2. ✅ ドキュメントがインデックスされているか
3. ✅ 埋め込みモデルがロードされているか
4. ✅ 検索結果が返されているか
5. ✅ リランキングが動作しているか

## 緊急時の対処

### システムリセット
```bash
# 全コンテナ停止
docker-compose down

# ボリューム含めて削除
docker-compose down -v

# キャッシュクリア
docker system prune -a

# 再構築
./scripts/docker_build_rag.sh --no-cache
```

### プロセス強制終了
```bash
# Python プロセス終了
pkill -f python
pkill -f uvicorn

# GPU プロセスリセット
nvidia-smi --gpu-reset
```