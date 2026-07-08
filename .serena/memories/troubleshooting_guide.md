# トラブルシューティングガイド

## 🚨 よくある問題と解決方法

### 1. GPU関連の問題

#### CUDA out of memory エラー
```bash
# 症状
RuntimeError: CUDA out of memory. Tried to allocate X GB

# 解決方法
# 1. GPUメモリをクリア
docker exec ai-ft-container python -c "import torch; torch.cuda.empty_cache()"

# 2. 他のプロセスを確認して終了
nvidia-smi
kill -9 [PID]

# 3. バッチサイズを減らす
# configs/model_config.yaml で batch_size を調整

# 4. gradient_checkpointing を有効化
# training_args に gradient_checkpointing=True を追加

# 5. 量子化を使用
# QLoRA (4bit) または 8bit 量子化を使用
```

#### GPUが認識されない
```bash
# 確認コマンド
docker exec ai-ft-container nvidia-smi

# 解決方法
# 1. Docker の GPU サポート確認
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# 2. nvidia-container-runtime のインストール
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# 3. Docker daemon 再起動
sudo systemctl restart docker
```

### 2. メモリ不足

#### システムメモリ不足
```bash
# 症状
MemoryError または Killed プロセス

# 解決方法
# 1. スワップ追加
sudo dd if=/dev/zero of=/swapfile bs=1G count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. Docker メモリ制限の確認
docker update ai-ft-container --memory="32g" --memory-swap="-1"

# 3. 不要なキャッシュクリア
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### 3. モジュールインポートエラー

#### ModuleNotFoundError
```bash
# 症状
ModuleNotFoundError: No module named 'XXX'

# 解決方法
# 1. パスの確認
docker exec ai-ft-container python -c "import sys; print('\n'.join(sys.path))"

# 2. 必要なパスを追加
docker exec ai-ft-container bash -c "export PYTHONPATH=/workspace:$PYTHONPATH"

# 3. 依存関係の再インストール
docker exec ai-ft-container pip install -r requirements.txt --force-reinstall

# 4. キャッシュクリア
docker exec ai-ft-container find . -type d -name __pycache__ -exec rm -r {} +
```

### 4. ポート競合

#### Address already in use
```bash
# 症状
[Errno 98] Address already in use

# 解決方法
# 1. 使用中のプロセスを特定
sudo lsof -i :8050
sudo netstat -tlnp | grep 8050

# 2. プロセスを終了
sudo fuser -k 8050/tcp

# 3. Docker コンテナの確認
docker ps | grep 8050
docker stop [CONTAINER_ID]
```

### 5. Qdrant接続エラー

#### Connection refused to Qdrant
```bash
# 症状
Failed to connect to Qdrant at localhost:6333

# 解決方法
# 1. Qdrant コンテナの状態確認
docker ps | grep qdrant
docker logs qdrant-container

# 2. Qdrant 再起動
docker restart qdrant-container

# 3. ネットワーク確認
docker network ls
docker network inspect ai-ft-network

# 4. ヘルスチェック
curl http://localhost:6333/collections
```

### 6. モデル読み込みエラー

#### HuggingFace モデルダウンロードエラー
```bash
# 症状
OSError: Can't load model from 'model_name'

# 解決方法
# 1. HF_TOKEN の確認
docker exec ai-ft-container env | grep HF_TOKEN

# 2. キャッシュディレクトリの確認
docker exec ai-ft-container ls -la /workspace/hf_cache

# 3. 手動ダウンロード
docker exec ai-ft-container python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('model_name', cache_dir='/workspace/hf_cache')
"

# 4. オフラインモードの設定
export TRANSFORMERS_OFFLINE=1
```

### 7. 訓練が進まない

#### Loss が下がらない
```bash
# 確認事項
# 1. 学習率の確認
grep learning_rate configs/*.yaml

# 2. データの確認
head -n 5 data/train.jsonl

# 3. モデルの勾配確認
# training_utils.py に以下を追加:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")

# 解決方法
# - 学習率を調整 (1e-5 〜 1e-3)
# - warmup_steps を増やす
# - データの品質を確認
```

### 8. Docker ビルドエラー

#### イメージビルド失敗
```bash
# 症状
docker build failed

# 解決方法
# 1. キャッシュなしでビルド
docker build --no-cache -t ai-ft-image .

# 2. ビルドログの詳細確認
docker build --progress=plain -t ai-ft-image .

# 3. Docker システムクリーンアップ
docker system prune -a --volumes

# 4. ディスク容量確認
df -h
docker system df
```

### 9. WebSocket 接続エラー

#### WebSocket connection failed
```bash
# 症状
WebSocket connection to ws://localhost:8050/ws/... failed

# 解決方法
# 1. nginx/プロキシ設定確認
# WebSocket のアップグレードヘッダーを許可

# 2. CORS 設定確認
# main_unified.py で origins を確認

# 3. ファイアウォール確認
sudo ufw status
sudo ufw allow 8050
```

### 10. ログ関連の問題

#### ログが出力されない
```bash
# 解決方法
# 1. ログレベルの確認
docker exec ai-ft-container python -c "
import logging
print(logging.getLogger().level)
"

# 2. ログ設定の確認
grep -r "logging" src/utils/logger.py

# 3. 標準エラー出力の確認
docker logs ai-ft-container 2>&1

# 4. ログファイルの権限確認
docker exec ai-ft-container ls -la logs/
```

## 🔍 デバッグテクニック

### 1. インタラクティブデバッグ
```python
# コードに追加
import pdb; pdb.set_trace()

# または
import ipdb; ipdb.set_trace()  # より高機能
```

### 2. リモートデバッグ（VSCode）
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/workspace"
                }
            ]
        }
    ]
}
```

### 3. プロファイリング
```python
# メモリプロファイリング
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 処理

# 時間プロファイリング
import cProfile
cProfile.run('train_model()')
```

## 📞 サポート情報

### ログ収集スクリプト
```bash
#!/bin/bash
# collect_debug_info.sh

echo "=== System Info ===" > debug_report.txt
uname -a >> debug_report.txt
echo -e "\n=== Docker Info ===" >> debug_report.txt
docker version >> debug_report.txt
echo -e "\n=== GPU Info ===" >> debug_report.txt
nvidia-smi >> debug_report.txt
echo -e "\n=== Container Logs ===" >> debug_report.txt
docker logs ai-ft-container --tail 1000 >> debug_report.txt
echo -e "\n=== Python Packages ===" >> debug_report.txt
docker exec ai-ft-container pip freeze >> debug_report.txt

echo "Debug report saved to debug_report.txt"
```