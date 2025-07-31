# 大規模モデル（17B・22B・32B・70B）セットアップガイド

## 概要

Webインターフェースで17B、22B、32B、70Bの大規模モデルが表示されない場合の手動セットアップ方法を説明します。

### 🆕 新規追加モデル
- **CyberAgent CALM3-22B**: 日本語特化の大規模モデル

## 手動でのモデルダウンロード手順

### 1. Hugging Faceトークンの準備（必要な場合）

一部のモデルは認証が必要です。以下の手順でトークンを設定してください：

```bash
# Hugging Faceにログイン
huggingface-cli login
# トークンを入力
```

または環境変数で設定：
```bash
export HF_TOKEN="your_huggingface_token"
```

### 2. モデルのダウンロード

#### 方法1: Pythonスクリプトでダウンロード

**実行方法**:

```bash
# Dockerコンテナに入る
docker exec -it ai-ft-container bash

# /workspaceディレクトリに移動
cd /workspace

# Pythonスクリプトを作成
cat > download_models.py << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 22Bモデルの例（CALM3）
model_name = "cyberagent/calm3-22b-chat"  # または "Qwen/Qwen2.5-17B-Instruct"
print(f"Downloading {model_name}...")

try:
    # トークナイザーのダウンロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer downloaded")
    
    # モデルのダウンロード（キャッシュに保存）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"  # ダウンロードのみなのでCPU指定
    )
    print("✓ Model downloaded successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

# スクリプトを実行
python3 download_models.py
```

#### 方法2: Hugging Face CLIでダウンロード

**実行場所**: Dockerコンテナ内の `/workspace` ディレクトリ

```bash
# まず、Dockerコンテナに入る
docker exec -it ai-ft-container bash

# /workspace ディレクトリに移動
cd /workspace

# modelsディレクトリを作成（存在しない場合）
mkdir -p models

# 17Bモデル
huggingface-cli download Qwen/Qwen2.5-17B-Instruct --local-dir ./models/Qwen2.5-17B

# 22Bモデル（CALM3） 🆕
huggingface-cli download cyberagent/calm3-22b-chat --local-dir ./models/calm3-22b

# 32Bモデル
huggingface-cli download cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B --local-dir ./models/calm3-32B

# 70Bモデル（オプション）
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ./models/Llama-3.1-70B
```

**注意**: 
- ダウンロードしたモデルは自動的にHugging Faceのキャッシュにも保存されます
- `--local-dir`で指定したディレクトリは参照用のコピーです
- 実際の使用時はキャッシュから自動的に読み込まれます

### 3. ディスク容量の確認

大規模モデルには十分なディスク容量が必要です：

- 17Bモデル: 約35GB
- 22Bモデル: 約44GB
- 32Bモデル: 約65GB
- 70Bモデル: 約140GB

```bash
# 空き容量確認
df -h /workspace
```

### 4. モデルの事前ロード（オプション）

システム起動時にモデルをキャッシュにロードしておくことで、初回使用時の待ち時間を短縮できます：

```python
# preload_models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

models_to_preload = [
    "Qwen/Qwen2.5-17B-Instruct",
    "cyberagent/calm3-DeepSeek-R1-Distill-Qwen-32B"
]

for model_name in models_to_preload:
    print(f"Preloading {model_name}...")
    try:
        # キャッシュにダウンロード
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        print(f"✓ {model_name} preloaded successfully")
    except Exception as e:
        print(f"✗ Failed to preload {model_name}: {e}")
```

### 5. Webインターフェースの再起動

モデルがダウンロードされたら、Webインターフェースを再起動します：

```bash
# Dockerコンテナを再起動
docker-compose restart

# または、アプリケーションのみ再起動
docker exec ai-ft-container supervisorctl restart web
```

## トラブルシューティング

### 問題: モデルがリストに表示されない

1. **キャッシュをクリア**
   ```bash
   docker exec ai-ft-container rm -rf /tmp/cache/*
   ```

2. **アプリケーションログを確認**
   ```bash
   docker exec ai-ft-container tail -f /workspace/logs/web.log
   ```

### 問題: メモリ不足エラー

1. **4bit量子化を使用**
   - WebインターフェースでQLoRAを選択
   - または手動で`load_in_4bit=True`を指定

2. **シーケンス長を短縮**
   - トレーニング設定で最大シーケンス長を256に設定

### 問題: ダウンロードが遅い

1. **ミラーサイトを使用**
   ```bash
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

2. **部分的なダウンロード**
   - 必要なファイルのみダウンロード（safetensorsファイルなど）

## 推奨事項

1. **初回セットアップ時**
   - まず小さいモデル（3B）で動作確認
   - 段階的に大きいモデルを追加

2. **プロダクション環境**
   - モデルは事前にダウンロード・検証
   - 専用のモデルストレージを用意

3. **開発環境**
   - シンボリックリンクでキャッシュを共有
   ```bash
   ln -s /shared/models ~/.cache/huggingface
   ```

## サポート

問題が解決しない場合は、以下の情報を含めてサポートにお問い合わせください：

- 使用しているGPUの種類とメモリ容量
- エラーメッセージの全文
- `/workspace/logs/`内のログファイル