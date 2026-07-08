# ファインチューニング済みモデルの実用ガイド

## 📋 目次

1. [保存場所とファイル構成](#保存場所とファイル構成)
2. [基本的な読み込み方法](#基本的な読み込み方法)
3. [実用的な活用例](#実用的な活用例)
4. [本番運用での実装](#本番運用での実装)
5. [パフォーマンス最適化](#パフォーマンス最適化)
6. [トラブルシューティング](#トラブルシューティング)

## 📁 保存場所とファイル構成

### 保存ディレクトリ

```bash
# Dockerコンテナ内の保存場所
/workspace/lora_demo_YYYYMMDD_HHMMSS/

# ホストマシンにコピー
docker cp ai-ft-container:/workspace/lora_demo_20250626_074248 ./my_trained_model
```

### ファイル構成

```
lora_demo_20250626_074248/
├── adapter_model.safetensors    # LoRA重み（1.6MB）⭐ 最重要
├── adapter_config.json          # LoRA設定（763B）
├── tokenizer.json              # トークナイザー（3.4MB）
├── vocab.json                  # 語彙辞書（780KB）
├── merges.txt                  # BPEマージ（446KB）
├── special_tokens_map.json     # 特殊トークン（131B）
├── tokenizer_config.json       # トークナイザー設定（507B）
└── README.md                   # モデル情報（5KB）
```

### 重要ファイルの説明

| ファイル | 用途 | 重要度 |
|----------|------|--------|
| `adapter_model.safetensors` | **実際の学習結果** | ⭐⭐⭐ |
| `adapter_config.json` | LoRA設定 | ⭐⭐⭐ |
| `tokenizer.json` | テキスト処理 | ⭐⭐ |
| `vocab.json` | 語彙変換 | ⭐⭐ |
| その他 | 補助ファイル | ⭐ |

## 🔧 基本的な読み込み方法

### 最小限の読み込みコード

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_trained_model(adapter_path, base_model_name="distilgpt2"):
    """ファインチューニング済みモデルの読み込み"""
    
    # ベースモデルとトークナイザー
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # RTX A5000 x2で自動最適化
    )
    
    # LoRAアダプターの読み込み
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

# 使用例
model, tokenizer = load_trained_model("/path/to/lora_demo_20250626_074248")
```

### テキスト生成の基本

```python
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """テキスト生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用例
result = generate_text(model, tokenizer, "Hello, how are you")
print(result)
```

## 🚀 実用的な活用例

### 1. 対話システム

```python
class ChatBot:
    def __init__(self, adapter_path):
        self.model, self.tokenizer = load_trained_model(adapter_path)
        self.conversation_history = []
    
    def chat(self, user_input):
        # 会話履歴を含めたプロンプト作成
        prompt = self._build_prompt(user_input)
        
        # 応答生成
        response = generate_text(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_new_tokens=100
        )
        
        # 履歴更新
        self.conversation_history.append({
            'user': user_input,
            'bot': response
        })
        
        return response
    
    def _build_prompt(self, user_input):
        # 会話履歴を考慮したプロンプト構築
        context = "\\n".join([
            f"User: {turn['user']}\\nBot: {turn['bot']}" 
            for turn in self.conversation_history[-3:]  # 最新3ターン
        ])
        return f"{context}\\nUser: {user_input}\\nBot:"

# 使用例
bot = ChatBot("/path/to/lora_demo_20250626_074248")
response = bot.chat("こんにちは")
print(response)
```

### 2. 文書要約システム

```python
class DocumentSummarizer:
    def __init__(self, adapter_path):
        self.model, self.tokenizer = load_trained_model(adapter_path)
    
    def summarize(self, document, max_summary_length=200):
        """文書要約"""
        prompt = f"以下の文書を要約してください:\\n\\n{document}\\n\\n要約:"
        
        # 長い文書の場合は分割処理
        if len(document) > 1000:
            return self._summarize_long_document(document, max_summary_length)
        
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_summary_length
        )
    
    def _summarize_long_document(self, document, max_length):
        # 長い文書を分割して要約
        chunks = [document[i:i+800] for i in range(0, len(document), 800)]
        summaries = []
        
        for chunk in chunks:
            summary = self.summarize(chunk, max_length//len(chunks))
            summaries.append(summary)
        
        # 部分要約を統合
        final_prompt = f"以下の要約をまとめてください:\\n{' '.join(summaries)}\\n\\n最終要約:"
        return generate_text(
            self.model,
            self.tokenizer,
            final_prompt,
            max_new_tokens=max_length
        )

# 使用例
summarizer = DocumentSummarizer("/path/to/lora_demo_20250626_074248")
summary = summarizer.summarize("長い文書...")
```

### 3. Q&Aシステム

```python
class QASystem:
    def __init__(self, adapter_path, knowledge_base):
        self.model, self.tokenizer = load_trained_model(adapter_path)
        self.knowledge_base = knowledge_base  # 知識ベース
    
    def answer_question(self, question):
        """質問応答"""
        # 関連する知識を検索
        relevant_info = self._search_knowledge(question)
        
        # プロンプト構築
        prompt = f"""
知識ベース:
{relevant_info}

質問: {question}
回答:"""
        
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=150
        )
    
    def _search_knowledge(self, question):
        # 簡単な知識検索（実際はベクトル検索等を使用）
        relevant = []
        for item in self.knowledge_base:
            if any(keyword in item.lower() for keyword in question.lower().split()):
                relevant.append(item)
        return "\\n".join(relevant[:3])  # 上位3件

# 使用例
knowledge = [
    "東京は日本の首都です。",
    "富士山は日本で最も高い山です。",
    "桜は日本の国花です。"
]

qa_system = QASystem("/path/to/lora_demo_20250626_074248", knowledge)
answer = qa_system.answer_question("日本の首都はどこですか？")
```

## 🌐 本番運用での実装

### 1. Flask WebAPIサーバー

```python
from flask import Flask, request, jsonify
import threading
import time

app = Flask(__name__)

# グローバルモデル（起動時に1回読み込み）
model = None
tokenizer = None
model_lock = threading.Lock()

def initialize_model():
    """サーバー起動時にモデル初期化"""
    global model, tokenizer
    print("Loading model...")
    model, tokenizer = load_trained_model("/path/to/lora_demo_20250626_074248")
    print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/generate', methods=['POST'])
def generate_text_api():
    """テキスト生成API"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # スレッドセーフな生成
        with model_lock:
            result = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        
        return jsonify({
            'result': result,
            'prompt': prompt,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_api():
    """チャットAPI"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        # セッション管理（実際はRedis等を使用）
        chat_prompt = f"User: {message}\\nBot:"
        
        with model_lock:
            response = generate_text(
                model, 
                tokenizer, 
                chat_prompt, 
                max_new_tokens=100
            )
        
        # Botの部分のみ抽出
        bot_response = response.split("Bot:")[-1].strip()
        
        return jsonify({
            'response': bot_response,
            'session_id': session_id,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

### 2. FastAPI サーバー（非同期対応）

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uvicorn

app = FastAPI(title="Fine-tuned Model API")

# リクエストモデル
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# グローバルモデル
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """起動時のモデル読み込み"""
    global model, tokenizer
    print("Loading model...")
    model, tokenizer = load_trained_model("/path/to/lora_demo_20250626_074248")
    print("Model loaded!")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/generate")
async def generate_text_async(request: GenerateRequest):
    """非同期テキスト生成"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # CPU集約的タスクを別スレッドで実行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_text,
            model,
            tokenizer,
            request.prompt,
            request.max_tokens,
            request.temperature
        )
        
        return {"result": result, "prompt": request.prompt}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_async(request: ChatRequest):
    """非同期チャット"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        chat_prompt = f"User: {request.message}\\nBot:"
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            generate_text,
            model,
            tokenizer,
            chat_prompt,
            100,
            0.7
        )
        
        bot_response = response.split("Bot:")[-1].strip()
        
        return {
            "response": bot_response,
            "session_id": request.session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Dockerでの本番デプロイ

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu20.04

# 依存関係インストール
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# アプリケーションコピー
COPY app/ /app
COPY models/ /models

WORKDIR /app

# 環境変数
ENV MODEL_PATH=/models/lora_demo_20250626_074248
ENV PORT=8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
  CMD curl -f http://localhost:$PORT/health || exit 1

# サーバー起動
CMD ["python3", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/lora_demo_20250626_074248
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    restart: unless-stopped
```

## ⚡ パフォーマンス最適化

### 1. RTX A5000 x2での最適化

```python
def load_optimized_model(adapter_path, base_model_name="distilgpt2"):
    """RTX A5000 x2最適化済みモデル読み込み"""
    
    # メモリ効率的な設定
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True  # 高速トークナイザー使用
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # FP16で高速化
        device_map="auto",          # マルチGPU自動配置
        low_cpu_mem_usage=True,     # CPU メモリ節約
        trust_remote_code=True
    )
    
    # LoRAアダプター読み込み
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        torch_dtype=torch.float16
    )
    
    # 推論モードに設定
    model.eval()
    
    return model, tokenizer
```

### 2. バッチ処理最適化

```python
def batch_generate(model, tokenizer, prompts, batch_size=4):
    """バッチでの効率的な生成"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # バッチエンコーディング
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # デコード
        batch_results = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        results.extend(batch_results)
    
    return results
```

### 3. キャッシュ機能

```python
import functools
import hashlib

@functools.lru_cache(maxsize=1000)
def cached_generate(prompt_hash, max_tokens, temperature):
    """生成結果のキャッシュ"""
    # 実際の生成処理
    return generate_text(model, tokenizer, prompt, max_tokens, temperature)

def generate_with_cache(prompt, max_tokens=50, temperature=0.7):
    """キャッシュ付き生成"""
    # プロンプトのハッシュ化
    prompt_hash = hashlib.md5(f"{prompt}_{max_tokens}_{temperature}".encode()).hexdigest()
    return cached_generate(prompt_hash, max_tokens, temperature)
```

## 🔧 トラブルシューティング

### よくある問題と解決策

#### 1. CUDA Out of Memory

```python
# 解決策: バッチサイズを小さくする
def handle_memory_error():
    try:
        result = model.generate(...)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()  # メモリクリア
        # より小さなバッチサイズで再試行
        result = model.generate(..., batch_size=1)
    
    return result
```

#### 2. モデル読み込みエラー

```python
def robust_model_loading(adapter_path, max_retries=3):
    """ロバストなモデル読み込み"""
    for attempt in range(max_retries):
        try:
            model, tokenizer = load_trained_model(adapter_path)
            return model, tokenizer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)  # 5秒待機
```

#### 3. 推論速度の問題

```python
# 解決策: モデルのコンパイル（PyTorch 2.0+）
def optimize_model_for_inference(model):
    """推論用モデル最適化"""
    # TorchScript変換
    try:
        model = torch.jit.script(model)
    except:
        print("TorchScript conversion failed, using original model")
    
    # PyTorch 2.0のコンパイル
    try:
        model = torch.compile(model)
    except:
        print("Torch compile not available")
    
    return model
```

### ログとモニタリング

```python
import logging
import time
import psutil
import torch

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_api.log'),
            logging.StreamHandler()
        ]
    )

def monitor_performance(func):
    """パフォーマンス監視デコレータ"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        logger.info(f"Function: {func.__name__}")
        logger.info(f"Execution time: {end_time - start_time:.2f}s")
        logger.info(f"Memory usage: {(end_memory - start_memory) / 1024**2:.1f}MB")
        
        return result
    return wrapper

# 使用例
@monitor_performance
def generate_text_monitored(model, tokenizer, prompt):
    return generate_text(model, tokenizer, prompt)
```

## 📊 パフォーマンス指標

### RTX A5000 x2環境での実測値

| 項目 | 値 |
|------|---|
| モデルサイズ | 6.2MB（全ファイル） |
| LoRA重み | 1.6MB |
| 推論メモリ使用量 | 2-4GB |
| 単一生成速度 | 0.1-0.5秒/50トークン |
| バッチ生成速度 | 1-3秒/4バッチ |
| 起動時間 | 5-10秒 |

### 推奨設定

| 用途 | batch_size | max_tokens | temperature |
|------|------------|------------|-------------|
| リアルタイムチャット | 1 | 30-50 | 0.7-0.9 |
| 文書要約 | 2-4 | 100-200 | 0.5-0.7 |
| バッチ処理 | 4-8 | 50-100 | 0.6-0.8 |

## 🎯 まとめ

ファインチューニング済みモデルは以下の特徴を持ちます：

### 🚀 **優位性**
- **軽量**: LoRA重みは1.6MBのみ
- **高速**: RTX A5000 x2で最適化済み
- **柔軟**: 複数のタスクに対応可能
- **スケーラブル**: WebAPIやマイクロサービスに組み込み可能

### 💡 **実用化のポイント**
- ベースモデルは共有、LoRAのみ差し替え
- マルチGPU環境での自動最適化
- キャッシュとバッチ処理による高速化
- 本番運用でのモニタリングとログ

### 🎯 **次のステップ**
- より大きなモデル（8B-13B）での学習
- 特定ドメインデータでの専門化
- 複数のLoRAアダプターの管理
- A/Bテストによる品質評価

この実用ガイドを参考に、ファインチューニング済みモデルを効果的に活用してください。