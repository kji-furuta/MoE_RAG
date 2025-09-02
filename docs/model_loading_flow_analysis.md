# モデル設定実行時の動作フロー分析

## 質問：モデル設定を実行すると「LoRA量子化実行」は行われるか？

### 回答：**いいえ、自動的には行われません**

## 現在のモデルロードフロー

### 1. RAGシステム起動時の動作

```python
# src/rag/core/query_engine.py - LLMGenerator.__init__

if config.llm.use_finetuned == true:
    ├─ GPUメモリチェック (30GB以上必要)
    │   ├─ メモリ十分 → _load_model()実行
    │   │   └─ AutoModelForCausalLM.from_pretrained()
    │   │       └─ そのままロード（量子化なし）
    │   └─ メモリ不足 → Ollamaフォールバック
    │       └─ llama3.2:3b を使用
    └─ 結果：量子化は実行されない
```

### 2. 実際のコード動作

#### 2.1 メモリチェック
```python
def _check_memory_for_model(self) -> bool:
    # 22Bモデルには最低30GBが必要
    required_memory = 30
    return max_free_memory >= required_memory
```

#### 2.2 モデルロード処理
```python
def _load_model(self):
    # 通常のモデルロード（量子化なし）
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs  # 量子化設定は含まれない
    )
```

#### 2.3 量子化オプション（未使用）
```python
# _get_optimized_model_kwargs内に量子化コードは存在するが
# load_in_8bitフラグが設定ファイルで制御されているだけ
if llm_config.load_in_8bit:
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,  # 実際は4-bit
        ...
    )
```

## 問題点

### 1. LoRAアダプタの扱い
- **現状**: LoRAアダプタパスを直接AutoModelForCausalLMで読み込もうとする
- **問題**: LoRAアダプタは完全モデルではないため、エラーになる
- **量子化**: 実行されない

### 2. 量子化タイミング
- **現状**: モデルロード時に動的量子化のオプションのみ
- **問題**: LoRA + ベースモデルのマージが必要
- **解決**: マージ→量子化→保存の事前処理が必要

### 3. メモリ制約
- **32Bモデル**: 64GB必要（FP16）
- **GPUメモリ**: 36GB（不足）
- **結果**: 常にOllamaフォールバック

## 必要な実装

### 方法1: 事前量子化（推奨）

```bash
# 1. 事前にLoRAを量子化
python scripts/quantize_finetuned_for_ollama.py \
    --lora-path outputs/lora_20250830_223432

# 2. Ollamaに登録
ollama create deepseek-r1-q4 -f Modelfile

# 3. RAG設定更新
use_ollama_fallback: true
ollama_model: deepseek-r1-q4
```

### 方法2: 自動量子化機能の追加

```python
class LLMGenerator:
    def _load_model(self):
        if self._is_lora_adapter(model_path):
            # LoRAアダプタを検出
            if self.auto_quantize:
                # 自動的に量子化実行
                quantized_path = self._quantize_lora(model_path)
                self._register_to_ollama(quantized_path)
                self._enable_ollama_fallback()
            else:
                # エラー：LoRAは直接ロード不可
                raise ValueError("LoRA adapter cannot be loaded directly")
```

### 方法3: Web UIからの量子化実行

```python
@app.post("/api/quantize-model")
async def quantize_model(request: QuantizeRequest):
    # Web UIから量子化を実行
    quantizer = FinetunedModelQuantizer(
        lora_path=request.lora_path,
        quantization_level=request.level
    )
    result = quantizer.run()
    
    # 自動的にRAG設定を更新
    update_rag_config(model_name=result.ollama_model)
    
    return {"status": "success", "model": result.ollama_model}
```

## 結論

### 現在の動作
1. `use_finetuned: true`設定時
2. モデルパスからそのままロード試行
3. メモリ不足でOllamaフォールバック
4. **量子化は一切実行されない**

### 必要なアクション
1. **手動で量子化スクリプト実行**（即座に可能）
2. **自動量子化機能の実装**（開発必要）
3. **Web UIへの統合**（ユーザビリティ向上）

### 推奨手順
```bash
# 今すぐ実行可能
cd /workspace
python scripts/quantize_finetuned_for_ollama.py \
    --lora-path outputs/lora_20250830_223432 \
    --quantization Q4_K_M

# Ollama登録
ollama create rag-deepseek-r1 -f outputs/quantized_finetuned/Modelfile

# RAG設定更新
echo "ollama_model: rag-deepseek-r1" >> src/rag/config/rag_config.yaml
```

これにより、ファインチューニング済みモデルが量子化されてRAGシステムで使用可能になります。