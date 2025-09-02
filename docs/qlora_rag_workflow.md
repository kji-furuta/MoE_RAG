# QLoRA + RAG統合ワークフロー

## 概要
このドキュメントは、QLoRAでファインチューニングしたモデルをRAGシステムで使用するための完全なワークフローを説明します。

## 問題の背景
DeepSeek-R1-Distill-Qwen-32B（32Bパラメータ）を2x RTX A5000（各24GB）環境で動作させるには、メモリ制約により以下の工夫が必要です：

### メモリ要件
- **FP16**: 32B × 2バイト = 64GB（利用不可）
- **4bit量子化**: 32B × 0.5バイト = 16GB（利用可能）

## システムアーキテクチャ

```
[ファインチューニング] → [マージ] → [GGUF変換] → [量子化] → [Ollama] → [RAG]
     (QLoRA 4bit)        (FP16)      (FP16)       (Q4_K_M)
```

## 詳細ワークフロー

### 1. ファインチューニング（QLoRA）
**場所**: ファインチューニングページ  
**設定**: LoRAファインチューニング（自動的にQLoRA 4bitが使用される）

```python
# src/training/lora_finetuning.py
if self.lora_config.use_qlora:  # デフォルトでTrue
    # 4bit量子化でベースモデルをロード
    model = load_in_4bit(base_model)
    # LoRAアダプターのみ学習
```

**出力**:
- `/workspace/outputs/lora_*/final_lora_model/` - LoRAアダプター
- マージモデルは作成されない（QLoRAのため）

### 2. 量子化ボタン（RAGシステム）
**場所**: RAGシステムページ → 検索モデルの設定 → 量子化開始

**処理内容** (`scripts/qlora_to_ollama.py`):

#### Step 1: LoRAアダプター検出
```python
# 最新のLoRAアダプターを自動検出
lora_path = find_lora_adapter()
# 例: /workspace/outputs/lora_20250830_223432/final_lora_model
```

#### Step 2: FP16マージ
```python
# ベースモデル + LoRAアダプター → FP16マージモデル
merge_lora_to_fp16(lora_path, base_model, output_path)
# 出力: /workspace/outputs/ollama_conversion/merged_model_fp16/
```

#### Step 3: GGUF変換
```python
# FP16モデル → GGUF形式
convert_to_gguf(fp16_model_path, gguf_path)
# 出力: /workspace/outputs/ollama_conversion/model-f16.gguf
```

#### Step 4: 量子化
```python
# GGUF F16 → GGUF Q4_K_M（4bit量子化）
quantize_gguf(gguf_f16_path, gguf_q4_path)
# 出力: /workspace/outputs/ollama_conversion/deepseek-finetuned-q4_k_m.gguf
```

#### Step 5: Ollama登録
```python
# Modelfile作成 & Ollama登録
register_with_ollama(modelfile_path, "deepseek-finetuned")
```

#### Step 6: RAG設定更新
```python
# src/rag/config/rag_config.yaml を自動更新
update_rag_config("deepseek-finetuned")
```

### 3. ドキュメントアップロード
**場所**: RAGシステムページ → PDFファイルアップロード

- PDFをベクトルデータベースにインデックス
- ファインチューニング済みモデルで埋め込み生成

### 4. 質問応答
**場所**: RAGシステムページ → ハイブリッド検索・質問応答

- ベクトル検索 + キーワード検索のハイブリッド
- ファインチューニング済みモデル（Ollama経由）で回答生成

## トラブルシューティング

### エラー: "Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'"
**原因**: 量子化されたモデルを直接GGUF変換しようとしている  
**解決**: `qlora_to_ollama.py`を使用してFP16でマージしてから変換

### エラー: "Ollamaサービスが起動していません"
**解決**: 別ターミナルで以下を実行
```bash
ollama serve
```

### エラー: "メモリ不足"
**対策**:
1. 他のプロセスを終了
2. GPUメモリをクリア: `torch.cuda.empty_cache()`
3. CPUオフロードを有効化

## 設定ファイル

### LoRA設定（デフォルト）
```python
# src/training/lora_finetuning.py
LoRAConfig(
    use_qlora=True,      # QLoRA使用
    qlora_4bit=True,     # 4bit量子化
    r=16,                # LoRAランク
    lora_alpha=32,       # スケーリング係数
)
```

### RAG設定（量子化後）
```yaml
# src/rag/config/rag_config.yaml
llm:
  provider: ollama
  ollama:
    model: deepseek-finetuned
    base_url: http://localhost:11434
    temperature: 0.7
```

## メモリ使用量

| ステージ | メモリ使用量 |
|---------|------------|
| QLoRAファインチューニング | 16-20GB VRAM |
| FP16マージ（一時的） | 32-40GB RAM |
| GGUF変換 | 64GB ディスク |
| Q4_K_M量子化後 | 16GB ディスク |
| Ollama実行時 | 16-20GB RAM |

## コマンドライン実行（手動）

全自動化されていますが、手動実行する場合：

```bash
# 1. QLoRA → Ollama変換
python /workspace/scripts/qlora_to_ollama.py

# 2. 確認
ollama list
ollama run deepseek-finetuned "テスト質問"

# 3. RAGシステム起動
python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050
```

## まとめ

このワークフローにより、メモリ制約のある環境でも32Bモデルを：
1. **効率的にファインチューニング**（QLoRA 4bit）
2. **正しく変換**（FP16経由でGGUF化）
3. **RAGシステムで活用**（Ollama経由）

できるようになりました。