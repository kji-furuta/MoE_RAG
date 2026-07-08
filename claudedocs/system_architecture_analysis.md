# AI Fine-tuning Toolkit システム構造解析報告書

**作成日**: 2025-11-27
**対象システム**: MoE_RAG (AI Fine-tuning Toolkit)
**解析対象**: LoRA学習→Ollama変換→RAG統合フロー

---

## エグゼクティブサマリー

本システムは、DeepSeek-R1-Distill-Qwen-32B-Japaneseをベースモデルとした、メモリ効率的な日本語LLMファインチューニング＋RAGシステムです。LoRAによる軽量学習、GGUF量子化によるモデル軽量化、OllamaによるCPU推論を組み合わせることで、限られたハードウェアリソースでも大規模モデルの実用的な運用を実現しています。

**主要な強み**:
- LoRA (r=64, alpha=128)による高効率ファインチューニング
- 4bit量子化(q4_k_m)によるメモリ使用量75%削減
- ハイブリッド検索(ベクトル70% + キーワード30%)による高精度文書検索
- メモリ不足時の自動Ollamaフォールバック機構

**主要な課題**:
- 32Bモデルには最低20GB GPUメモリが必要（実運用では40GB推奨）
- Ollama統合の複雑性とエラーハンドリングの脆弱性
- 継続学習機能とMoE機能の未成熟な統合

---

## 1. システムアーキテクチャ概要

### 1.1 全体フロー

```
[学習データ]
    ↓
[LoRAファインチューニング]
  - ベースモデル: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
  - LoRA r=64, alpha=128
  - 学習率: 5e-6
  - バッチサイズ: 4
  - Max Length: 2048
    ↓
[LoRAアダプター出力]
  - 保存先: outputs/final_lora_model/
  - サイズ: ~200MB (ベース32B対比で99.4%削減)
    ↓
[GGUF変換 + Ollama登録]
  - llama.cpp/convert_lora_to_gguf.py
  - 量子化: q4_k_m (4bit)
  - Modelfile作成 → ollama create
    ↓
[Ollama Model Repository]
  - ローカルモデル: deepseek-32b-finetuned:latest
  - APIエンドポイント: http://localhost:11434
    ↓
[RAGシステム統合]
  - LLMGenerator.use_ollama_fallback = True
  - HybridSearchEngine (Vector + Keyword)
  - Qdrantベクトルストア (multilingual-e5-large)
    ↓
[統合Webインターフェース]
  - FastAPI (main_unified.py)
  - ポート: 8050
  - エンドポイント: /rag/query, /api/generate
```

### 1.2 主要コンポーネント

| コンポーネント | 実装ファイル | 役割 |
|--------------|-------------|------|
| **LoRAトレーナー** | [src/training/lora_finetuning.py](../src/training/lora_finetuning.py:1) | QLoRA対応のファインチューニング実行 |
| **Ollama変換** | [scripts/convert/convert_finetuned_to_ollama.py](../scripts/convert/convert_finetuned_to_ollama.py:1) | LoRA→GGUF→Ollamaモデル作成 |
| **Ollama統合** | [app/ollama_integration.py](../app/ollama_integration.py:1) | HuggingFace⇔Ollamaハイブリッド推論管理 |
| **RAGエンジン** | [src/rag/core/query_engine.py](../src/rag/core/query_engine.py:1) | ハイブリッド検索+LLM生成統合 |
| **LLM生成器** | [src/rag/core/query_engine.py:63-632](../src/rag/core/query_engine.py#L63-L632) | メモリ最適化ロード+Ollamaフォールバック |
| **Web API** | [app/main_unified.py](../app/main_unified.py) | FastAPI統合サーバー |

---

## 2. LoRAファインチューニングシステム詳細

### 2.1 LoRA設定パラメータ

**現在の設定** ([src/training/lora_finetuning.py:43-64](../src/training/lora_finetuning.py#L43-L64)):

```python
class LoRAConfig:
    r: int = 16                        # デフォルト: 16 → 推奨: 64
    lora_alpha: int = 32               # デフォルト: 32 → 推奨: 128
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_qlora: bool = False            # QLoRAオプション
    qlora_4bit: bool = True            # 4bit量子化
```

**ユーザー指定の最適化設定**:

| パラメータ | デフォルト | 推奨値 | 効果 |
|----------|-----------|--------|------|
| LoRA r | 16 | **64** | ランク数↑ → 表現力向上、メモリ4倍増加 |
| LoRA Alpha | 32 | **128** | スケーリング係数 (通常2×r) |
| 学習率 | 2e-4 | **5e-6** | 大規模モデルの安定学習 |
| エポック数 | 3 | **1** | 過学習防止、時間短縮 |
| Max Length | 512 | **2048** | 長文コンテキスト対応 |
| Batch size | 4 | **4** | 32Bモデルのメモリ制約 |

### 2.2 量子化戦略

**QLoRA実装** ([src/training/lora_finetuning.py:93-106](../src/training/lora_finetuning.py#L93-L106)):

```python
def _get_bnb_config(self) -> BitsAndBytesConfig:
    if self.lora_config.qlora_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,    # ダブル量子化で精度向上
            bnb_4bit_quant_type="nf4"          # NormalFloat4
        )
    else:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
```

**メモリ削減効果**:
- **32B FP16**: 64GB
- **32B 4bit**: 16GB (75%削減)
- **32B 8bit**: 32GB (50%削減)

### 2.3 ターゲットモジュール自動検出

**DeepSeek/Qwen対応** ([src/training/lora_finetuning.py:158-179](../src/training/lora_finetuning.py#L158-L179)):

```python
def _find_target_modules(self) -> List[str]:
    model_type = self.model.config.model_type.lower()

    if "llama" in model_type:  # Qwen/DeepSeekもLlama系
        return ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    elif "gpt-neox" in model_type:
        return ["attention.query_key_value", "attention.dense",
                "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
    # ... 他のモデルタイプ
```

**DeepSeek-R1-Distill-Qwen-32B-Japaneseの場合**:
- モデルタイプ: `qwen2` (Llama系アーキテクチャ)
- ターゲット: 7つのプロジェクション層 (Attention + MLP)
- 学習可能パラメータ: 約2億パラメータ (全体の0.6%)

---

## 3. Ollama統合システム

### 3.1 GGUF変換プロセス

**変換スクリプト** ([scripts/convert/convert_finetuned_to_ollama.py:48-91](../scripts/convert/convert_finetuned_to_ollama.py#L48-L91)):

```python
def _run_gguf_conversion(self, model_name: str) -> Dict[str, Any]:
    output_file = self.output_dir / f"{model_name}.gguf"

    cmd = [
        "python3", "-m", "llama_cpp.convert",
        str(self.model_path),
        "--outfile", str(output_file),
        "--outtype", "q4_k_m"  # 4bit量子化 (k-quant, medium)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
```

**量子化メソッド**: `q4_k_m`
- **q**: 量子化レベル (4bit)
- **k**: k-quant (改良型量子化)
- **m**: medium (精度と速度のバランス)

**変換後のサイズ比較**:
- **LoRAアダプター**: ~200MB
- **GGUF q4_k_m**: ~8GB (ベースモデル統合後)
- **元のFP16モデル**: ~64GB

### 3.2 Modelfile設定

**Ollama用プロンプトテンプレート** ([scripts/convert/convert_finetuned_to_ollama.py:93-117](../scripts/convert/convert_finetuned_to_ollama.py#L93-L117)):

```modelfile
FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"

TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """あなたは道路設計の専門家です。質問に対して正確で分かりやすい回答を提供してください。"""
```

### 3.3 ハイブリッドモデルマネージャー

**メモリ状況に応じた自動切り替え** ([app/ollama_integration.py:97-129](../app/ollama_integration.py#L97-L129)):

```python
def load_model(self, model_name: str, force_ollama: bool = False) -> bool:
    # Ollamaを強制使用またはメモリ不足の場合
    if force_ollama or self.should_use_ollama(model_name):
        logger.info(f"🔄 Ollamaモデルを使用")
        self.use_ollama = True
        return self._test_ollama_connection()

    # HuggingFaceモデルをロード
    logger.info(f"📦 HuggingFaceモデルをロード: {model_name}")
    self.current_model, self.current_tokenizer = load_model_with_optimization(
        model_name,
        device_map="auto",
        load_in_4bit=True  # メモリ節約のため4bit量子化
    )
```

**メモリ判定ロジック** ([app/ollama_integration.py:71-95](../app/ollama_integration.py#L71-L95)):

```python
def should_use_ollama(self, model_name: str) -> bool:
    gpu_free, ram_free = self.check_memory_availability()

    # モデルサイズの推定
    model_size_map = {
        "32b": 64, "22b": 44, "14b": 28, "8b": 16, "3b": 6
    }

    # 量子化考慮（4bit想定）
    estimated_size_quantized = estimated_size / 4

    # メモリ不足判定
    if gpu_free < estimated_size_quantized or gpu_free < self.memory_threshold_gb:
        logger.warning(f"⚠️ メモリ不足: 必要={estimated_size_quantized:.1f}GB, 利用可能={gpu_free:.1f}GB")
        return True  # Ollamaを使用

    return False
```

---

## 4. RAGシステム統合

### 4.1 LLM生成器のメモリ最適化

**初期化フロー** ([src/rag/core/query_engine.py:66-123](../src/rag/core/query_engine.py#L66-L123)):

```python
class LLMGenerator:
    def __init__(self, config: RAGConfig, load_model: bool = True):
        # 継続学習モデルマネージャー
        self.continual_manager = None
        self.use_continual = False

        # 動的LoRA適用モード
        self.dynamic_lora_engine = None
        self.use_dynamic_lora = False

        # Ollamaフォールバック
        self.use_ollama_fallback = False
        self.ollama = None

        # 設定に基づいてOllamaモードを初期化
        if config.llm.provider == 'ollama':
            self._enable_ollama_fallback()
```

**メモリチェック** ([src/rag/core/query_engine.py:124-139](../src/rag/core/query_engine.py#L124-L139)):

```python
def _check_memory_for_model(self) -> bool:
    if not torch.cuda.is_available():
        return False

    # 全GPUのメモリをチェック
    gpu_count = torch.cuda.device_count()
    max_free_memory = 0

    for i in range(gpu_count):
        free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
        max_free_memory = max(max_free_memory, free_memory)

    # 32Bモデルには最低20GB必要 (実際は40GB推奨)
    required_memory = 20
    return max_free_memory >= required_memory
```

**動的量子化** ([src/rag/core/query_engine.py:260-342](../src/rag/core/query_engine.py#L260-L342)):

```python
def _get_optimized_model_kwargs(self, llm_config) -> Dict[str, Any]:
    model_kwargs = {
        'torch_dtype': torch.float16,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True
    }

    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info(current_device)[0] / (1024**3)

        if free_memory < 8:  # 8GB未満の場合
            # 4bit量子化を適用
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )

        # メモリ配分を最適化（各GPUの70%を使用）
        max_memory_dict = {}
        for i in range(torch.cuda.device_count()):
            gpu_free = torch.cuda.mem_get_info(i)[0] / (1024**3)
            gpu_safe = max(1, int(gpu_free * 0.7))
            max_memory_dict[i] = f"{gpu_safe}GB"
        max_memory_dict['cpu'] = '32GB'

        model_kwargs.update({
            'device_map': 'auto',
            'max_memory': max_memory_dict
        })
```

### 4.2 ハイブリッド検索システム

**検索重み付け** ([src/rag/config/rag_config.yaml:151-172](../src/rag/config/rag_config.yaml#L151-L172)):

```yaml
retrieval:
  hybrid_search:
    enabled: true
    vector_weight: 0.7    # ベクトル検索の重み
    keyword_weight: 0.3   # キーワード検索の重み
  keyword_engine:
    backend: bm25
    max_features: 30000
    ngram_range: (2, 4)   # 2-gramから4-gramまで
    min_df: 2
    rebuild_threshold: 200
  reranking:
    enabled: true
    model: gpt-neox-20b-lora-20250906_104924:latest
  top_k: 10
  rerank_top_k: 5
```

**ベクトルストア** ([src/rag/config/rag_config.yaml:215-223](../src/rag/config/rag_config.yaml#L215-L223)):

```yaml
vector_store:
  type: qdrant
  qdrant:
    url: http://qdrant:6333
    collection_name: road_design_docs
    vector_dim: 1024
    prefer_grpc: false
    timeout: 60
```

**埋め込みモデル** ([src/rag/config/rag_config.yaml:19-25](../src/rag/config/rag_config.yaml#L19-L25)):

```yaml
embedding:
  model_name: intfloat/multilingual-e5-large
  embedding_dim: 1024
  max_length: 512
  batch_size: 32
  device: cuda
  normalize_embeddings: true
```

### 4.3 Ollamaフォールバック生成

**メモリ不足時の自動切り替え** ([src/rag/core/query_engine.py:531-595](../src/rag/core/query_engine.py#L531-L595)):

```python
def _ollama_generation(self, prompt: str, context: str) -> str:
    # メモリ不足の警告メッセージを追加
    memory_warning = ""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        max_free_memory = 0
        total_free_memory = 0

        for i in range(gpu_count):
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
            total_free_memory += free_mem
            max_free_memory = max(max_free_memory, free_mem)

        if max_free_memory < 20:  # 32Bモデルには最低20GB必要
            memory_warning = (
                f"\n\n【システム通知】GPUメモリ不足のため、ファインチューニング済みモデルを読み込めません。\n"
                f"最大単一GPU空きメモリ: {max_free_memory:.2f}GB / 必要メモリ: 約20GB以上\n"
                f"合計GPU空きメモリ: {total_free_memory:.2f}GB (GPU数: {gpu_count})\n"
                f"代替モデル（Ollama）で回答を生成しています。\n"
            )

    # Ollamaモデル名を設定から取得
    ollama_model = 'llama3.2:3b'  # デフォルト
    if hasattr(self.config.llm, 'ollama') and hasattr(self.config.llm.ollama, 'model'):
        ollama_model = self.config.llm.ollama.model

    # Ollamaで生成
    result = self.ollama.generate_text(
        model_name=ollama_model,
        prompt=full_prompt,
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )
```

**拡張RAGプロンプト** ([src/rag/core/query_engine.py:1462-1502](../src/rag/core/query_engine.py#L1462-L1502)):

```python
def _build_enhanced_rag_prompt(self, query: str, context: str) -> str:
    if context:
        prompt = f"""# 道路設計の専門家としての回答

あなたは経験豊富な道路設計の専門家です。以下の参考資料を基に、質問に対して**詳細で実用的な回答**を提供してください。

## 参考資料
{context}

## 質問
{query}

## 回答の指示
1. **具体的で詳細な説明**を提供してください
2. **数値や基準値**は参考資料から正確に引用してください
3. **実務での注意点やポイント**を含めてください
4. **関連する法規や基準**があれば言及してください
5. **1500-2000文字程度**の充実した回答をお願いします
6. 参考資料の情報を根拠として、**[出典: …]という形で出典を明記**してください

## 回答"""
```

---

## 5. システム特徴と技術的強み

### 5.1 メモリ効率最適化

**1. 段階的メモリフォールバック**

```
┌─────────────────────────────────────────────────┐
│ 標準モード (GPU >= 40GB)                          │
│ - ベースモデル: DeepSeek-32B FP16               │
│ - LoRAアダプター: r=64 適用                      │
│ - メモリ使用量: ~36GB                            │
└─────────────────────────────────────────────────┘
              ↓ (メモリ不足)
┌─────────────────────────────────────────────────┐
│ 量子化モード (GPU 20-40GB)                       │
│ - ベースモデル: DeepSeek-32B 4bit量子化          │
│ - LoRAアダプター: r=64 適用                      │
│ - メモリ使用量: ~18GB                            │
└─────────────────────────────────────────────────┘
              ↓ (メモリ不足)
┌─────────────────────────────────────────────────┐
│ Ollamaフォールバックモード (GPU < 20GB)          │
│ - Ollamaモデル: deepseek-32b-finetuned:latest   │
│ - GGUF q4_k_m量子化                              │
│ - メモリ使用量: CPU RAM ~8GB                     │
└─────────────────────────────────────────────────┘
```

**2. マルチGPU対応**

```python
# 各GPUの70%を使用
max_memory_dict = {}
for i in range(torch.cuda.device_count()):
    gpu_free = torch.cuda.mem_get_info(i)[0] / (1024**3)
    gpu_safe = max(1, int(gpu_free * 0.7))
    max_memory_dict[i] = f"{gpu_safe}GB"
max_memory_dict['cpu'] = '32GB'  # CPUメモリ
```

### 5.2 LoRA高精度学習

**特徴**:
1. **高ランク (r=64)**: 表現力向上、複雑なドメイン知識の学習
2. **低学習率 (5e-6)**: 大規模モデルの安定学習
3. **長文対応 (2048トークン)**: 道路設計基準書の詳細記述対応
4. **ダブル量子化**: QLoRA使用時の精度向上

**学習可能パラメータ比較**:

| モデル | 全体パラメータ | LoRA学習 | 学習効率 |
|-------|--------------|---------|---------|
| DeepSeek-32B (r=16) | 320億 | 5千万 | 0.16% |
| DeepSeek-32B (r=64) | 320億 | 2億 | 0.62% |
| Full Fine-tuning | 320億 | 320億 | 100% |

### 5.3 ハイブリッド検索の高精度化

**BM25 + ベクトル検索**:

```python
# ハイブリッドスコア計算
final_score = (vector_score * 0.7) + (keyword_score * 0.3)

# Rerankingでさらに精度向上
reranked_results = self.reranker.rerank(
    query=query,
    results=search_results,
    top_k=5
)
```

**検索精度の向上要因**:
1. **ベクトル検索 (70%)**: 意味的類似性
2. **キーワード検索 (30%)**: 専門用語・数値の正確一致
3. **Reranking**: GPT-NeoX-20Bによる再順位付け
4. **メタデータフィルター**: 文書タイプ・バージョン・章節による絞り込み

---

## 6. 課題と改善提案

### 6.1 現在の課題

#### 6.1.1 メモリ管理の脆弱性

**問題点**:
- 32Bモデルの実運用には理論上20GBだが、実際は40GB推奨
- メモリチェックが楽観的 (70%使用を許可) → OOM発生リスク
- マルチGPUメモリ配分が均等でない場合の対応不足

**エビデンス**:
```python
# query_engine.py:162-169
required_memory = 20  # GB
if max_free_memory < required_memory:
    logger.warning(f"GPUメモリ不足: 最大単一GPU {max_free_memory:.2f}GB")
    # → 実際は30GB以上必要なケースあり
```

**改善提案**:
```python
# より保守的なメモリチェック
def _check_memory_for_model(self) -> bool:
    # 32B 4bit量子化の実測必要メモリ
    MODEL_MEMORY_REQUIREMENTS = {
        "32b": {"4bit": 18, "8bit": 36, "fp16": 64},
        "22b": {"4bit": 12, "8bit": 24, "fp16": 44},
        # ...
    }

    required = MODEL_MEMORY_REQUIREMENTS["32b"]["4bit"]
    safety_margin = 1.3  # 30%のマージン
    required_with_margin = required * safety_margin  # 23.4GB

    return max_free_memory >= required_with_margin
```

#### 6.1.2 Ollama統合の複雑性

**問題点**:
- Docker環境からWSL/ホストのOllama接続が複雑
- 接続先URL試行ロジックが脆弱 ([scripts/convert/ollama_integration.py:14-38](../scripts/convert/ollama_integration.py#L14-L38))
- モデル登録失敗時のエラーハンドリング不足

**改善提案**:
```python
# 環境変数ベースの設定
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# リトライ機構
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
)
def _connect_ollama(url: str) -> bool:
    response = requests.get(f"{url}/api/tags", timeout=5)
    return response.status_code == 200
```

#### 6.1.3 継続学習機能の未統合

**問題点**:
- ContinualModelManagerが初期化されるが実際には使用されていない
- タスク検出ロジック ([query_engine.py:369-393](../src/rag/core/query_engine.py#L369-L393)) が実行されない
- EWC (Elastic Weight Consolidation) の効果が不明

**改善提案**:
```python
# タスク検出を有効化
def _detect_query_task(self, query_text: str) -> Optional[str]:
    """クエリから関連タスクを検出"""
    task_keywords = {
        "task1_horizontal_curve": ["平面曲線", "曲線半径", "片勾配"],
        "task2_vertical_curve": ["縦断曲線", "縦断勾配", "視距"],
        # ...
    }

    for task_name, keywords in task_keywords.items():
        if any(kw in query_text for kw in keywords):
            return task_name
    return None
```

#### 6.1.4 プロンプトテンプレートの固定化

**問題点**:
- Modelfileのプロンプトテンプレートが固定 ([convert_finetuned_to_ollama.py:107-114](../scripts/convert/convert_finetuned_to_ollama.py#L107-L114))
- ChatMLフォーマット (`<|im_start|>`) がDeepSeek-R1に最適化されていない可能性

**改善提案**:
```python
# モデルタイプごとのテンプレート
PROMPT_TEMPLATES = {
    "deepseek": """{{ if .System }}System: {{ .System }}\n{{ end }}User: {{ .Prompt }}\nAssistant:""",
    "qwen": """<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant""",
    "llama": """[INST] {{ .System }}\n{{ .Prompt }} [/INST]"""
}
```

### 6.2 性能最適化提案

#### 6.2.1 バッチ処理の最適化

**現状**: バッチサイズ4 (固定)

**提案**: 動的バッチサイズ調整
```python
def get_optimal_batch_size(model_size_gb: int, available_memory_gb: float) -> int:
    """利用可能なメモリに基づいて最適なバッチサイズを計算"""
    # 1サンプルあたりのメモリ使用量を推定
    memory_per_sample = model_size_gb * 0.1  # 経験則

    # 安全マージンを考慮
    safe_memory = available_memory_gb * 0.8
    optimal_batch = int(safe_memory / memory_per_sample)

    # 最小・最大値で制限
    return max(1, min(optimal_batch, 16))
```

#### 6.2.2 LoRAマージの高速化

**現状**: マージ処理がボトルネック ([lora_finetuning.py:384-395](../src/training/lora_finetuning.py#L384-L395))

**提案**: Fast LoRAマージ
```python
# PEFTライブラリの高速化機能を利用
def _save_final_model_fast(self):
    if not self.lora_config.use_qlora:
        from peft import merge_and_unload
        merged_model = self.model.merge_and_unload(
            progressbar=True,
            safe_merge=True  # セーフマージで精度保証
        )
```

#### 6.2.3 キャッシング戦略

**提案**: 中間結果のキャッシング
```python
from functools import lru_cache
import hashlib

class CachedLLMGenerator(LLMGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def generate(self, prompt: str, context: str, **kwargs) -> str:
        # キャッシュキー生成
        cache_key = hashlib.md5(
            f"{prompt}{context}".encode()
        ).hexdigest()

        if cache_key in self.cache:
            logger.info("キャッシュヒット")
            return self.cache[cache_key]

        # 生成
        result = super().generate(prompt, context, **kwargs)
        self.cache[cache_key] = result
        return result
```

---

## 7. 推奨設定とベストプラクティス

### 7.1 ハードウェア要件

#### 最小構成
- **GPU**: NVIDIA RTX 3090 (24GB) x1
- **RAM**: 64GB
- **ストレージ**: NVMe SSD 500GB
- **用途**: 軽量モデル (8B以下) + Ollamaフォールバック

#### 推奨構成 (32Bモデル)
- **GPU**: NVIDIA A100 (40GB) x2
- **RAM**: 128GB
- **ストレージ**: NVMe SSD 1TB
- **用途**: 32Bモデル full fine-tuning + RAG

#### 最適構成
- **GPU**: NVIDIA H100 (80GB) x2
- **RAM**: 256GB
- **ストレージ**: NVMe SSD 2TB
- **用途**: 70Bモデル + MoE + 継続学習

### 7.2 LoRAハイパーパラメータチューニング

**タスク別推奨設定**:

| タスク | r | alpha | 学習率 | エポック | 説明 |
|--------|---|-------|-------|---------|------|
| **汎用QA** | 16-32 | 32-64 | 1e-4 | 3 | 標準設定 |
| **ドメイン特化** | 64-128 | 128-256 | 5e-6 | 1-2 | 本システムの用途 |
| **少量データ** | 8-16 | 16-32 | 2e-4 | 5-10 | 過学習防止 |
| **大規模データ** | 32-64 | 64-128 | 3e-5 | 2-3 | バランス重視 |

### 7.3 Ollama運用ベストプラクティス

**1. モデル命名規則**
```bash
# タスク名_バージョン_ベースモデル_量子化レベル:latest
ollama create task1_v1_deepseek-32b_q4km:latest
```

**2. 量子化レベルの選択**

| 量子化 | サイズ | 精度 | 速度 | 用途 |
|-------|--------|------|------|------|
| q4_0 | 小 | 低 | 速 | プロトタイピング |
| q4_k_m | 中 | 中 | 中 | **推奨・本番環境** |
| q5_k_m | 中 | 高 | 中 | 高精度要求時 |
| q8_0 | 大 | 最高 | 遅 | ベンチマーク |

**3. Modelfileのバージョン管理**
```bash
# Gitで管理
git add modelfiles/deepseek-32b-finetuned.Modelfile
git commit -m "feat: Update system prompt for v2"
```

---

## 8. まとめと戦略的提言

### 8.1 システムの総合評価

**技術的成熟度**: ★★★★☆ (4/5)
- LoRAファインチューニング: 成熟
- Ollama統合: 実用レベル
- RAGシステム: 高度な実装
- 継続学習: 未完成

**運用安定性**: ★★★☆☆ (3/5)
- メモリ管理: 改善の余地あり
- エラーハンドリング: 基本機能は実装
- ログ出力: 充実
- 監視機能: 未実装

**スケーラビリティ**: ★★★★☆ (4/5)
- マルチGPU対応: 良好
- バッチ処理: 基本機能あり
- 分散学習: 未実装

### 8.2 優先度別改善ロードマップ

#### フェーズ1: 安定性向上 (1-2週間)
1. ✅ **メモリチェックの保守化** (Priority: HIGH)
   - 安全マージン30%追加
   - 実測値ベースの閾値設定

2. ✅ **Ollama接続の堅牢化** (Priority: HIGH)
   - リトライ機構
   - 環境変数ベース設定

3. ✅ **エラーハンドリング強化** (Priority: MEDIUM)
   - カスタム例外クラス
   - 詳細ログ出力

#### フェーズ2: 性能最適化 (2-4週間)
1. ✅ **動的バッチサイズ調整** (Priority: MEDIUM)
2. ✅ **キャッシング戦略** (Priority: LOW)
3. ✅ **LoRAマージ高速化** (Priority: LOW)

#### フェーズ3: 機能拡張 (1-2ヶ月)
1. ✅ **継続学習の完全統合** (Priority: MEDIUM)
   - タスク検出ロジック実装
   - EWC効果検証

2. ✅ **MoE統合の安定化** (Priority: LOW)
3. ✅ **監視ダッシュボード** (Priority: LOW)

### 8.3 結論

本システムは、**LoRA高効率学習**、**GGUF量子化**、**Ollamaフォールバック**を組み合わせることで、限られたハードウェアリソースでも大規模モデルの実用運用を実現した先進的な実装です。

**主要な成功要因**:
1. メモリ最適化の多段階フォールバック
2. LoRA r=64による高精度ドメイン適応
3. ハイブリッド検索による高精度文書検索
4. Ollama統合によるCPU推論オプション

**今後の戦略的方向性**:
1. **短期**: 安定性向上とエラーハンドリング強化
2. **中期**: 継続学習の完全統合とMoE最適化
3. **長期**: マルチモーダル対応と分散学習実装

本システムは、道路設計特化型RAGシステムとして十分な実用性を持ち、適切な改善により、さらに高度な実運用環境での活用が期待できます。

---

## 付録A: 主要ファイル構造

```
MoE_RAG/
├── src/
│   ├── training/
│   │   ├── lora_finetuning.py           # LoRAトレーニング実装
│   │   ├── continual_learning_pipeline.py  # 継続学習パイプライン
│   │   └── ewc_utils.py                 # EWC実装
│   ├── rag/
│   │   ├── core/
│   │   │   ├── query_engine.py          # RAGクエリエンジン
│   │   │   └── citation_engine.py       # 引用エンジン
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py         # ハイブリッド検索
│   │   │   └── reranker.py              # リランカー
│   │   └── config/
│   │       └── rag_config.yaml          # RAG設定
│   └── inference/
│       └── vllm_integration.py          # vLLM統合
├── app/
│   ├── main_unified.py                  # FastAPI統合サーバー
│   ├── ollama_integration.py            # Ollama統合
│   └── memory_optimized_loader.py       # メモリ最適化ローダー
├── scripts/
│   └── convert/
│       ├── convert_finetuned_to_ollama.py  # Ollama変換スクリプト
│       └── ollama_integration.py        # Ollama統合ユーティリティ
├── config/
│   └── model_config.yaml                # モデル設定
└── outputs/
    ├── final_lora_model/                # LoRAアダプター
    └── continual_task_*/                # 継続学習モデル
```

## 付録B: 用語集

| 用語 | 説明 |
|------|------|
| **LoRA** | Low-Rank Adaptation - 大規模モデルの効率的ファインチューニング手法 |
| **QLoRA** | Quantized LoRA - 量子化を組み合わせたLoRA |
| **GGUF** | GPT-Generated Unified Format - llama.cppの量子化モデルフォーマット |
| **q4_k_m** | 4bit k-quant medium - GGUF量子化メソッド |
| **RAG** | Retrieval-Augmented Generation - 検索拡張生成 |
| **BM25** | Best Matching 25 - 確率的情報検索モデル |
| **EWC** | Elastic Weight Consolidation - 継続学習のための重み正則化 |
| **MoE** | Mixture of Experts - エキスパート混合モデル |
| **Qdrant** | ベクトル類似検索エンジン |
| **ChatML** | Chat Markup Language - チャット形式のプロンプトフォーマット |

## 付録C: 参考リソース

1. **LoRA論文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **QLoRA論文**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. **llama.cpp**: https://github.com/ggerganov/llama.cpp
4. **Ollama**: https://ollama.ai/
5. **Qdrant**: https://qdrant.tech/
6. **PEFT**: https://github.com/huggingface/peft

---

**報告書終了**
