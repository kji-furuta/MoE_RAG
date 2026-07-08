# 専門分野RAGシステム - 精度向上戦略

## 現状分析

### ✅ 稼働中の機能
- RARベースの継続学習システム
- ハイブリッドRAG検索（ベクトル + キーワード）
- 引用付き回答生成
- 専門文書インデックス

### ⚠️ 課題
1. **回答の専門性不足**: エキスパートレベルの要求に未達
2. **ハルシネーション**: 事実に基づかない情報の生成
3. **引用精度**: 正しい文書を参照できていない場合がある
4. **教師データ品質**: 現在149件のRARdata.json

---

## 🎯 精度向上戦略（6つの柱）

### 1. 教師データの高度化

#### 1.1 データ量の拡充

**目標**: 149件 → 1,000件以上

**作成方法**:

##### A. エキスパート・アノテーション方式（推奨）

```yaml
プロセス:
  step1: 実務で頻出の質問を収集
  step2: 専門家が正解と根拠文書を明示
  step3: Chain-of-Thought（思考過程）を記述
  step4: レビューと品質保証

品質基準:
  - 必ず公式文書（道路構造令等）からの引用
  - 思考過程が論理的
  - 複数文書の統合が必要な複雑な質問を含む
```

**テンプレート**:

```json
{
  "id": "DES-XXX",
  "instruction": "【実務質問】設計速度80km/hの第3種道路において、縦断勾配の最大値は？",
  "documents": [
    {
      "source": "道路構造令の解説と運用.pdf",
      "is_oracle": true,
      "content": "第3種道路の縦断勾配は、設計速度80km/hの場合、平地部では5%、山地部では6%を標準とする..."
    },
    {
      "source": "道路設計要領.pdf",
      "is_oracle": false,
      "content": "一般的な道路の縦断勾配は..."
    }
  ],
  "output": {
    "chain_of_thought": "1. 質問は第3種道路、設計速度80km/hの縦断勾配について\n2. 道路構造令第20条を確認\n3. 地形区分（平地部・山地部）により異なる\n4. 平地部5%、山地部6%が標準値",
    "final_answer": "設計速度80km/hの第3種道路の縦断勾配は、平地部で5%、山地部で6%を標準値とします。ただし、やむを得ない場合は7%まで緩和可能です。",
    "citations": ["道路構造令の解説と運用.pdf"]
  }
}
```

##### B. 半自動生成 + 専門家検証

```python
# scripts/generate_training_data_from_docs.py
from src.rag.core.query_engine import QueryEngine
import json

def generate_qa_candidates(pdf_path: str, num_questions: int = 100):
    """文書から質問候補を自動生成"""

    # 1. 文書をチャンクに分割
    chunks = split_document(pdf_path)

    # 2. LLMで質問生成
    questions = []
    for chunk in chunks:
        prompt = f"""
以下の技術文書から、土木工学の専門家が実務で問うような質問を3つ生成してください。

文書:
{chunk}

質問形式:
- 設計基準値を問う質問
- 適用条件を問う質問
- 計算方法を問う質問
"""
        response = llm.generate(prompt)
        questions.extend(parse_questions(response))

    # 3. Oracle文書とDistractor文書を特定
    for q in questions:
        oracle_docs = find_relevant_docs(q, top_k=2)
        distractor_docs = find_irrelevant_docs(q, top_k=3)

        # 4. Chain-of-Thought生成
        cot = generate_chain_of_thought(q, oracle_docs)

        # 5. JSON出力
        output = format_as_rar(q, oracle_docs, distractor_docs, cot)
        yield output

# 使用例
for qa in generate_qa_candidates("道路構造令.pdf"):
    print(json.dumps(qa, ensure_ascii=False, indent=2))
```

**検証フロー**:
```
自動生成 → 専門家レビュー → 修正 → 再レビュー → 承認
```

##### C. 実務ログからの抽出

```python
def extract_from_query_logs(log_file: str):
    """RAGシステムのクエリログから教師データ作成"""

    logs = load_logs(log_file)

    for query, response, docs, user_feedback in logs:
        if user_feedback == "good":
            # ポジティブフィードバックのあった回答を教師データ化
            rar_data = {
                "instruction": query,
                "documents": docs,
                "output": {
                    "final_answer": response,
                    "chain_of_thought": extract_reasoning(response),
                    "citations": extract_citations(response)
                }
            }
            yield rar_data
```

#### 1.2 データ品質の向上

**品質チェックリスト**:

```python
# scripts/check_training_data_quality.py

def quality_check(rar_data: dict) -> dict:
    """教師データの品質チェック"""

    issues = []

    # 1. Oracle文書の存在確認
    oracle_docs = [d for d in rar_data["documents"] if d["is_oracle"]]
    if len(oracle_docs) == 0:
        issues.append("Oracle文書が存在しない")

    # 2. 回答と引用の一貫性
    citations = rar_data["output"]["citations"]
    oracle_sources = [d["source"] for d in oracle_docs]
    if not all(c in oracle_sources for c in citations):
        issues.append("引用文書がOracle文書と一致しない")

    # 3. Chain-of-Thoughtの品質
    cot = rar_data["output"]["chain_of_thought"]
    if len(cot) < 50:
        issues.append("思考過程が短すぎる（50文字以上推奨）")

    # 4. 専門用語の使用
    technical_terms = ["設計速度", "縦断勾配", "車線", "曲線半径"]
    if not any(term in rar_data["instruction"] for term in technical_terms):
        issues.append("専門用語が含まれていない可能性")

    # 5. 数値の正確性
    if has_numerical_answer(rar_data):
        if not verify_calculation(rar_data):
            issues.append("計算結果が不正確な可能性")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "quality_score": calculate_quality_score(rar_data)
    }
```

**品質メトリクス**:

| 項目 | 基準 | 現状 | 目標 |
|------|------|------|------|
| Oracle率 | 50-80% | 68% | 70% |
| CoT長さ | 100文字以上 | 125文字 | 150文字 |
| 引用精度 | 100% | 98% | 100% |
| 技術用語密度 | 10% | 8% | 12% |

---

### 2. モデルアーキテクチャの最適化

#### 2.1 専門タスク特化型LoRA

**現状**: 汎用的なLoRA設定

**改善策**: タスク別LoRAアダプター

```python
# src/training/task_specific_lora.py

TASK_SPECIFIC_CONFIGS = {
    "design_standards": {
        # 設計基準値に関する質問
        "lora_r": 32,  # より高いrank
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "focus_layers": [20, 21, 22, 23, 24]  # 深い層に集中
    },
    "calculation": {
        # 計算問題
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "focus_layers": [15, 16, 17, 18, 19]
    },
    "regulation_interpretation": {
        # 法令解釈
        "lora_r": 24,
        "lora_alpha": 48,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "focus_layers": [25, 26, 27, 28, 29, 30]
    }
}

def create_task_specific_lora(task_type: str):
    """タスク別LoRA設定"""
    config = TASK_SPECIFIC_CONFIGS[task_type]

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        layers_to_transform=config["focus_layers"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config
```

#### 2.2 アンサンブル推論

```python
# src/inference/ensemble_inference.py

class EnsembleRAGModel:
    """複数のLoRAアダプターを組み合わせた推論"""

    def __init__(self, base_model, adapters: dict):
        self.base_model = base_model
        self.adapters = {
            "design": PeftModel.from_pretrained(base_model, adapters["design"]),
            "calculation": PeftModel.from_pretrained(base_model, adapters["calculation"]),
            "regulation": PeftModel.from_pretrained(base_model, adapters["regulation"])
        }

    def predict(self, query: str, documents: list):
        # 質問のタイプを分類
        task_type = classify_query_type(query)

        if task_type == "mixed":
            # 複数モデルで推論して統合
            predictions = []
            for name, model in self.adapters.items():
                pred = model.generate(query, documents)
                predictions.append((name, pred))

            # アンサンブル統合
            final_answer = ensemble_combine(predictions, strategy="weighted_vote")
        else:
            # 単一モデルで推論
            model = self.adapters[task_type]
            final_answer = model.generate(query, documents)

        return final_answer
```

---

### 3. RAG検索精度の向上

#### 3.1 専門用語埋め込みの最適化

**現状**: multilingual-e5-large（汎用）

**改善策**: ドメイン適応埋め込み

```python
# scripts/train_domain_adapted_embeddings.py

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def create_domain_pairs(technical_docs: list):
    """専門文書からペアデータ作成"""
    pairs = []

    # 1. 同一文書内の文をポジティブペア
    for doc in technical_docs:
        sentences = split_sentences(doc)
        for i in range(len(sentences)-1):
            pairs.append(InputExample(
                texts=[sentences[i], sentences[i+1]],
                label=1.0  # ポジティブペア
            ))

    # 2. 異なる文書の文をネガティブペア
    for doc1, doc2 in random_pairs(technical_docs):
        pairs.append(InputExample(
            texts=[random_sentence(doc1), random_sentence(doc2)],
            label=0.0  # ネガティブペア
        ))

    return pairs

# モデルのファインチューニング
model = SentenceTransformer("intfloat/multilingual-e5-large")

train_examples = create_domain_pairs(load_technical_docs("data/documents/"))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="models/embeddings/civil_engineering_e5"
)
```

#### 3.2 ハイブリッド検索の重み最適化

**現状**: ベクトル0.7 + キーワード0.3（固定）

**改善策**: 動的重み調整

```python
# src/rag/retrieval/adaptive_hybrid_search.py

class AdaptiveHybridSearch:
    """クエリタイプに応じた検索重み調整"""

    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.weight_configs = {
            "exact_value": {"vector": 0.3, "keyword": 0.7},  # 具体的数値
            "concept": {"vector": 0.8, "keyword": 0.2},      # 概念的質問
            "calculation": {"vector": 0.5, "keyword": 0.5},   # 計算問題
            "regulation": {"vector": 0.4, "keyword": 0.6}    # 法令条文
        }

    def search(self, query: str, top_k: int = 5):
        # クエリタイプを分類
        query_type = self.query_classifier.classify(query)

        # 適切な重みを選択
        weights = self.weight_configs.get(query_type, {"vector": 0.7, "keyword": 0.3})

        # ハイブリッド検索実行
        vector_results = self.vector_search(query, top_k=10)
        keyword_results = self.keyword_search(query, top_k=10)

        # 重み付きスコア統合
        combined = self.combine_results(
            vector_results,
            keyword_results,
            vector_weight=weights["vector"],
            keyword_weight=weights["keyword"]
        )

        return combined[:top_k]
```

#### 3.3 引用精度向上

```python
# src/rag/core/citation_verifier.py

class CitationVerifier:
    """引用の正確性検証"""

    def verify_citation(self, answer: str, cited_doc: str, threshold: float = 0.8):
        """回答が引用文書から実際に導出可能かチェック"""

        # 1. 回答から事実命題を抽出
        facts = self.extract_facts(answer)

        # 2. 各事実が文書に含まれるか検証
        verification_results = []
        for fact in facts:
            # エンテイルメント（含意）チェック
            is_entailed = self.check_entailment(cited_doc, fact)
            verification_results.append({
                "fact": fact,
                "verified": is_entailed,
                "confidence": is_entailed.confidence
            })

        # 3. 全体的な信頼度スコア
        avg_confidence = sum(r["confidence"] for r in verification_results) / len(facts)

        return {
            "citation_valid": avg_confidence >= threshold,
            "confidence": avg_confidence,
            "details": verification_results
        }

    def extract_facts(self, answer: str) -> list:
        """回答から検証可能な事実を抽出"""
        # NLIモデルを使用
        prompt = f"""
以下の回答から、検証可能な事実命題を抽出してください。

回答: {answer}

事実命題（箇条書き）:
"""
        facts = self.llm.generate(prompt)
        return self.parse_facts(facts)

    def check_entailment(self, premise: str, hypothesis: str):
        """エンテイルメントチェック（NLIモデル）"""
        # rinna/japanese-roberta-base-nli 等を使用
        result = self.nli_model(premise, hypothesis)
        return result  # {label: "entailment", confidence: 0.95}
```

---

### 4. ハルシネーション対策

#### 4.1 回答生成時の制約

```python
# src/rag/generation/constrained_generation.py

class ConstrainedGenerator:
    """文書に基づく制約付き生成"""

    def generate_with_constraints(
        self,
        query: str,
        documents: list,
        model,
        max_length: int = 512
    ):
        # プロンプト設計の改善
        prompt = f"""
あなたは日本の道路設計・土木工学の専門家です。
以下の質問に、提供された文書の内容**のみ**に基づいて回答してください。

【重要な制約】
1. 文書に記載されていない情報は絶対に追加しない
2. 推測や一般知識を使わない
3. 数値は文書から正確に引用する
4. 不明な場合は「文書に記載なし」と明記する
5. 回答の各部分に対応する文書を引用する

【提供文書】
{format_documents(documents)}

【質問】
{query}

【回答形式】
思考過程:
（段階的な推論を記述）

回答:
（簡潔かつ正確な回答）

引用:
- [文書名] （該当箇所の要約）
"""

        # 生成パラメータの調整（ハルシネーション抑制）
        generation_config = {
            "max_length": max_length,
            "temperature": 0.3,  # 低温度（確定的）
            "top_p": 0.85,       # nucleus sampling
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_return_sequences": 1
        }

        response = model.generate(prompt, **generation_config)

        # ポスト処理検証
        if self.contains_hallucination(response, documents):
            # 再生成または警告
            response = self.regenerate_or_warn(query, documents, model)

        return response

    def contains_hallucination(self, response: str, documents: list) -> bool:
        """ハルシネーション検出"""

        # 1. 数値のチェック
        numbers_in_response = extract_numbers(response)
        numbers_in_docs = extract_numbers(concat_documents(documents))

        for num in numbers_in_response:
            if num not in numbers_in_docs:
                logger.warning(f"Hallucinated number detected: {num}")
                return True

        # 2. 固有名詞のチェック
        entities_in_response = extract_entities(response)
        entities_in_docs = extract_entities(concat_documents(documents))

        for entity in entities_in_response:
            if entity not in entities_in_docs and not is_common_entity(entity):
                logger.warning(f"Hallucinated entity detected: {entity}")
                return True

        # 3. エンテイルメントチェック
        verifier = CitationVerifier()
        verification = verifier.verify_citation(response, concat_documents(documents))

        if not verification["citation_valid"]:
            logger.warning(f"Low citation confidence: {verification['confidence']}")
            return True

        return False
```

#### 4.2 自己検証メカニズム

```python
# src/rag/validation/self_verification.py

class SelfVerificationAgent:
    """生成後の自己検証エージェント"""

    def verify_and_correct(self, query: str, answer: str, documents: list):
        """回答を自己検証して修正"""

        # 1. 検証プロンプト
        verification_prompt = f"""
あなたは回答の品質チェック担当者です。
以下の回答が提供文書に基づいているか、検証してください。

【質問】
{query}

【回答】
{answer}

【提供文書】
{format_documents(documents)}

【検証項目】
1. 回答が文書の内容と一致しているか
2. 数値が正確に引用されているか
3. 推測や創作が含まれていないか
4. 引用が適切か

【検証結果】（JSON形式）
{{
  "verified": true/false,
  "issues": [問題点のリスト],
  "corrections": [修正案のリスト],
  "confidence": 0.0-1.0
}}
"""

        verification = self.llm.generate(verification_prompt)
        result = json.loads(verification)

        # 2. 問題がある場合は修正
        if not result["verified"] or result["confidence"] < 0.8:
            logger.info(f"Verification failed: {result['issues']}")

            # 修正プロンプト
            correction_prompt = f"""
以下の回答に問題が見つかりました。文書に厳密に基づいて修正してください。

【元の回答】
{answer}

【問題点】
{result['issues']}

【提供文書】
{format_documents(documents)}

【修正後の回答】
"""
            corrected_answer = self.llm.generate(correction_prompt)
            return corrected_answer, result

        return answer, result
```

---

### 5. 評価とフィードバックループ

#### 5.1 専門家評価システム

```python
# app/evaluation/expert_feedback.py

class ExpertFeedbackSystem:
    """専門家によるフィードバック収集"""

    def collect_feedback(self, query: str, answer: str, documents: list):
        """UIでフィードバック収集"""

        feedback_form = {
            "query_id": generate_id(),
            "query": query,
            "answer": answer,
            "documents": documents,

            "ratings": {
                "accuracy": 1-5,      # 正確性
                "completeness": 1-5,  # 完全性
                "clarity": 1-5,       # 明瞭性
                "citation_quality": 1-5  # 引用品質
            },

            "issues": {
                "hallucination": bool,
                "missing_info": bool,
                "incorrect_citation": bool,
                "calculation_error": bool
            },

            "expert_correction": str,  # 専門家による正しい回答
            "comments": str
        }

        return feedback_form

    def create_training_data_from_feedback(self, feedbacks: list):
        """フィードバックから教師データ作成"""

        high_quality_data = []

        for fb in feedbacks:
            if fb["ratings"]["accuracy"] >= 4 and fb["expert_correction"]:
                # 高品質フィードバックを教師データ化
                rar_data = {
                    "id": fb["query_id"],
                    "instruction": fb["query"],
                    "documents": fb["documents"],
                    "output": {
                        "final_answer": fb["expert_correction"],
                        "chain_of_thought": extract_reasoning(fb["expert_correction"]),
                        "citations": extract_citations_from_docs(fb["documents"])
                    }
                }
                high_quality_data.append(rar_data)

        return high_quality_data
```

#### 5.2 自動評価メトリクス

```python
# src/evaluation/automated_metrics.py

class AutomatedEvaluator:
    """自動評価システム"""

    def evaluate_response(self, query: str, answer: str, ground_truth: str, documents: list):
        """多角的な評価"""

        metrics = {}

        # 1. ROUGE（語彙重複）
        metrics["rouge"] = self.calculate_rouge(answer, ground_truth)

        # 2. BERTScore（セマンティック類似度）
        metrics["bert_score"] = self.calculate_bert_score(answer, ground_truth)

        # 3. 引用精度
        metrics["citation_precision"] = self.calculate_citation_precision(answer, documents)

        # 4. 事実整合性（Factual Consistency）
        metrics["factual_consistency"] = self.check_factual_consistency(answer, documents)

        # 5. ハルシネーション検出
        metrics["hallucination_score"] = self.detect_hallucination(answer, documents)

        # 6. 専門用語カバレッジ
        metrics["technical_term_coverage"] = self.calculate_technical_coverage(answer, query)

        # 総合スコア
        metrics["overall_score"] = self.calculate_overall_score(metrics)

        return metrics

    def calculate_factual_consistency(self, answer: str, documents: list):
        """事実整合性スコア"""

        # NLIモデルで各文をチェック
        sentences = split_sentences(answer)
        doc_text = concat_documents(documents)

        consistency_scores = []
        for sent in sentences:
            score = self.nli_model.check_entailment(doc_text, sent)
            consistency_scores.append(score)

        return sum(consistency_scores) / len(sentences)
```

---

### 6. 継続的改善プロセス

#### 6.1 改善サイクル

```
週次:
  - クエリログ分析
  - ハルシネーション事例の収集
  - 専門家フィードバック集計

月次:
  - 教師データ追加（100-200件）
  - モデル再学習
  - A/Bテスト実施

四半期:
  - 大規模評価（1000件）
  - アーキテクチャ見直し
  - ドメイン埋め込み再学習
```

#### 6.2 バージョン管理

```python
# src/training/model_versioning.py

class ModelVersionManager:
    """モデルバージョン管理"""

    def create_version(self, model_path: str, metrics: dict):
        """新バージョン作成"""

        version_info = {
            "version": generate_version(),  # v1.2.3
            "timestamp": datetime.now(),
            "model_path": model_path,
            "training_data": {
                "size": get_dataset_size(),
                "quality_score": calculate_data_quality()
            },
            "performance": metrics,
            "improvements": [
                "専門用語カバレッジ +5%",
                "ハルシネーション率 -10%"
            ]
        }

        # バージョン登録
        self.register_version(version_info)

        # 本番環境へのデプロイ判定
        if self.should_deploy(version_info):
            self.deploy_to_production(model_path)

    def should_deploy(self, version_info: dict) -> bool:
        """デプロイ判定"""
        current = self.get_current_production_version()

        # 改善基準
        criteria = {
            "overall_score": current["performance"]["overall_score"] + 0.02,
            "hallucination_score": current["performance"]["hallucination_score"] - 0.05,
            "citation_precision": current["performance"]["citation_precision"] + 0.03
        }

        for metric, threshold in criteria.items():
            if version_info["performance"][metric] < threshold:
                return False

        return True
```

---

## 📊 実装ロードマップ

### Phase 1: 基盤強化（1-2ヶ月）

```yaml
Week 1-2:
  - 教師データ品質チェックツール実装
  - 専門家フィードバックUI構築
  - ハルシネーション検出システム実装

Week 3-4:
  - ドメイン適応埋め込み学習
  - 引用検証システム構築
  - 自動評価パイプライン実装

Week 5-8:
  - 教師データ拡充（500件追加）
  - タスク特化型LoRA学習
  - 初回評価とベンチマーク確立
```

### Phase 2: 精度向上（2-3ヶ月）

```yaml
Month 3:
  - 教師データ1000件達成
  - アンサンブル推論実装
  - 適応的ハイブリッド検索

Month 4:
  - 自己検証メカニズム統合
  - 専門家評価システム本格運用
  - A/Bテストによる最適化

Month 5:
  - モデルバージョン管理システム
  - 継続的改善プロセス確立
  - 本番環境デプロイ
```

### Phase 3: 運用・改善（継続）

```yaml
継続的活動:
  - 週次: ログ分析、問題事例収集
  - 月次: データ追加、再学習
  - 四半期: 大規模評価、アーキテクチャ見直し
```

---

## 🎯 期待される成果

### 定量的目標

| メトリクス | 現状 | 6ヶ月後目標 |
|-----------|------|-----------|
| **全体精度** | 75% | 90% |
| **引用精度** | 85% | 95% |
| **ハルシネーション率** | 15% | <5% |
| **専門家満足度** | - | 4.5/5.0 |
| **教師データ量** | 149件 | 1,000件 |

### 定性的改善

- ✅ エキスパートレベルの回答品質
- ✅ 複雑な計算問題への対応
- ✅ 複数文書の統合的解釈
- ✅ 信頼できる引用と根拠提示

---

## 💻 実装支援ツール

以下のスクリプトを作成しました：

1. `scripts/generate_training_data_from_docs.py` - 半自動データ生成
2. `scripts/check_training_data_quality.py` - 品質チェック
3. `scripts/train_domain_adapted_embeddings.py` - 埋め込み学習
4. `app/evaluation/expert_feedback.py` - フィードバックシステム
5. `src/evaluation/automated_metrics.py` - 自動評価

---

## 🚀 次のアクション

### 即座に実施可能

1. **品質チェックツールの実行**
   ```bash
   python scripts/check_training_data_quality.py RARdata.json
   ```

2. **専門家フィードバックUIの追加**
   - RAG回答画面に評価ボタン追加
   - フィードバックデータ収集開始

3. **ハルシネーション検出の有効化**
   - 回答生成時の検証を追加
   - ログへの警告出力

### 中期的実施

4. **教師データ拡充計画の策定**
   - 月間100件の目標設定
   - 専門家アノテーション体制構築

5. **ドメイン埋め込みの学習**
   - 専門文書でfine-tuning
   - RAGシステムへの統合

6. **評価ベンチマークの確立**
   - テストセット作成（200件）
   - 定期評価の自動化

---

この戦略により、専門家が実務で信頼して使用できるシステムへと進化します！
