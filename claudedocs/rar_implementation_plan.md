# RAR形式学習データ実装計画

## 概要
検索拡張推論（Retrieval-Augmented Reasoning: RAR）形式への段階的移行計画

## フェーズ1: パイロット実験（1-2ヶ月）

### 目標
- RAR形式の有効性を小規模（100件）で検証
- データ生成パイプラインの確立
- ベースライン性能の測定

### 実装ステップ

#### Step 1: ゴールドスタンダードデータ作成（2週間）
- 道路構造令の一部（第15条-第20条）に絞る
- 専門家が手動で100件のRAR形式データを作成
- 各データに以下を含める：
  - 質問（instruction）
  - 関連文書2-3件（正解1件+ディストラクタ1-2件）
  - 思考プロセス（chain_of_thought）
  - 最終回答（final_answer）
  - 逐語的引用（citations）

**サンプルデータ構造**:
```json
{
  "id": "pilot-001",
  "instruction": "設計速度80km/hの道路で、最小曲線半径はいくつですか？",
  "documents": [
    {
      "source": "道路構造令 第15条",
      "is_oracle": true,
      "content": "車道の屈曲部には、当該道路の設計速度に応じ、次の表の右欄に掲げる値以上とするものとする。設計速度80km/h: 280m"
    },
    {
      "source": "道路橋示方書 耐風設計編",
      "is_oracle": false,
      "content": "橋梁の設計風速は10分間平均風速として..."
    }
  ],
  "output": {
    "chain_of_thought": "設計速度80km/hの最小曲線半径を求める。道路構造令第15条の表を確認すると、設計速度80km/hに対応する曲線半径は280mである。道路橋示方書は橋梁の耐風設計に関する文書であり、今回の質問には無関係。",
    "final_answer": "設計速度80km/hの道路における最小曲線半径は280mです。",
    "citations": [
      {
        "source": "道路構造令 第15条",
        "quote": "設計速度80km/h: 280m"
      }
    ]
  }
}
```

#### Step 2: 自動データ生成パイプライン構築（3週間）

**使用ツール**: GPT-4o (教師モデル)

```python
# データ生成プロンプト例
TEACHER_PROMPT = """
あなたはベテランの土木設計技師です。
以下の文書チャンクから、実務的な質問を作成し、思考過程とともに回答してください。

【文書】
{document_content}

【出力形式】
1. 質問（instruction）: 若手技術者が実務で遭遇しそうな具体的な質問
2. 思考プロセス（chain_of_thought）: 文書を読み解き、答えに至る論理的ステップ
3. 最終回答（final_answer）: 簡潔で正確な回答
4. 引用（citations）: 回答の根拠となる文書の逐語的引用

【重要】
- 思考プロセスは接続詞を用いた自然な文章で記述
- 引用は原文から一字一句そのまま抜き出す
- 質問は複雑な応用問題を含める
"""
```

**パイプライン構成**:
1. PDF → OCR → テキスト抽出（Azure AI Document Intelligence）
2. セマンティック・チャンキング（条文ごとに分割）
3. GPT-4oによる質問・CoT・回答生成
4. 品質フィルタリング（引用の一致確認）
5. 専門家レビュー（5%サンプリング）

#### Step 3: 小規模モデル検証（3週間）

**検証環境**:
- ベースモデル: DeepSeek-R1-Distill-Qwen-7B（軽量版）
- 学習手法: QLoRA（4bit量子化）
- GPU: A100 x1

**比較実験**:
- 条件A: 現行Alpaca形式（100件）
- 条件B: RAR形式（100件）
- 条件C: RAR形式 + ディストラクタ（100件）

**評価指標**:
1. 回答精度（Accuracy）
2. 引用正確性（Citation Precision）
3. ハルシネーション率
4. 思考プロセスの一貫性（GPT-4でスコアリング）

## フェーズ2: パイプライン拡張（3-4ヶ月）

### 目標
- データ生成の完全自動化
- 1,000件規模のRAR形式データセット構築
- 継続的事前学習（CPT）の実施

### 実装内容

#### データ生成の自動化
```python
# data_generation_pipeline.py
class RARDataGenerator:
    def __init__(self, teacher_model="gpt-4o"):
        self.teacher = teacher_model
        self.ocr_client = AzureDocumentIntelligence()

    def process_document(self, pdf_path):
        # 1. OCR処理
        chunks = self.ocr_client.extract_chunks(pdf_path)

        # 2. 質問生成
        qa_pairs = []
        for chunk in chunks:
            # 正解文書として使用
            oracle_doc = {
                "source": chunk.metadata["source"],
                "is_oracle": True,
                "content": chunk.text
            }

            # ディストラクタ選定（ベクトル類似度で関連性が低い文書）
            distractor_docs = self.select_distractors(chunk, n=2)

            # GPT-4oで生成
            rar_data = self.generate_rar_format(
                oracle=oracle_doc,
                distractors=distractor_docs
            )

            qa_pairs.append(rar_data)

        return qa_pairs

    def generate_rar_format(self, oracle, distractors):
        documents = [oracle] + distractors

        prompt = f"""
        質問、思考プロセス、回答、引用を生成してください。

        【文書】
        {json.dumps(documents, ensure_ascii=False, indent=2)}

        【出力JSON形式】
        {{
          "instruction": "質問文",
          "output": {{
            "chain_of_thought": "段階的な推論",
            "final_answer": "最終回答",
            "citations": [...]
          }}
        }}
        """

        response = self.teacher.generate(prompt)
        return json.loads(response)
```

#### 継続的事前学習（CPT）

**D-CPT Law適用**:
```python
# D-CPT Lawによる最適化
# L(N,D,r) = E + A/N^α + B·r^η/D^β + C/r'^γ

def optimize_training_params(
    model_size_n: int,
    domain_data_size: int,
    general_data_size: int
):
    # フェーズ2: 経験的データ収集
    experimental_configs = [
        {"N": 7e9, "D": 1e9, "r": 0.5},
        {"N": 7e9, "D": 5e9, "r": 0.7},
        {"N": 7e9, "D": 10e9, "r": 0.9},
    ]

    losses = []
    for config in experimental_configs:
        loss = run_small_experiment(config)
        losses.append((config, loss))

    # フェーズ3: パラメータ推定
    params = fit_dcpt_law(losses)

    # フェーズ4: 最適配分の予測
    optimal_r = predict_optimal_ratio(
        params,
        model_size_n,
        domain_data_size
    )

    return optimal_r
```

## フェーズ3: 本番展開（5-6ヶ月）

### 目標
- 15,000件の全データ処理
- 32Bモデルでの本格学習
- RAGシステムとの完全統合

### 実装内容

#### 大規模データ生成
- 社内1.5万件データの完全処理
- 品質管理の自動化
- 専門家レビューの効率化

#### モデル学習
```yaml
training_config:
  base_model: "DeepSeek-R1-Distill-Qwen-32B"
  method: "QLoRA"
  gpu: "A100 80GB x4"

  data:
    domain_data: 15000  # RAR形式
    general_data: "SlimPajama (サンプリング)"
    mix_ratio: 0.8  # D-CPT Lawで最適化

  hyperparameters:
    learning_rate: 2e-5
    batch_size: 4
    gradient_accumulation: 8
    lora_r: 64
    lora_alpha: 128
```

#### RAG統合
```python
# rag_rar_integration.py
class RAREnhancedRAG:
    def __init__(self, rar_trained_model, vector_store):
        self.model = rar_trained_model
        self.vs = vector_store

    def query(self, question):
        # 1. ハイブリッド検索
        docs = self.vs.hybrid_search(question, top_k=5)

        # 2. RAR形式プロンプト構築
        prompt = self.build_rar_prompt(question, docs)

        # 3. CoT付き回答生成
        response = self.model.generate(prompt)

        # 4. 引用抽出と検証
        citations = self.extract_citations(response)

        return {
            "answer": response["final_answer"],
            "chain_of_thought": response["chain_of_thought"],
            "citations": citations
        }

    def build_rar_prompt(self, question, documents):
        return f"""
        以下の文書を読み、質問に段階的に答えてください。

        【質問】
        {question}

        【文書】
        {self.format_documents(documents)}

        【回答形式】
        1. 思考プロセス: 文書をどう読み解いたか
        2. 最終回答: 簡潔な答え
        3. 引用: 根拠となる文書の逐語的引用
        """
```

## 評価計画

### 定量評価
| 指標 | 測定方法 | 目標値 |
|------|---------|--------|
| 回答精度 | 専門家評価との一致率 | >90% |
| 引用正確性 | 引用箇所の検証 | >95% |
| ハルシネーション率 | 虚偽情報の検出 | <5% |
| CoT一貫性 | GPT-4スコアリング | >0.8 |

### 定性評価
- ベテラン技術者によるブラインドテスト
- 「新人教育に使えるか」の実用性評価
- 「最終チェック補助として信頼できるか」の評価

## リソース要件

### 計算リソース
- **フェーズ1**: A100 x1（1週間）
- **フェーズ2**: A100 x4（1ヶ月）
- **フェーズ3**: A100 x8（2ヶ月）

### 人的リソース
- データサイエンティスト: 2名
- 土木設計専門家（レビュー）: 1名（週10時間）
- MLエンジニア: 2名

### データ生成コスト
- GPT-4o API使用料: 約$10,000（15,000件生成）
- Azure Document Intelligence: 約$5,000

## リスクと対策

| リスク | 影響度 | 対策 |
|--------|--------|------|
| GPT-4生成データの品質不足 | 高 | 専門家レビュー5%、品質フィルタリング |
| 計算リソース不足 | 中 | クラウドGPU（AWS/Azure）の一時利用 |
| 壊滅的忘却 | 中 | D-CPT Lawで混合比率最適化 |

## 成功基準

### フェーズ1成功基準
- RAR形式が現行Alpaca形式を上回る性能
- データ生成パイプラインの確立
- R² > 0.97（D-CPT Law適合度）

### フェーズ2成功基準
- 1,000件データセットの品質検証
- CPTによるドメイン理解の向上
- ハルシネーション率 < 10%

### フェーズ3成功基準
- 15,000件完全処理
- 実運用レベルの性能（精度>90%）
- 専門家による実用性承認

## 次のステップ

1. **即座に開始**: フェーズ1のゴールドスタンダードデータ作成
2. **並行作業**: データ生成パイプラインの実装
3. **定期レビュー**: 2週間ごとの進捗確認と方針調整
