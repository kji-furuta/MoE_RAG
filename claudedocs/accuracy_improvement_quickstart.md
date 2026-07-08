# 精度向上 - クイックスタートガイド

## 🚀 即座に実施可能な改善（今日から）

### 1. フィードバック収集の開始

#### UIに評価ボタンを追加

[templates/index.html](templates/index.html) のRAG回答表示部分に以下を追加：

```html
<!-- RAG回答表示の下に追加 -->
<div id="feedback-section" style="margin-top: 20px; padding: 15px; border-top: 2px solid #eee;">
    <h4>📊 この回答の評価</h4>
    <div style="display: flex; gap: 10px; margin: 10px 0;">
        <button onclick="submitFeedback('excellent')" class="btn" style="background: #27ae60;">
            ✅ 正確
        </button>
        <button onclick="submitFeedback('good')" class="btn" style="background: #3498db;">
            👍 概ね良好
        </button>
        <button onclick="submitFeedback('poor')" class="btn" style="background: #e74c3c;">
            ❌ 不正確
        </button>
    </div>

    <div id="feedback-details" style="display: none; margin-top: 10px;">
        <h5>詳細フィードバック（任意）</h5>
        <div style="margin: 10px 0;">
            <label><input type="checkbox" id="fb-hallucination"> ハルシネーション（事実と異なる）</label><br>
            <label><input type="checkbox" id="fb-incomplete"> 情報不足</label><br>
            <label><input type="checkbox" id="fb-wrong-citation"> 引用が不適切</label><br>
            <label><input type="checkbox" id="fb-calculation-error"> 計算エラー</label>
        </div>
        <textarea id="fb-comment" placeholder="具体的なコメント（任意）" style="width: 100%; min-height: 80px;"></textarea>
        <button onclick="submitDetailedFeedback()" class="btn">送信</button>
    </div>
</div>

<script>
function submitFeedback(rating) {
    // フィードバック送信
    fetch('/api/rag/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: currentQuery,
            answer: currentAnswer,
            rating: rating,
            timestamp: new Date().toISOString()
        })
    });

    // 詳細フィードバックフォーム表示
    if (rating === 'poor') {
        document.getElementById('feedback-details').style.display = 'block';
    } else {
        alert('フィードバックありがとうございます！');
    }
}

function submitDetailedFeedback() {
    const feedback = {
        query: currentQuery,
        answer: currentAnswer,
        issues: {
            hallucination: document.getElementById('fb-hallucination').checked,
            incomplete: document.getElementById('fb-incomplete').checked,
            wrong_citation: document.getElementById('fb-wrong-citation').checked,
            calculation_error: document.getElementById('fb-calculation-error').checked
        },
        comment: document.getElementById('fb-comment').value,
        timestamp: new Date().toISOString()
    };

    fetch('/api/rag/feedback/detailed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(feedback)
    });

    alert('詳細フィードバックを送信しました。ありがとうございます！');
    document.getElementById('feedback-details').style.display = 'none';
}
</script>
```

#### バックエンドAPI追加

```python
# app/main_unified.py に追加

@app.post("/api/rag/feedback")
async def save_feedback(feedback: dict):
    """フィードバック保存"""
    import json
    from datetime import datetime

    feedback_file = "data/feedback/rag_feedback.jsonl"
    os.makedirs("data/feedback", exist_ok=True)

    with open(feedback_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback, ensure_ascii=False) + '\n')

    return {"status": "success", "message": "フィードバックを保存しました"}

@app.post("/api/rag/feedback/detailed")
async def save_detailed_feedback(feedback: dict):
    """詳細フィードバック保存"""
    import json

    detailed_file = "data/feedback/detailed_feedback.jsonl"
    os.makedirs("data/feedback", exist_ok=True)

    with open(detailed_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback, ensure_ascii=False) + '\n')

    # 不正確な回答は専門家レビュー用にフラグ
    if any(feedback['issues'].values()):
        review_file = "data/feedback/needs_expert_review.jsonl"
        with open(review_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + '\n')

    return {"status": "success", "message": "詳細フィードバックを保存しました"}
```

---

### 2. ハルシネーション検出の有効化

#### 簡易検出システム

```python
# src/rag/validation/simple_hallucination_detector.py

import re
from typing import List, Dict

class SimpleHallucinationDetector:
    """簡易ハルシネーション検出"""

    def detect(self, answer: str, documents: List[str]) -> Dict:
        """ハルシネーション検出"""

        issues = []
        doc_text = " ".join(documents)

        # 1. 数値チェック
        answer_numbers = self.extract_numbers(answer)
        doc_numbers = self.extract_numbers(doc_text)

        for num in answer_numbers:
            if num not in doc_numbers:
                issues.append({
                    "type": "number_mismatch",
                    "value": num,
                    "severity": "high"
                })

        # 2. 固有名詞チェック（簡易版）
        answer_entities = self.extract_technical_terms(answer)
        doc_entities = self.extract_technical_terms(doc_text)

        for entity in answer_entities:
            if entity not in doc_entities:
                issues.append({
                    "type": "entity_mismatch",
                    "value": entity,
                    "severity": "medium"
                })

        # 3. 確信度の低い表現チェック
        uncertain_phrases = ["おそらく", "かもしれません", "と思われます", "推測"]
        for phrase in uncertain_phrases:
            if phrase in answer:
                issues.append({
                    "type": "uncertainty",
                    "value": phrase,
                    "severity": "low"
                })

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "confidence": 1.0 - (len(issues) * 0.1)  # 簡易スコア
        }

    def extract_numbers(self, text: str) -> set:
        """数値抽出"""
        # 整数と小数
        numbers = re.findall(r'\d+\.?\d*', text)
        return set(numbers)

    def extract_technical_terms(self, text: str) -> set:
        """専門用語抽出（簡易版）"""
        technical_patterns = [
            r'第[1-9]種',
            r'第[1-9]級',
            r'\d+km/h',
            r'\d+%',
            r'\d+メートル',
            r'\d+m'
        ]
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        return terms
```

#### RAGシステムに統合

```python
# src/rag/core/query_engine.py に追加

from src.rag.validation.simple_hallucination_detector import SimpleHallucinationDetector

class QueryEngine:
    def __init__(self):
        # ... 既存の初期化 ...
        self.hallucination_detector = SimpleHallucinationDetector()

    def query(self, query: str, top_k: int = 5):
        # ... 既存の処理 ...

        # 回答生成後にハルシネーション検出
        detection_result = self.hallucination_detector.detect(
            answer=final_answer,
            documents=[doc['content'] for doc in retrieved_docs]
        )

        # ログに警告を記録
        if detection_result['has_issues']:
            logger.warning(f"Potential hallucination detected: {detection_result['issues']}")

            # 信頼度が低い場合は警告を追加
            if detection_result['confidence'] < 0.7:
                final_answer += "\n\n⚠️ 注意: この回答には検証が必要な可能性があります。"

        return {
            "answer": final_answer,
            "documents": retrieved_docs,
            "hallucination_check": detection_result
        }
```

---

### 3. クエリログ分析

#### ログ収集の開始

```python
# app/main_unified.py のRAGエンドポイントに追加

import json
from datetime import datetime

@app.post("/rag/query")
async def rag_query(request: dict):
    # ... 既存の処理 ...

    # クエリログ保存
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": request['query'],
        "answer": result['answer'],
        "top_k": request.get('top_k', 5),
        "documents_used": len(result['documents']),
        "hallucination_score": result.get('hallucination_check', {}).get('confidence', 1.0)
    }

    log_file = "data/logs/rag_queries.jsonl"
    os.makedirs("data/logs", exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    return result
```

#### 週次分析スクリプト

```python
# scripts/analyze_query_logs.py

import json
from collections import Counter
from datetime import datetime, timedelta

def analyze_weekly_logs(log_file: str = "data/logs/rag_queries.jsonl"):
    """週次クエリログ分析"""

    # 過去7日間のログ読み込み
    week_ago = datetime.now() - timedelta(days=7)
    logs = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            log = json.loads(line)
            if datetime.fromisoformat(log['timestamp']) >= week_ago:
                logs.append(log)

    print(f"📊 過去7日間のクエリ分析 ({len(logs)}件)")
    print("=" * 60)

    # ハルシネーション傾向
    low_confidence = [l for l in logs if l.get('hallucination_score', 1.0) < 0.7]
    print(f"\n⚠️  低信頼度回答: {len(low_confidence)}件 ({len(low_confidence)/len(logs)*100:.1f}%)")

    # よくある質問パターン
    queries = [l['query'] for l in logs]
    print(f"\n🔍 頻出クエリパターン:")
    # 簡易的なパターン抽出
    for query in Counter(queries).most_common(5):
        print(f"  - {query[0][:50]}... ({query[1]}回)")

    # 改善候補
    print(f"\n💡 改善候補:")
    if low_confidence:
        print(f"  1. 信頼度の低い{len(low_confidence)}件の回答を専門家レビュー")
        print(f"  2. 頻出質問を教師データに追加")

if __name__ == "__main__":
    analyze_weekly_logs()
```

---

## 📈 1週間後の確認事項

### データ収集状況
```bash
# フィードバック収集数
wc -l data/feedback/rag_feedback.jsonl

# 専門家レビュー待ち
wc -l data/feedback/needs_expert_review.jsonl

# クエリログ
wc -l data/logs/rag_queries.jsonl
```

### 分析実行
```bash
# 週次ログ分析
python scripts/analyze_query_logs.py

# 品質チェック（改善後）
python scripts/check_training_data_quality.py RARdata_updated.json
```

---

## 🎯 1ヶ月後の目標

| 指標 | 目標値 |
|------|-------|
| **フィードバック収集** | 100件以上 |
| **専門家レビュー** | 20件以上 |
| **新規教師データ** | +50件 |
| **ハルシネーション検出** | 全クエリで実施 |
| **モデル再学習** | 1回実施 |

---

## 🚀 次のステップ（優先順位順）

### 優先度: 高（今週中）
1. ✅ フィードバックUI実装
2. ✅ ハルシネーション検出統合
3. ✅ クエリログ収集開始

### 優先度: 中（今月中）
4. 専門家レビュープロセス確立
5. 教師データ50件追加
6. モデル再学習（1回目）

### 優先度: 低（来月以降）
7. ドメイン埋め込み学習
8. タスク特化型LoRA
9. アンサンブル推論

---

## 💻 実装コマンド

### フィードバックシステム起動
```bash
# UIコード更新
# templates/index.html を編集

# サーバー再起動
docker-compose restart ai-ft
```

### ハルシネーション検出有効化
```bash
# 検出器作成
mkdir -p src/rag/validation
# simple_hallucination_detector.py を作成

# RAGエンジン更新
# src/rag/core/query_engine.py を編集

# テスト
docker exec ai-ft-container python3 -c "
from src.rag.validation.simple_hallucination_detector import SimpleHallucinationDetector
detector = SimpleHallucinationDetector()
result = detector.detect('設計速度は100km/hです', ['設計速度80km/hの場合'])
print(result)
"
```

### 週次分析実行
```bash
# 分析スクリプト実行
python scripts/analyze_query_logs.py

# 結果をレポート保存
python scripts/analyze_query_logs.py > reports/weekly_$(date +%Y%m%d).txt
```

---

## 📞 サポート

質問や問題がある場合:
1. [accuracy_improvement_strategy.md](accuracy_improvement_strategy.md) - 完全戦略
2. `data/feedback/` - フィードバックデータ確認
3. `data/logs/` - クエリログ確認

この精度向上プロセスにより、システムは継続的に改善されます！
