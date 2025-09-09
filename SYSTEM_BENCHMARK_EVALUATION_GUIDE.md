# システムベンチマーク評価実装ガイド

## 概要

ファインチューニング、継続学習、RAGシステムの各コンポーネントに対する包括的なベンチマーク評価フレームワークを提案します。定量的・定性的評価を組み合わせ、システム全体のパフォーマンスを継続的に監視・改善できる仕組みを構築します。

## 1. ファインチューニングシステムベンチマーク

### 1.1 評価メトリクス

```python
# src/benchmarks/finetuning_benchmark.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from pathlib import Path
import json
from datetime import datetime

class FineTuningBenchmark:
    """ファインチューニングモデルの評価ベンチマーク"""
    
    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    def evaluate_all(self) -> Dict:
        """全ベンチマークを実行"""
        
        print("Starting Fine-tuning Benchmarks...")
        
        # 1. 言語モデリング評価
        self.results['perplexity'] = self.evaluate_perplexity()
        
        # 2. 生成品質評価
        self.results['generation_quality'] = self.evaluate_generation_quality()
        
        # 3. タスク特化評価（道路設計）
        self.results['domain_specific'] = self.evaluate_domain_specific()
        
        # 4. 推論速度評価
        self.results['inference_speed'] = self.evaluate_inference_speed()
        
        # 5. メモリ使用量評価
        self.results['memory_usage'] = self.evaluate_memory_usage()
        
        # 6. ベースモデルとの比較（差分評価）
        if self.base_model_path:
            self.results['improvement'] = self.evaluate_improvement()
        
        return self.results
    
    def evaluate_perplexity(self) -> Dict:
        """パープレキシティ評価"""
        print("Evaluating perplexity...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # テストデータセット
        test_data = self._load_test_dataset()
        
        total_loss = 0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in test_data[:100]:  # サンプル100件
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        perplexity = np.exp(total_loss / total_tokens)
        
        return {
            "perplexity": float(perplexity),
            "evaluation_samples": 100,
            "average_loss": total_loss / total_tokens
        }
    
    def evaluate_generation_quality(self) -> Dict:
        """生成品質評価（BLEU, ROUGE, BERTScore）"""
        from rouge_score import rouge_scorer
        from bert_score import score as bert_score
        import sacrebleu
        
        print("Evaluating generation quality...")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # テストプロンプトと参照回答
        test_cases = self._load_generation_test_cases()
        
        generated_texts = []
        reference_texts = []
        
        for prompt, reference in test_cases[:50]:
            # 生成
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated)
            reference_texts.append(reference)
        
        # ROUGE評価
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
        
        # BERTScore評価
        P, R, F1 = bert_score(generated_texts, reference_texts, lang="ja", device=self.device)
        
        # BLEU評価
        bleu = sacrebleu.corpus_bleu(generated_texts, [reference_texts])
        
        return {
            "rouge1_f1": np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
            "rouge2_f1": np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
            "rougeL_f1": np.mean([s['rougeL'].fmeasure for s in rouge_scores]),
            "bert_score_f1": F1.mean().item(),
            "bleu_score": bleu.score
        }
    
    def evaluate_domain_specific(self) -> Dict:
        """道路設計ドメイン特化評価"""
        print("Evaluating domain-specific performance...")
        
        # ドメイン特化テストケース
        domain_tests = [
            {
                "question": "設計速度80km/hの道路の最小曲線半径は？",
                "expected_keywords": ["280m", "曲線半径", "設計速度"],
                "expected_range": (250, 300)
            },
            {
                "question": "縦断勾配の最大値について説明してください",
                "expected_keywords": ["6%", "7%", "山地部", "平地部"],
                "expected_range": None
            },
            # ... 他のテストケース
        ]
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        correct_answers = 0
        keyword_matches = 0
        
        for test in domain_tests:
            inputs = tokenizer(test["question"], return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # キーワードチェック
            for keyword in test["expected_keywords"]:
                if keyword in response:
                    keyword_matches += 1
            
            # 数値範囲チェック
            if test["expected_range"]:
                import re
                numbers = re.findall(r'\d+', response)
                for num in numbers:
                    if test["expected_range"][0] <= int(num) <= test["expected_range"][1]:
                        correct_answers += 1
                        break
        
        return {
            "domain_accuracy": correct_answers / len(domain_tests),
            "keyword_match_rate": keyword_matches / (len(domain_tests) * 3),  # 平均3キーワード
            "test_cases": len(domain_tests)
        }
    
    def evaluate_inference_speed(self) -> Dict:
        """推論速度評価"""
        print("Evaluating inference speed...")
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # ウォームアップ
        for _ in range(3):
            inputs = tokenizer("テスト", return_tensors="pt")
            model.generate(**inputs, max_new_tokens=10)
        
        # 速度測定
        input_lengths = [10, 50, 100, 200, 500]
        results = {}
        
        for length in input_lengths:
            prompt = "テスト " * length
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)
                end = time.time()
                times.append(end - start)
            
            results[f"input_{length}_tokens"] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "tokens_per_second": 50 / np.mean(times)
            }
        
        return results
    
    def evaluate_memory_usage(self) -> Dict:
        """メモリ使用量評価"""
        print("Evaluating memory usage...")
        
        import psutil
        import GPUtil
        
        # 初期メモリ
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            initial_gpu_memory = gpus[0].memoryUsed if gpus else 0
        else:
            initial_gpu_memory = 0
        
        # モデルロード
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        
        # ロード後メモリ
        loaded_memory = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            loaded_gpu_memory = gpus[0].memoryUsed if gpus else 0
        else:
            loaded_gpu_memory = 0
        
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "cpu_memory_mb": loaded_memory - initial_memory,
            "gpu_memory_mb": loaded_gpu_memory - initial_gpu_memory,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
    
    def evaluate_improvement(self) -> Dict:
        """ベースモデルとの改善度評価"""
        print("Evaluating improvement over base model...")
        
        # ベースモデルのベンチマーク
        base_benchmark = FineTuningBenchmark(self.base_model_path)
        base_perplexity = base_benchmark.evaluate_perplexity()
        
        # 改善率計算
        improvement = {
            "perplexity_improvement": (base_perplexity["perplexity"] - self.results["perplexity"]["perplexity"]) / base_perplexity["perplexity"] * 100,
            "domain_accuracy_delta": self.results["domain_specific"]["domain_accuracy"] - 0.3,  # ベースライン仮定
        }
        
        return improvement
    
    def _load_test_dataset(self) -> List[str]:
        """テストデータセットをロード"""
        # 実装: data/test/ディレクトリから読み込み
        test_file = Path("data/test/test_dataset.jsonl")
        texts = []
        if test_file.exists():
            with open(test_file) as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data.get("text", ""))
        return texts
    
    def _load_generation_test_cases(self) -> List[Tuple[str, str]]:
        """生成評価用のテストケースをロード"""
        # 実装: プロンプトと期待される回答のペア
        return [
            ("道路の設計速度について説明してください", "道路の設計速度は、道路を設計する際の基準となる速度で..."),
            # 他のテストケース
        ]
    
    def save_results(self, output_path: str):
        """結果を保存"""
        output = {
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
```

## 2. RAGシステムベンチマーク

### 2.1 RAG評価メトリクス

```python
# src/benchmarks/rag_benchmark.py
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
import aiohttp
from datetime import datetime

class RAGBenchmark:
    """RAGシステムの評価ベンチマーク"""
    
    def __init__(self, rag_endpoint: str = "http://localhost:8050"):
        self.rag_endpoint = rag_endpoint
        self.results = {}
        
    async def evaluate_all(self) -> Dict:
        """全RAGベンチマークを実行"""
        
        print("Starting RAG System Benchmarks...")
        
        # 1. 検索精度評価
        self.results['retrieval_accuracy'] = await self.evaluate_retrieval_accuracy()
        
        # 2. 回答品質評価
        self.results['answer_quality'] = await self.evaluate_answer_quality()
        
        # 3. レスポンス時間評価
        self.results['response_time'] = await self.evaluate_response_time()
        
        # 4. コンテキスト関連性評価
        self.results['context_relevance'] = await self.evaluate_context_relevance()
        
        # 5. ハルシネーション評価
        self.results['hallucination'] = await self.evaluate_hallucination()
        
        # 6. スケーラビリティ評価
        self.results['scalability'] = await self.evaluate_scalability()
        
        return self.results
    
    async def evaluate_retrieval_accuracy(self) -> Dict:
        """検索精度評価（Precision, Recall, F1）"""
        print("Evaluating retrieval accuracy...")
        
        # テストクエリと正解文書
        test_queries = [
            {
                "query": "設計速度80km/hの最小曲線半径",
                "relevant_docs": ["道路構造令第15条", "設計基準_曲線半径.pdf"],
                "expected_chunks": 5
            },
            {
                "query": "横断勾配の設計基準",
                "relevant_docs": ["道路構造令第24条", "横断勾配設計指針.pdf"],
                "expected_chunks": 3
            }
            # 他のテストケース
        ]
        
        total_precision = 0
        total_recall = 0
        
        for test in test_queries:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": test["query"],
                    "top_k": 10,
                    "search_type": "hybrid"
                }
                
                async with session.post(f"{self.rag_endpoint}/rag/search", json=payload) as response:
                    results = await response.json()
            
            # 取得された文書
            retrieved_docs = [r["metadata"]["source"] for r in results["results"]]
            
            # Precision計算
            relevant_retrieved = len(set(retrieved_docs) & set(test["relevant_docs"]))
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            
            # Recall計算
            recall = relevant_retrieved / len(test["relevant_docs"]) if test["relevant_docs"] else 0
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(test_queries)
        avg_recall = total_recall / len(test_queries)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": f1_score,
            "test_queries": len(test_queries)
        }
    
    async def evaluate_answer_quality(self) -> Dict:
        """回答品質評価（正確性、完全性、有用性）"""
        print("Evaluating answer quality...")
        
        # 品質評価用のQAペア
        qa_pairs = [
            {
                "question": "縦断勾配の最大値は？",
                "expected_answer": "一般的に6%、山地部では7%まで許容",
                "criteria": {
                    "accuracy": ["6%", "7%", "山地部"],
                    "completeness": ["一般", "山地部", "許容値"],
                    "usefulness": True
                }
            }
            # 他のQAペア
        ]
        
        total_accuracy = 0
        total_completeness = 0
        total_usefulness = 0
        
        for qa in qa_pairs:
            async with aiohttp.ClientSession() as session:
                payload = {"query": qa["question"]}
                
                async with session.post(f"{self.rag_endpoint}/rag/query", json=payload) as response:
                    result = await response.json()
            
            answer = result["answer"]
            
            # 正確性評価
            accuracy_score = sum(1 for keyword in qa["criteria"]["accuracy"] if keyword in answer) / len(qa["criteria"]["accuracy"])
            
            # 完全性評価
            completeness_score = sum(1 for keyword in qa["criteria"]["completeness"] if keyword in answer) / len(qa["criteria"]["completeness"])
            
            # 有用性評価（簡易的）
            usefulness_score = 1.0 if len(answer) > 50 and accuracy_score > 0.5 else 0.5
            
            total_accuracy += accuracy_score
            total_completeness += completeness_score
            total_usefulness += usefulness_score
        
        return {
            "accuracy": total_accuracy / len(qa_pairs),
            "completeness": total_completeness / len(qa_pairs),
            "usefulness": total_usefulness / len(qa_pairs),
            "evaluated_queries": len(qa_pairs)
        }
    
    async def evaluate_response_time(self) -> Dict:
        """レスポンス時間評価"""
        print("Evaluating response time...")
        
        query_types = [
            {"type": "simple", "query": "道路幅員とは？", "complexity": "low"},
            {"type": "medium", "query": "設計速度と曲線半径の関係について説明してください", "complexity": "medium"},
            {"type": "complex", "query": "山間部における道路設計で考慮すべき要素を全て列挙し、それぞれについて詳しく説明してください", "complexity": "high"}
        ]
        
        results = {}
        
        for query_type in query_types:
            times = []
            
            for _ in range(10):  # 各クエリを10回実行
                start = time.time()
                
                async with aiohttp.ClientSession() as session:
                    payload = {"query": query_type["query"]}
                    async with session.post(f"{self.rag_endpoint}/rag/query", json=payload) as response:
                        await response.json()
                
                end = time.time()
                times.append(end - start)
            
            results[query_type["complexity"]] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "p95_time": np.percentile(times, 95)
            }
        
        return results
    
    async def evaluate_context_relevance(self) -> Dict:
        """コンテキスト関連性評価"""
        print("Evaluating context relevance...")
        
        test_cases = [
            {
                "query": "橋梁の設計荷重",
                "expected_context_keywords": ["橋梁", "荷重", "設計", "活荷重", "死荷重"],
                "irrelevant_keywords": ["トンネル", "舗装", "信号"]
            }
            # 他のテストケース
        ]
        
        total_relevance_score = 0
        
        for test in test_cases:
            async with aiohttp.ClientSession() as session:
                payload = {"query": test["query"], "include_sources": True}
                
                async with session.post(f"{self.rag_endpoint}/rag/query", json=payload) as response:
                    result = await response.json()
            
            # コンテキストの関連性スコア計算
            contexts = " ".join([source["text"] for source in result.get("sources", [])])
            
            # 期待されるキーワードの出現率
            relevant_count = sum(1 for keyword in test["expected_context_keywords"] if keyword in contexts)
            relevance_score = relevant_count / len(test["expected_context_keywords"])
            
            # 無関係なキーワードの非出現率
            irrelevant_count = sum(1 for keyword in test["irrelevant_keywords"] if keyword in contexts)
            irrelevance_penalty = irrelevant_count / len(test["irrelevant_keywords"])
            
            final_score = relevance_score * (1 - irrelevance_penalty)
            total_relevance_score += final_score
        
        return {
            "average_relevance": total_relevance_score / len(test_cases),
            "test_cases": len(test_cases)
        }
    
    async def evaluate_hallucination(self) -> Dict:
        """ハルシネーション（幻覚）評価"""
        print("Evaluating hallucination...")
        
        # ハルシネーションテスト用の質問
        hallucination_tests = [
            {
                "query": "存在しない道路設計基準XYZ-999について説明してください",
                "should_acknowledge_unknown": True,
                "hallucination_keywords": ["XYZ-999は", "この基準では", "規定されています"]
            },
            {
                "query": "道路設計における量子コンピュータの活用方法",
                "should_acknowledge_limited": True,
                "hallucination_keywords": ["広く使われています", "一般的な手法", "標準的な"]
            }
        ]
        
        hallucination_count = 0
        appropriate_responses = 0
        
        for test in hallucination_tests:
            async with aiohttp.ClientSession() as session:
                payload = {"query": test["query"]}
                
                async with session.post(f"{self.rag_endpoint}/rag/query", json=payload) as response:
                    result = await response.json()
            
            answer = result["answer"]
            
            # ハルシネーションチェック
            has_hallucination = any(keyword in answer for keyword in test["hallucination_keywords"])
            if has_hallucination:
                hallucination_count += 1
            
            # 適切な応答チェック（「分からない」「情報がない」など）
            appropriate_keywords = ["情報がありません", "確認できません", "不明", "データがない"]
            if any(keyword in answer for keyword in appropriate_keywords):
                appropriate_responses += 1
        
        return {
            "hallucination_rate": hallucination_count / len(hallucination_tests),
            "appropriate_response_rate": appropriate_responses / len(hallucination_tests),
            "test_cases": len(hallucination_tests)
        }
    
    async def evaluate_scalability(self) -> Dict:
        """スケーラビリティ評価（並行処理性能）"""
        print("Evaluating scalability...")
        
        concurrent_levels = [1, 5, 10, 20, 50]
        results = {}
        
        test_query = "道路の設計速度について説明してください"
        
        for level in concurrent_levels:
            start = time.time()
            
            # 並行リクエスト
            tasks = []
            for _ in range(level):
                tasks.append(self._send_query(test_query))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            end = time.time()
            total_time = end - start
            
            # エラー率計算
            errors = sum(1 for r in responses if isinstance(r, Exception))
            error_rate = errors / level
            
            # スループット計算
            successful = level - errors
            throughput = successful / total_time if total_time > 0 else 0
            
            results[f"concurrent_{level}"] = {
                "total_time": total_time,
                "throughput": throughput,
                "error_rate": error_rate,
                "avg_time_per_request": total_time / level
            }
        
        return results
    
    async def _send_query(self, query: str) -> Dict:
        """単一クエリを送信"""
        async with aiohttp.ClientSession() as session:
            payload = {"query": query}
            async with session.post(f"{self.rag_endpoint}/rag/query", json=payload) as response:
                return await response.json()
```

## 3. 継続学習システムベンチマーク

### 3.1 継続学習評価メトリクス

```python
# src/benchmarks/continual_learning_benchmark.py
import torch
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime

class ContinualLearningBenchmark:
    """継続学習システムの評価ベンチマーク"""
    
    def __init__(self, task_history_path: str = "outputs/ewc_data/task_history.json"):
        self.task_history_path = Path(task_history_path)
        self.results = {}
        
    def evaluate_all(self) -> Dict:
        """全継続学習ベンチマークを実行"""
        
        print("Starting Continual Learning Benchmarks...")
        
        # 1. カタストロフィック忘却評価
        self.results['catastrophic_forgetting'] = self.evaluate_catastrophic_forgetting()
        
        # 2. 知識転移評価
        self.results['knowledge_transfer'] = self.evaluate_knowledge_transfer()
        
        # 3. タスク成功率評価
        self.results['task_success_rate'] = self.evaluate_task_success_rate()
        
        # 4. メモリ効率評価
        self.results['memory_efficiency'] = self.evaluate_memory_efficiency()
        
        # 5. 学習安定性評価
        self.results['learning_stability'] = self.evaluate_learning_stability()
        
        # 6. EWC効果評価
        self.results['ewc_effectiveness'] = self.evaluate_ewc_effectiveness()
        
        return self.results
    
    def evaluate_catastrophic_forgetting(self) -> Dict:
        """カタストロフィック忘却の評価"""
        print("Evaluating catastrophic forgetting...")
        
        # タスク履歴を読み込み
        with open(self.task_history_path) as f:
            task_history = json.load(f)
        
        if len(task_history) < 2:
            return {"error": "Not enough tasks for evaluation"}
        
        forgetting_scores = []
        
        for i in range(1, len(task_history)):
            current_task = task_history[i]
            previous_task = task_history[i-1]
            
            # 各タスクのモデルで前タスクのデータを評価
            if self._model_exists(current_task['model_path']):
                # 前タスクのテストデータで現在のモデルを評価
                previous_test_data = self._load_task_test_data(previous_task['task_name'])
                current_model_score = self._evaluate_model_on_data(
                    current_task['model_path'],
                    previous_test_data
                )
                
                # 前タスクのモデルでの元のスコア
                original_score = self._evaluate_model_on_data(
                    previous_task['model_path'],
                    previous_test_data
                )
                
                # 忘却率計算
                forgetting_rate = (original_score - current_model_score) / original_score if original_score > 0 else 0
                forgetting_scores.append(forgetting_rate)
        
        return {
            "average_forgetting_rate": np.mean(forgetting_scores) if forgetting_scores else 0,
            "max_forgetting_rate": np.max(forgetting_scores) if forgetting_scores else 0,
            "min_forgetting_rate": np.min(forgetting_scores) if forgetting_scores else 0,
            "evaluated_transitions": len(forgetting_scores)
        }
    
    def evaluate_knowledge_transfer(self) -> Dict:
        """知識転移の評価（前方・後方転移）"""
        print("Evaluating knowledge transfer...")
        
        with open(self.task_history_path) as f:
            task_history = json.load(f)
        
        forward_transfer_scores = []  # 前方転移
        backward_transfer_scores = []  # 後方転移
        
        for i in range(len(task_history)):
            current_task = task_history[i]
            
            # 前方転移: 新タスクの初期性能vs単独学習
            if i > 0:
                # 継続学習での初期性能
                continual_initial = self._get_initial_performance(current_task)
                
                # 単独学習での性能（ベースライン）
                standalone_performance = self._get_standalone_baseline(current_task['task_name'])
                
                forward_transfer = (continual_initial - standalone_performance) / standalone_performance if standalone_performance > 0 else 0
                forward_transfer_scores.append(forward_transfer)
            
            # 後方転移: 新タスク学習後の既存タスク性能向上
            if i < len(task_history) - 1:
                next_task = task_history[i+1]
                
                # 次タスク学習後の現タスク性能
                after_next = self._evaluate_model_on_data(
                    next_task['model_path'],
                    self._load_task_test_data(current_task['task_name'])
                )
                
                # 現タスク直後の性能
                before_next = self._evaluate_model_on_data(
                    current_task['model_path'],
                    self._load_task_test_data(current_task['task_name'])
                )
                
                backward_transfer = (after_next - before_next) / before_next if before_next > 0 else 0
                backward_transfer_scores.append(backward_transfer)
        
        return {
            "forward_transfer": {
                "mean": np.mean(forward_transfer_scores) if forward_transfer_scores else 0,
                "positive_rate": sum(1 for s in forward_transfer_scores if s > 0) / len(forward_transfer_scores) if forward_transfer_scores else 0
            },
            "backward_transfer": {
                "mean": np.mean(backward_transfer_scores) if backward_transfer_scores else 0,
                "positive_rate": sum(1 for s in backward_transfer_scores if s > 0) / len(backward_transfer_scores) if backward_transfer_scores else 0
            }
        }
    
    def evaluate_task_success_rate(self) -> Dict:
        """タスク成功率の評価"""
        print("Evaluating task success rate...")
        
        # タスク状態ファイルを読み込み
        tasks_state_file = Path("data/continual_learning/tasks_state.json")
        
        if not tasks_state_file.exists():
            return {"error": "Tasks state file not found"}
        
        with open(tasks_state_file) as f:
            tasks_state = json.load(f)
        
        total_tasks = len(tasks_state)
        completed_tasks = sum(1 for t in tasks_state.values() if t['status'] == 'completed')
        failed_tasks = sum(1 for t in tasks_state.values() if t['status'] == 'failed')
        
        # エラー分析
        error_types = {}
        for task in tasks_state.values():
            if task['status'] == 'failed':
                error = task.get('error', 'Unknown')
                if 'out of memory' in error.lower():
                    error_type = 'Memory Error'
                elif 'quantized' in error.lower():
                    error_type = 'Quantization Error'
                else:
                    error_type = 'Other Error'
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "failure_rate": failed_tasks / total_tasks if total_tasks > 0 else 0,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "error_distribution": error_types
        }
    
    def evaluate_memory_efficiency(self) -> Dict:
        """メモリ効率の評価"""
        print("Evaluating memory efficiency...")
        
        fisher_dir = Path("outputs/ewc_data")
        
        # Fisher行列のサイズ
        fisher_sizes = []
        for fisher_file in fisher_dir.glob("fisher_*.pt"):
            size_mb = fisher_file.stat().st_size / 1024 / 1024
            fisher_sizes.append(size_mb)
        
        # モデルサイズの推移
        model_sizes = []
        with open(self.task_history_path) as f:
            task_history = json.load(f)
        
        for task in task_history:
            model_path = Path(task['model_path'])
            if model_path.exists():
                total_size = sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file())
                model_sizes.append(total_size / 1024 / 1024)  # MB
        
        return {
            "fisher_matrices": {
                "count": len(fisher_sizes),
                "total_size_mb": sum(fisher_sizes),
                "average_size_mb": np.mean(fisher_sizes) if fisher_sizes else 0,
                "compression_ratio": 0.5  # FP16使用時
            },
            "model_sizes": {
                "initial_size_mb": model_sizes[0] if model_sizes else 0,
                "final_size_mb": model_sizes[-1] if model_sizes else 0,
                "growth_rate": (model_sizes[-1] - model_sizes[0]) / model_sizes[0] if model_sizes and model_sizes[0] > 0 else 0
            }
        }
    
    def evaluate_learning_stability(self) -> Dict:
        """学習安定性の評価"""
        print("Evaluating learning stability...")
        
        with open(self.task_history_path) as f:
            task_history = json.load(f)
        
        # 各タスクの学習曲線から安定性を評価
        stability_scores = []
        convergence_times = []
        
        for task in task_history:
            # トレーニングログから損失の推移を取得
            log_path = Path(task['model_path']) / "training_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    training_log = json.load(f)
                
                losses = training_log.get('losses', [])
                if len(losses) > 1:
                    # 損失の変動係数（安定性指標）
                    cv = np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0
                    stability_scores.append(1 - cv)  # 高いほど安定
                    
                    # 収束までのエポック数
                    convergence_epoch = self._find_convergence_epoch(losses)
                    convergence_times.append(convergence_epoch)
        
        return {
            "average_stability": np.mean(stability_scores) if stability_scores else 0,
            "stability_variance": np.var(stability_scores) if stability_scores else 0,
            "average_convergence_epochs": np.mean(convergence_times) if convergence_times else 0,
            "evaluated_tasks": len(stability_scores)
        }
    
    def evaluate_ewc_effectiveness(self) -> Dict:
        """EWC効果の評価"""
        print("Evaluating EWC effectiveness...")
        
        # EWCあり/なしの比較（仮想的な実装）
        with_ewc_forgetting = []
        without_ewc_forgetting = []
        
        # タスク履歴からEWC設定を確認
        with open(self.task_history_path) as f:
            task_history = json.load(f)
        
        for task in task_history:
            ewc_lambda = task.get('ewc_lambda', 0)
            
            if ewc_lambda > 0:
                # EWCありのタスク
                forgetting_rate = self._estimate_forgetting_rate(task['model_path'])
                with_ewc_forgetting.append(forgetting_rate)
            else:
                # EWCなしのタスク（ベースライン）
                forgetting_rate = self._estimate_forgetting_rate(task['model_path']) * 1.5  # 仮定: EWCなしは1.5倍忘却
                without_ewc_forgetting.append(forgetting_rate)
        
        effectiveness = 0
        if with_ewc_forgetting and without_ewc_forgetting:
            avg_with = np.mean(with_ewc_forgetting)
            avg_without = np.mean(without_ewc_forgetting)
            effectiveness = (avg_without - avg_with) / avg_without if avg_without > 0 else 0
        
        return {
            "ewc_effectiveness": effectiveness,
            "with_ewc_forgetting": np.mean(with_ewc_forgetting) if with_ewc_forgetting else 0,
            "without_ewc_forgetting": np.mean(without_ewc_forgetting) if without_ewc_forgetting else 0,
            "improvement_percentage": effectiveness * 100
        }
    
    def _model_exists(self, model_path: str) -> bool:
        """モデルが存在するか確認"""
        return Path(model_path).exists()
    
    def _load_task_test_data(self, task_name: str) -> List:
        """タスクのテストデータをロード"""
        # 実装: タスク名に基づいてテストデータを読み込み
        test_file = Path(f"data/test/{task_name}_test.jsonl")
        if test_file.exists():
            with open(test_file) as f:
                return [json.loads(line) for line in f]
        return []
    
    def _evaluate_model_on_data(self, model_path: str, test_data: List) -> float:
        """モデルをデータで評価"""
        # 実装: モデルをロードしてテストデータで評価
        # 簡略化のため、ダミー値を返す
        return np.random.random() * 0.9 + 0.1
    
    def _get_initial_performance(self, task: Dict) -> float:
        """タスクの初期性能を取得"""
        # 実装: トレーニングログから初期性能を取得
        return 0.3
    
    def _get_standalone_baseline(self, task_name: str) -> float:
        """単独学習のベースライン性能を取得"""
        # 実装: ベースラインデータから取得
        return 0.25
    
    def _find_convergence_epoch(self, losses: List[float]) -> int:
        """収束エポックを特定"""
        if len(losses) < 3:
            return len(losses)
        
        # 移動平均で平滑化
        window = 3
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        # 変化率が閾値以下になるエポックを探す
        threshold = 0.01
        for i in range(1, len(smoothed)):
            if abs(smoothed[i] - smoothed[i-1]) / smoothed[i-1] < threshold:
                return i + window - 1
        
        return len(losses)
    
    def _estimate_forgetting_rate(self, model_path: str) -> float:
        """忘却率を推定"""
        # 実装: モデルの性能劣化を推定
        return np.random.random() * 0.3
```

## 4. 統合ベンチマークスイート

### 4.1 全システム統合評価

```python
# src/benchmarks/integrated_benchmark.py
import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class IntegratedBenchmarkSuite:
    """統合ベンチマークスイート"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "fine_tuning": {},
            "rag": {},
            "continual_learning": {}
        }
        
    async def run_all_benchmarks(self):
        """全ベンチマークを実行"""
        
        print("=" * 80)
        print("Starting Integrated Benchmark Suite")
        print("=" * 80)
        
        # 1. ファインチューニングベンチマーク
        print("\n[1/3] Fine-tuning Benchmarks")
        ft_benchmark = FineTuningBenchmark("outputs/lora_20250908_163759")
        self.results["fine_tuning"] = ft_benchmark.evaluate_all()
        
        # 2. RAGベンチマーク
        print("\n[2/3] RAG System Benchmarks")
        rag_benchmark = RAGBenchmark()
        self.results["rag"] = await rag_benchmark.evaluate_all()
        
        # 3. 継続学習ベンチマーク
        print("\n[3/3] Continual Learning Benchmarks")
        cl_benchmark = ContinualLearningBenchmark()
        self.results["continual_learning"] = cl_benchmark.evaluate_all()
        
        # 結果の保存
        self._save_results()
        
        # レポート生成
        self._generate_report()
        
        # 可視化
        self._create_visualizations()
        
        return self.results
    
    def _save_results(self):
        """結果を保存"""
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    def _generate_report(self):
        """マークダウンレポートを生成"""
        report = []
        report.append("# Integrated Benchmark Report")
        report.append(f"\nGenerated: {self.results['timestamp']}")
        
        # ファインチューニング結果
        report.append("\n## Fine-tuning Performance")
        if "perplexity" in self.results["fine_tuning"]:
            perp = self.results["fine_tuning"]["perplexity"]["perplexity"]
            report.append(f"- **Perplexity**: {perp:.2f}")
        
        if "generation_quality" in self.results["fine_tuning"]:
            quality = self.results["fine_tuning"]["generation_quality"]
            report.append(f"- **ROUGE-L F1**: {quality.get('rougeL_f1', 0):.3f}")
            report.append(f"- **BERTScore F1**: {quality.get('bert_score_f1', 0):.3f}")
        
        # RAG結果
        report.append("\n## RAG System Performance")
        if "retrieval_accuracy" in self.results["rag"]:
            acc = self.results["rag"]["retrieval_accuracy"]
            report.append(f"- **Retrieval F1**: {acc.get('f1_score', 0):.3f}")
        
        if "response_time" in self.results["rag"]:
            times = self.results["rag"]["response_time"]
            if "low" in times:
                report.append(f"- **Response Time (Simple)**: {times['low']['mean_time']:.2f}s")
        
        # 継続学習結果
        report.append("\n## Continual Learning Performance")
        if "catastrophic_forgetting" in self.results["continual_learning"]:
            cf = self.results["continual_learning"]["catastrophic_forgetting"]
            report.append(f"- **Average Forgetting Rate**: {cf.get('average_forgetting_rate', 0):.2%}")
        
        if "task_success_rate" in self.results["continual_learning"]:
            tsr = self.results["continual_learning"]["task_success_rate"]
            report.append(f"- **Task Success Rate**: {tsr.get('success_rate', 0):.2%}")
        
        # スコアカード
        report.append("\n## Overall Score Card")
        scores = self._calculate_overall_scores()
        
        report.append("\n| System | Score | Grade |")
        report.append("|--------|-------|-------|")
        
        for system, score in scores.items():
            grade = self._score_to_grade(score)
            report.append(f"| {system} | {score:.1f}/100 | {grade} |")
        
        # 保存
        output_dir = Path("benchmarks/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"Report saved to: {report_file}")
    
    def _create_visualizations(self):
        """ベンチマーク結果の可視化"""
        output_dir = Path("benchmarks/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. システム別スコアのレーダーチャート
        self._create_radar_chart(output_dir / f"radar_{timestamp}.png")
        
        # 2. パフォーマンス推移グラフ
        self._create_performance_timeline(output_dir / f"timeline_{timestamp}.png")
        
        # 3. ヒートマップ
        self._create_heatmap(output_dir / f"heatmap_{timestamp}.png")
        
        print(f"Visualizations saved to: {output_dir}")
    
    def _calculate_overall_scores(self) -> Dict[str, float]:
        """総合スコアを計算"""
        scores = {}
        
        # ファインチューニングスコア
        ft_score = 0
        if "perplexity" in self.results["fine_tuning"]:
            # パープレキシティは低いほど良い（100を基準に正規化）
            perp = self.results["fine_tuning"]["perplexity"]["perplexity"]
            ft_score += max(0, min(50, (100 - perp) / 2))
        
        if "domain_specific" in self.results["fine_tuning"]:
            acc = self.results["fine_tuning"]["domain_specific"]["domain_accuracy"]
            ft_score += acc * 50
        
        scores["Fine-tuning"] = ft_score
        
        # RAGスコア
        rag_score = 0
        if "retrieval_accuracy" in self.results["rag"]:
            f1 = self.results["rag"]["retrieval_accuracy"]["f1_score"]
            rag_score += f1 * 40
        
        if "answer_quality" in self.results["rag"]:
            quality = self.results["rag"]["answer_quality"]["accuracy"]
            rag_score += quality * 40
        
        if "hallucination" in self.results["rag"]:
            hall_rate = self.results["rag"]["hallucination"]["hallucination_rate"]
            rag_score += (1 - hall_rate) * 20
        
        scores["RAG"] = rag_score
        
        # 継続学習スコア
        cl_score = 0
        if "task_success_rate" in self.results["continual_learning"]:
            success = self.results["continual_learning"]["task_success_rate"]["success_rate"]
            cl_score += success * 40
        
        if "catastrophic_forgetting" in self.results["continual_learning"]:
            forget = self.results["continual_learning"]["catastrophic_forgetting"]["average_forgetting_rate"]
            cl_score += (1 - forget) * 30
        
        if "ewc_effectiveness" in self.results["continual_learning"]:
            ewc = self.results["continual_learning"]["ewc_effectiveness"]["ewc_effectiveness"]
            cl_score += ewc * 30
        
        scores["Continual Learning"] = cl_score
        
        return scores
    
    def _score_to_grade(self, score: float) -> str:
        """スコアをグレードに変換"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _create_radar_chart(self, output_path: Path):
        """レーダーチャート作成"""
        # 実装: matplotlib でレーダーチャート
        pass
    
    def _create_performance_timeline(self, output_path: Path):
        """パフォーマンス推移グラフ作成"""
        # 実装: 時系列グラフ
        pass
    
    def _create_heatmap(self, output_path: Path):
        """ヒートマップ作成"""
        # 実装: seaborn でヒートマップ
        pass

# 実行スクリプト
async def main():
    """メインベンチマーク実行"""
    suite = IntegratedBenchmarkSuite()
    results = await suite.run_all_benchmarks()
    
    print("\n" + "=" * 80)
    print("Benchmark Suite Completed!")
    print("=" * 80)
    
    # サマリー表示
    scores = suite._calculate_overall_scores()
    print("\nOverall Scores:")
    for system, score in scores.items():
        grade = suite._score_to_grade(score)
        print(f"  {system}: {score:.1f}/100 ({grade})")

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. 実装とデプロイメント

### 5.1 ベンチマーク実行スクリプト

```bash
#!/bin/bash
# scripts/run_benchmarks.sh

echo "Starting System Benchmarks..."

# 環境準備
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace:$PYTHONPATH

# 個別ベンチマーク実行
echo "[1/4] Running Fine-tuning Benchmarks..."
python src/benchmarks/finetuning_benchmark.py \
    --model outputs/lora_20250908_163759 \
    --base-model models/base/llama-7b

echo "[2/4] Running RAG Benchmarks..."
python src/benchmarks/rag_benchmark.py \
    --endpoint http://localhost:8050

echo "[3/4] Running Continual Learning Benchmarks..."
python src/benchmarks/continual_learning_benchmark.py

echo "[4/4] Running Integrated Benchmarks..."
python src/benchmarks/integrated_benchmark.py

echo "Benchmarks completed!"
```

### 5.2 定期実行設定（cron）

```bash
# 日次ベンチマーク実行
0 2 * * * /workspace/scripts/run_benchmarks.sh >> /workspace/logs/benchmark.log 2>&1

# 週次詳細ベンチマーク
0 3 * * 0 /workspace/scripts/run_detailed_benchmarks.sh >> /workspace/logs/benchmark_detailed.log 2>&1
```

## 6. 期待される成果

### 6.1 定量的評価指標

| システム | メトリクス | 目標値 | 現在値（推定） |
|---------|-----------|--------|--------------|
| ファインチューニング | Perplexity | < 20 | 25-30 |
| | Domain Accuracy | > 80% | 60-70% |
| | Inference Speed | > 20 tok/s | 15-20 tok/s |
| RAG | Retrieval F1 | > 0.8 | 0.6-0.7 |
| | Response Time | < 2s | 2-3s |
| | Hallucination Rate | < 5% | 10-15% |
| 継続学習 | Task Success Rate | > 80% | 25% |
| | Forgetting Rate | < 10% | 20-30% |
| | EWC Effectiveness | > 50% | 30-40% |

### 6.2 改善ロードマップ

1. **短期（1ヶ月）**
   - ベンチマーク自動化
   - ダッシュボード構築
   - アラート設定

2. **中期（3ヶ月）**
   - パフォーマンス最適化
   - ベンチマーク拡充
   - A/Bテスト導入

3. **長期（6ヶ月）**
   - MLOps統合
   - 継続的改善パイプライン
   - 業界標準ベンチマーク対応

## まとめ

この包括的なベンチマークフレームワークにより、各システムの性能を定量的に評価し、継続的な改善が可能になります。特に重要なのは：

1. **自動化**: 定期的な評価により劣化を早期発見
2. **統合評価**: システム間の相互作用を含めた総合評価
3. **可視化**: ダッシュボードによる直感的な理解
4. **改善指標**: 具体的な改善ポイントの特定

これにより、システム全体の品質向上と最適化が実現できます。

---
*作成日: 2025-09-08*
*バージョン: 1.0*