#!/usr/bin/env python3
"""
教師データの品質チェックツール

使用方法:
    python scripts/check_training_data_quality.py RARdata.json
    python scripts/check_training_data_quality.py RARdata.json --verbose
    python scripts/check_training_data_quality.py RARdata.json --output report.json
"""

import json
import argparse
import re
from typing import Dict, List, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataQualityChecker:
    """教師データの品質チェッククラス"""

    def __init__(self):
        self.technical_terms = [
            "設計速度", "縦断勾配", "横断勾配", "曲線半径", "車線",
            "路肩", "中央帯", "停止距離", "視距", "構造令",
            "舗装", "盛土", "切土", "排水", "橋梁"
        ]

    def check_quality(self, data_path: str, verbose: bool = False) -> Dict:
        """品質チェック実行"""

        logger.info(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("Data must be a JSON array")
            return {"error": "Invalid format"}

        results = {
            "total_entries": len(data),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "issues": [],
            "quality_scores": [],
            "metrics": {
                "avg_oracle_ratio": 0,
                "avg_cot_length": 0,
                "avg_answer_length": 0,
                "citation_accuracy": 0,
                "technical_term_coverage": 0
            }
        }

        for idx, entry in enumerate(data):
            check_result = self.check_entry(entry, idx)
            results["quality_scores"].append(check_result["score"])

            if check_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

            if check_result["warnings"]:
                results["warnings"] += len(check_result["warnings"])

            if check_result["issues"]:
                results["issues"].append({
                    "entry_id": entry.get("id", f"entry_{idx}"),
                    "issues": check_result["issues"],
                    "warnings": check_result["warnings"]
                })

            if verbose and check_result["issues"]:
                logger.info(f"Entry {idx}: {check_result['issues']}")

        # 統計計算
        results["metrics"] = self.calculate_metrics(data)
        results["avg_quality_score"] = sum(results["quality_scores"]) / len(data)

        return results

    def check_entry(self, entry: Dict, idx: int) -> Dict:
        """個別エントリーのチェック"""

        issues = []
        warnings = []
        score = 100  # 満点から減点方式

        # 1. 必須フィールドチェック
        if "id" not in entry:
            issues.append("IDフィールドなし")
            score -= 10

        if "instruction" not in entry:
            issues.append("instructionフィールドなし")
            score -= 20

        if "documents" not in entry or not isinstance(entry["documents"], list):
            issues.append("documentsフィールドが無効")
            score -= 30
        else:
            # 2. Oracle文書チェック
            oracle_docs = [d for d in entry["documents"] if d.get("is_oracle", False)]
            if len(oracle_docs) == 0:
                issues.append("Oracle文書が存在しない")
                score -= 25

            oracle_ratio = len(oracle_docs) / len(entry["documents"]) if entry["documents"] else 0
            if oracle_ratio < 0.3:
                warnings.append(f"Oracle率が低い: {oracle_ratio:.1%}")
                score -= 5
            elif oracle_ratio > 0.9:
                warnings.append(f"Oracle率が高すぎる: {oracle_ratio:.1%}")
                score -= 5

        # 3. 出力フィールドチェック
        if "output" not in entry:
            issues.append("outputフィールドなし")
            score -= 30
        else:
            output = entry["output"]

            # Chain-of-Thoughtチェック
            if "chain_of_thought" not in output:
                issues.append("chain_of_thoughtなし")
                score -= 15
            else:
                cot = output["chain_of_thought"]
                if len(cot) < 50:
                    warnings.append(f"CoTが短い: {len(cot)}文字")
                    score -= 5

            # 回答チェック
            if "final_answer" not in output:
                issues.append("final_answerなし")
                score -= 15
            else:
                answer = output["final_answer"]
                if len(answer) < 20:
                    warnings.append(f"回答が短い: {len(answer)}文字")
                    score -= 3

            # 引用チェック
            if "citations" not in output:
                warnings.append("citationsなし")
                score -= 10
            else:
                citations = output["citations"]
                if entry.get("documents"):
                    oracle_sources = [d["source"] for d in entry["documents"] if d.get("is_oracle")]
                    for citation in citations:
                        if citation not in oracle_sources:
                            issues.append(f"引用が不正: {citation}")
                            score -= 10

        # 4. 専門用語のチェック
        instruction = entry.get("instruction", "")
        technical_count = sum(1 for term in self.technical_terms if term in instruction)
        if technical_count == 0:
            warnings.append("専門用語が含まれていない可能性")
            score -= 3

        # 5. 数値の存在チェック（計算問題の場合）
        if re.search(r'\d+', instruction) and "output" in entry:
            answer = entry["output"].get("final_answer", "")
            if not re.search(r'\d+', answer):
                warnings.append("質問に数値があるが回答に数値なし")
                score -= 5

        # スコアを0-100に制限
        score = max(0, min(100, score))

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "score": score
        }

    def calculate_metrics(self, data: List[Dict]) -> Dict:
        """統計メトリクスの計算"""

        oracle_ratios = []
        cot_lengths = []
        answer_lengths = []
        citation_matches = 0
        total_citations = 0
        technical_coverage = []

        for entry in data:
            # Oracle率
            if "documents" in entry and entry["documents"]:
                oracle_count = sum(1 for d in entry["documents"] if d.get("is_oracle", False))
                oracle_ratios.append(oracle_count / len(entry["documents"]))

            # CoT長さ
            if "output" in entry and "chain_of_thought" in entry["output"]:
                cot_lengths.append(len(entry["output"]["chain_of_thought"]))

            # 回答長さ
            if "output" in entry and "final_answer" in entry["output"]:
                answer_lengths.append(len(entry["output"]["final_answer"]))

            # 引用精度
            if "output" in entry and "citations" in entry["output"]:
                citations = entry["output"]["citations"]
                if "documents" in entry:
                    oracle_sources = [d["source"] for d in entry["documents"] if d.get("is_oracle")]
                    matches = sum(1 for c in citations if c in oracle_sources)
                    citation_matches += matches
                    total_citations += len(citations)

            # 専門用語カバレッジ
            instruction = entry.get("instruction", "")
            coverage = sum(1 for term in self.technical_terms if term in instruction)
            technical_coverage.append(coverage)

        return {
            "avg_oracle_ratio": sum(oracle_ratios) / len(oracle_ratios) if oracle_ratios else 0,
            "avg_cot_length": sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0,
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "citation_accuracy": citation_matches / total_citations if total_citations else 0,
            "avg_technical_terms": sum(technical_coverage) / len(technical_coverage) if technical_coverage else 0
        }

    def generate_report(self, results: Dict) -> str:
        """レポート生成"""

        report = f"""
================================================================================
教師データ品質チェック結果
================================================================================

📊 基本統計:
  総エントリー数: {results['total_entries']}
  合格: {results['passed']} ({results['passed']/results['total_entries']*100:.1f}%)
  不合格: {results['failed']} ({results['failed']/results['total_entries']*100:.1f}%)
  警告数: {results['warnings']}
  平均品質スコア: {results['avg_quality_score']:.1f}/100

📈 品質メトリクス:
  Oracle率: {results['metrics']['avg_oracle_ratio']:.1%}
  平均CoT長さ: {results['metrics']['avg_cot_length']:.0f}文字
  平均回答長さ: {results['metrics']['avg_answer_length']:.0f}文字
  引用精度: {results['metrics']['citation_accuracy']:.1%}
  平均専門用語数: {results['metrics']['avg_technical_terms']:.1f}個

"""

        if results['issues']:
            report += f"\n⚠️ 問題のあるエントリー ({len(results['issues'])}件):\n"
            for issue in results['issues'][:10]:  # 最初の10件のみ表示
                report += f"\n  [{issue['entry_id']}]\n"
                for i in issue['issues']:
                    report += f"    ❌ {i}\n"
                for w in issue['warnings']:
                    report += f"    ⚠️  {w}\n"

            if len(results['issues']) > 10:
                report += f"\n  ... 他 {len(results['issues']) - 10} 件\n"

        report += "\n" + "=" * 80 + "\n"

        # 推奨事項
        report += "\n💡 改善推奨事項:\n"

        if results['metrics']['avg_oracle_ratio'] < 0.5:
            report += "  - Oracle文書の比率を50%以上に増やすことを推奨\n"

        if results['metrics']['avg_cot_length'] < 100:
            report += "  - Chain-of-Thoughtをより詳細に記述することを推奨（100文字以上）\n"

        if results['metrics']['citation_accuracy'] < 0.9:
            report += "  - 引用とOracle文書の対応を確認してください\n"

        if results['metrics']['avg_technical_terms'] < 1:
            report += "  - 専門用語を含む質問を増やすことを推奨\n"

        return report


def main():
    parser = argparse.ArgumentParser(description="教師データ品質チェックツール")
    parser.add_argument("data_path", type=str, help="教師データファイルパス（JSON）")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")
    parser.add_argument("--output", type=str, help="結果をJSONファイルに保存")

    args = parser.parse_args()

    checker = TrainingDataQualityChecker()
    results = checker.check_quality(args.data_path, verbose=args.verbose)

    # レポート表示
    report = checker.generate_report(results)
    print(report)

    # JSON出力
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
