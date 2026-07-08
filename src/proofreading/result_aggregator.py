"""
チャンク結果統合モジュール
複数チャンクの校正結果を統合し、重複を排除してレポートを生成する
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ResultAggregator:
    """校正結果の統合・重複排除"""

    def aggregate(
        self,
        chunk_findings: List[List[Dict[str, Any]]],
        arithmetic_findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """全チャンクの結果を統合する

        Args:
            chunk_findings: 各チャンクのLLM校正結果リスト
            arithmetic_findings: 四則演算チェッカーの結果リスト

        Returns:
            統合されたレポート
        """
        # 全指摘をフラットにする
        all_findings: List[Dict[str, Any]] = []
        for findings in chunk_findings:
            all_findings.extend(findings)

        # 重複排除（同じ original + suggested の組み合わせ）
        deduplicated = self._deduplicate(all_findings)

        # 四則演算の指摘を追加（dict以外をスキップ）
        arithmetic_results = [
            {
                "position": f.get("position", 0),
                "type": f.get("finding_type", "arithmetic"),
                "original": f.get("original", ""),
                "suggested": f.get("message", ""),
                "severity": "error",
            }
            for f in arithmetic_findings
            if isinstance(f, dict)
        ]

        # 統合
        combined = deduplicated + arithmetic_results

        # サマリー統計
        summary = self._build_summary(deduplicated, arithmetic_results)

        return {
            "findings": combined,
            "arithmetic_findings": arithmetic_findings,
            "summary": summary,
        }

    def _deduplicate(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複する指摘を排除（オーバーラップ領域の重複対策）"""
        seen = set()
        result = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            key = (
                f.get("original", ""),
                f.get("suggested", ""),
                f.get("type", ""),
            )
            if key not in seen:
                seen.add(key)
                result.append(f)
        return result

    def _build_summary(
        self,
        proofreading: List[Dict[str, Any]],
        arithmetic: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """サマリー統計を生成"""
        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}

        for f in proofreading:
            if not isinstance(f, dict):
                continue
            t = f.get("type", "other")
            type_counts[t] = type_counts.get(t, 0) + 1
            s = f.get("severity", "info")
            if s in severity_counts:
                severity_counts[s] += 1

        return {
            "total_proofreading_findings": len(proofreading),
            "total_arithmetic_findings": len(arithmetic),
            "total_findings": len(proofreading) + len(arithmetic),
            "by_type": type_counts,
            "by_severity": severity_counts,
        }
