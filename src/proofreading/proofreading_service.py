"""
校正サービス（オーケストレーター）
チャンク分割 → LLM校正 → 四則演算チェック → 結果統合
"""

import asyncio
import logging
from typing import Dict, Any, AsyncGenerator, Optional

from .chunker import ProofreadingChunker
from .arithmetic_checker import ArithmeticChecker
from .claude_backend import get_backend, ClaudeBackend, OllamaChatBackend
from .result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


def _resolve_line_position(position_value, chunk_text: str, chunk_start: int) -> int:
    """LLMが返したposition値を原文の文字オフセットに変換する

    LLMが行番号 "L15" 形式で返した場合:
        チャンク内の行番号から文字オフセットを算出し、chunk_startを加算

    数値の場合:
        そのまま使用

    それ以外:
        chunk_startをフォールバック値として使用
    """
    import re

    if isinstance(position_value, str):
        # "L15" 形式の行番号
        line_match = re.match(r'^L(\d+)$', position_value.strip())
        if line_match:
            line_num = int(line_match.group(1))
            lines = chunk_text.split('\n')
            # 行番号→文字オフセット（チャンク先頭からの位置）
            offset = 0
            for j in range(min(line_num - 1, len(lines))):
                offset += len(lines[j]) + 1  # +1 for \n
            return chunk_start + offset

    if isinstance(position_value, int):
        return position_value

    # フォールバック
    return chunk_start


# 校正リクエスト検出キーワード
_PROOFREAD_KEYWORDS_JA = ["誤字", "脱字", "校正", "推敲", "表記ゆれ", "表記揺れ", "タイポ"]
_PROOFREAD_KEYWORDS_EN = ["proofread", "proofreading", "typo"]


def is_proofreading_request(text: str) -> bool:
    """テキストが校正リクエストかどうか判定する

    main_unified.py の3箇所に重複していたロジックを統一。
    """
    if not text:
        return False
    t_lower = text.lower()
    return (
        any(k in text for k in _PROOFREAD_KEYWORDS_JA)
        or any(k in t_lower for k in _PROOFREAD_KEYWORDS_EN)
    )


class ProofreadingService:
    """校正サービス本体"""

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 200,
        max_concurrency: int = 5,
        check_arithmetic: bool = True,
        backend_type: str = "claude",
    ):
        self.chunker = ProofreadingChunker(chunk_size=chunk_size, overlap=overlap)
        self.arithmetic_checker = ArithmeticChecker()
        self.aggregator = ResultAggregator()
        self.max_concurrency = max_concurrency
        self.check_arithmetic = check_arithmetic
        self.backend_type = backend_type

    async def proofread_document(
        self, text: str, check_arithmetic: Optional[bool] = None
    ) -> Dict[str, Any]:
        """文書全体を校正する（非ストリーミング）

        Args:
            text: 校正対象テキスト
            check_arithmetic: 四則演算チェックを行うか（Noneの場合はインスタンス設定に従う）

        Returns:
            統合された校正レポート
        """
        do_arithmetic = check_arithmetic if check_arithmetic is not None else self.check_arithmetic

        # バックエンド取得
        backend = get_backend(self.backend_type)

        # チャンク分割
        chunks = self.chunker.split(text)
        total = len(chunks)
        logger.info(f"校正開始: {len(text)}文字 → {total}チャンク (backend={backend.__class__.__name__})")

        # 並列LLM校正
        semaphore = asyncio.Semaphore(self.max_concurrency)
        chunk_findings = [[] for _ in range(total)]

        async def process_chunk(chunk):
            async with semaphore:
                result = await backend.proofread_chunk_async(
                    chunk.text, chunk.index, total
                )
                if result.success:
                    # 各指摘にチャンクのオフセットを付与（dict以外をスキップ）
                    valid_findings = [f for f in result.findings if isinstance(f, dict)]
                    for f in valid_findings:
                        f["position"] = _resolve_line_position(
                            f.get("position"), chunk.text, chunk.start_pos
                        )
                    chunk_findings[chunk.index] = valid_findings
                else:
                    logger.warning(f"チャンク {chunk.index} 校正失敗: {result.error}")

        await asyncio.gather(*[process_chunk(c) for c in chunks])

        # 四則演算チェック
        arith_findings = []
        if do_arithmetic:
            arith_findings = self.arithmetic_checker.check_text(text)

        # 結果統合
        report = self.aggregator.aggregate(chunk_findings, arith_findings)
        report["total_chars"] = len(text)
        report["total_chunks"] = total
        report["backend"] = backend.__class__.__name__

        return report

    async def proofread_document_stream(
        self, text: str, check_arithmetic: Optional[bool] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """文書全体を校正し、進捗をストリーミングで返す

        Yields:
            進捗・指摘・完了イベント
        """
        do_arithmetic = check_arithmetic if check_arithmetic is not None else self.check_arithmetic

        backend = get_backend(self.backend_type)
        chunks = self.chunker.split(text)
        total = len(chunks)

        yield {"event": "start", "total_chunks": total, "total_chars": len(text),
               "backend": backend.__class__.__name__}

        semaphore = asyncio.Semaphore(self.max_concurrency)
        all_findings = [[] for _ in range(total)]
        completed = 0

        async def process_chunk(chunk):
            nonlocal completed
            async with semaphore:
                result = await backend.proofread_chunk_async(
                    chunk.text, chunk.index, total
                )
                completed += 1
                if result.success:
                    valid_findings = [f for f in result.findings if isinstance(f, dict)]
                    for f in valid_findings:
                        f["position"] = _resolve_line_position(
                            f.get("position"), chunk.text, chunk.start_pos
                        )
                    all_findings[chunk.index] = valid_findings
                return chunk.index, result

        # 並列実行して完了順にイベントを発行
        tasks = [asyncio.create_task(process_chunk(c)) for c in chunks]

        for coro in asyncio.as_completed(tasks):
            chunk_idx, result = await coro
            yield {
                "event": "progress",
                "chunk": chunk_idx,
                "completed": completed,
                "total": total,
                "percent": round(completed / total * 100, 1),
                "findings_count": len(result.findings) if result.success else 0,
            }
            # 個別指摘もストリーム送信
            if result.success and result.findings:
                for f in result.findings:
                    yield {"event": "finding", "data": f}

        # 四則演算チェック
        arith_findings = []
        if do_arithmetic:
            arith_findings = self.arithmetic_checker.check_text(text)
            for af in arith_findings:
                yield {"event": "arithmetic", "data": af}

        # 最終統合
        report = self.aggregator.aggregate(all_findings, arith_findings)
        report["total_chars"] = len(text)
        report["total_chunks"] = total
        report["backend"] = backend.__class__.__name__

        yield {"event": "complete", "report": report}
