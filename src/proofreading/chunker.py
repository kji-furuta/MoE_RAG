"""
文書チャンク分割モジュール
校正用に文書を句点・改行で区切り、オーバーラップ付きチャンクに分割する
計算書のセクション境界（名称行）を優先分割点にし、文脈の分断を防ぐ
"""

import re
from typing import List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# セクション開始行パターン（名称行）
# 「コンクリート V＝」「型枠 A＝」「鉄筋 W＝」等
_SECTION_START = re.compile(
    r'^[^\d\-\s].+?[A-Za-z][=＝]',
    re.UNICODE | re.MULTILINE
)


@dataclass
class Chunk:
    """チャンク情報"""
    index: int
    text: str
    start_pos: int  # 原文中の開始位置
    end_pos: int    # 原文中の終了位置


class ProofreadingChunker:
    """校正用チャンク分割"""

    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        """
        Args:
            chunk_size: チャンクサイズ（文字数）
            overlap: オーバーラップ（文字数）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[Chunk]:
        """テキストをチャンクに分割する

        分割優先順位:
        1. セクション境界（名称行の直前）— 計算書の文脈を保持
        2. 空行
        3. 句点（。）や改行
        隣接チャンク間は overlap 文字分を重複させ、境界付近のエラーを検出可能にする。
        """
        if not text:
            return []

        text_len = len(text)
        if text_len <= self.chunk_size:
            return [Chunk(index=0, text=text, start_pos=0, end_pos=text_len)]

        chunks: List[Chunk] = []
        pos = 0
        idx = 0

        while pos < text_len:
            end = min(pos + self.chunk_size, text_len)

            # chunk_size 以内で最適な分割点を探す
            if end < text_len:
                cut = self._find_best_boundary(text, pos, end)
                if cut > pos:
                    end = cut

            chunk_text = text[pos:end]
            chunks.append(Chunk(index=idx, text=chunk_text, start_pos=pos, end_pos=end))

            # 次のチャンク開始位置（overlap 分だけ戻す）
            next_pos = end - self.overlap
            if next_pos <= pos:
                # オーバーラップが大きすぎて前に進まない場合は強制前進
                next_pos = end
            pos = next_pos
            idx += 1

        logger.info(f"文書を {len(chunks)} チャンクに分割 (total={text_len} chars, chunk_size={self.chunk_size})")
        return chunks

    def _find_best_boundary(self, text: str, start: int, end: int) -> int:
        """start〜end の範囲で最適な分割点を探す

        優先順位:
        1. セクション開始行（名称行）の直前
        2. 空行（\\n\\n）
        3. 句点・改行
        """
        search_from = start + self.chunk_size // 2  # 後半で探す

        # 優先1: セクション境界（名称行の直前）を探す
        section_cut = self._find_section_boundary(text, search_from, end)
        if section_cut > start:
            return section_cut

        # 優先2: 空行を探す
        blank_cut = self._find_blank_line(text, search_from, end)
        if blank_cut > start:
            return blank_cut

        # 優先3: 句点・改行で切る（従来のロジック）
        return self._find_sentence_boundary(text, start, end)

    def _find_section_boundary(self, text: str, search_from: int, end: int) -> int:
        """search_from〜end 内でセクション開始行（名称行）の直前を探す

        後方から探し、最後に見つかったセクション開始行の直前で切る。
        """
        best = -1
        for m in _SECTION_START.finditer(text, search_from, end):
            # セクション開始行の直前（その行の先頭の前）で切る
            line_start = m.start()
            # 改行文字の直後がセクション開始行なので、改行位置で切断
            if line_start > search_from:
                best = line_start
        return best

    def _find_blank_line(self, text: str, search_from: int, end: int) -> int:
        """search_from〜end 内で最後の空行（\\n\\n）を探す"""
        pos = text.rfind('\n\n', search_from, end)
        if pos >= search_from:
            return pos + 1  # 空行の次の行から新チャンク
        return -1

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """start〜end の範囲で最後の文境界（句点・改行）を探す"""
        best = -1
        search_from = max(start + self.chunk_size // 2, start)
        for i in range(end - 1, search_from - 1, -1):
            ch = text[i]
            if ch in ('。', '\n', '．'):
                best = i + 1  # 句点の次の文字を切断位置にする
                break
        return best if best > start else -1
