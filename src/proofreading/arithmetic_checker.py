"""
四則演算チェッカー + 表の整合性検証 + 丸め検証
LLMを使わず、正規表現 + Python算術評価で決定論的に検証する
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 全角→半角変換テーブル
_FULLWIDTH_MAP = str.maketrans(
    '０１２３４５６７８９＋－×÷＝（）．，',
    '0123456789+-*/=().,'
)


@dataclass
class ArithmeticFinding:
    """四則演算の検証結果"""
    position: int          # 原文中の位置（文字オフセット）
    original: str          # 原文の式
    expected: Optional[float]  # 正しい計算結果
    stated: Optional[float]    # 文書に記載された結果
    is_correct: bool
    message: str
    finding_type: str = "arithmetic"  # "arithmetic" or "table_sum"


@dataclass
class TableSumFinding:
    """表の整合性検証結果"""
    position: int
    table_context: str     # 表の周辺テキスト
    column_label: str      # 列名
    expected_sum: float
    stated_sum: float
    is_correct: bool
    message: str
    finding_type: str = "table_sum"


class ArithmeticChecker:
    """四則演算と表の整合性を検証するチェッカー"""

    # 四則演算パターン（半角・全角対応、2項演算、負数対応）
    # 例: 100 + 200 = 300, 5×3＝15, -0.80 × 10 ＝ -8.0
    _EQUATION_PATTERN = re.compile(
        r'(-?[\d,，\.．]+)'         # 左辺の数値1（負数可）
        r'\s*([+\-×÷*/＋－])\s*'    # 演算子
        r'(-?[\d,，\.．]+)'         # 左辺の数値2（負数可）
        r'\s*[=＝]\s*'              # 等号
        r'(-?[\d,，\.．]+)',        # 右辺（結果、負数可）
        re.UNICODE
    )

    # 連鎖演算パターン（3項以上の連鎖式、負数対応）
    # 例: 1.20 × 1.20 × 0.218 × 10 ＝ 3.139
    # 例: -0.80 × 0.80 × 0.218 × 10 ＝ -1.395
    _CHAIN_EQUATION_PATTERN = re.compile(
        r'(-?[\d,，\.．]+)'                                 # 最初の数値（負数可）
        r'((?:\s*[+\-×÷*/＋－]\s*[\d,，\.．]+){2,})'       # (演算子+数値) 2回以上
        r'\s*[=＝]\s*'                                       # 等号
        r'(-?[\d,，\.．]+)',                                 # 結果（負数可）
        re.UNICODE
    )

    # 合計行パターン（汎用テーブル用）
    _TOTAL_PATTERN = re.compile(
        r'(合計|計|小計|総計|Total|total|TOTAL|sum|SUM)'
        r'[：:\s]*'
        r'([\d,，\.．]+)',
        re.UNICODE
    )

    # 数値行パターン（表データの行）
    _TABLE_ROW_PATTERN = re.compile(
        r'^(.+?)[：:\t\s]+([\d,，\.．]+)$',
        re.MULTILINE | re.UNICODE
    )

    # 小計行パターン: 「計 ＝ 1.744」「小計 ＝ 17.440」
    _SUBTOTAL_PATTERN = re.compile(
        r'(計|小計|合計)\s*[=＝]\s*(-?[\d,，\.．]+)',
        re.UNICODE
    )

    # 式の結果パターン: 演算子を含む行の「＝ 数値」部分を抽出
    # 例: "1.20 × 1.20 × 0.218 × 10 ＝ 3.139" → 3.139
    _EQUATION_RESULT_PATTERN = re.compile(
        r'[×÷+\-*/＋－].*?[=＝]\s*(-?[\d,，\.．]+)',
        re.UNICODE
    )

    # 丸め検証パターン: 行末の「式結果 + 単位 + 数量」を検出
    # 数式の途中の数値を誤検出しないよう、行末に位置する組合せのみ対象
    # 例: "計 ＝ 1.744 ｍ3 1.7"  →  exact=1.744, quantity=1.7
    # 例: "＝ 47.760 kg 47.8"    →  exact=47.760, quantity=47.8
    # 括弧付き: "計 ＝ 17.440 式(ｍ 2 ) ( 17.4 )" →  exact=17.440, quantity=17.4
    _ROUNDING_PATTERN = re.compile(
        r'[=＝]\s*(-?[\d,，\.．]+)'           # 計算結果（正確値）
        r'\s+((?:ｍ[23]?|m[23]?|kg|式|組|本|箇所|個|ｍ２|枚|[^\d\-\s\(\)]{1,5})' # 単位
        r'(?:\([^\)]*\))?)'                    # 単位の括弧部分（任意）
        r'\s+\(?'                              # 空白 + 数量の開始（括弧あり/なし）
        r'\s*(-?[\d,，\.．]+)'                 # 数量（丸め後の値）
        r'\s*\)?\s*$',                         # 行末まで
        re.UNICODE | re.MULTILINE
    )

    def check_text(self, text: str, base_offset: int = 0) -> List[Dict[str, Any]]:
        """テキスト中の四則演算と表の整合性を検証する

        Args:
            text: チェック対象テキスト
            base_offset: 原文中のオフセット（チャンク使用時）

        Returns:
            検証結果のリスト
        """
        findings = []
        # 連鎖演算（3項以上）を先にチェックし、マッチ範囲を記録
        chain_ranges, chain_findings = self._check_chain_equations(text, base_offset)
        findings.extend(chain_findings)
        # 2項演算チェック（連鎖式の範囲内は除外して誤検出を防ぐ）
        findings.extend(self._check_equations(text, base_offset, exclude_ranges=chain_ranges))
        findings.extend(self._check_subtotals(text, base_offset))
        findings.extend(self._check_rounding(text, base_offset))
        findings.extend(self._check_table_sums(text, base_offset))
        return findings

    def _parse_number(self, s: str) -> Optional[float]:
        """数値文字列をfloatに変換（全角・カンマ対応）"""
        try:
            normalized = s.translate(_FULLWIDTH_MAP)
            normalized = normalized.replace(',', '')
            return float(normalized)
        except (ValueError, TypeError):
            return None

    def _check_chain_equations(self, text: str, base_offset: int):
        """連鎖演算式（3項以上）を検出し検証する

        例: 1.20 × 1.20 × 0.218 × 10 ＝ 3.139

        Returns:
            (chain_ranges, findings) のタプル
            chain_ranges: [(start, end), ...] マッチした範囲のリスト（2項チェック除外用）
            findings: 検証結果のリスト
        """
        findings = []
        chain_ranges = []
        _chain_op_pattern = re.compile(r'\s*([+\-×÷*/＋－])\s*([\d,，\.．]+)')

        for m in self._CHAIN_EQUATION_PATTERN.finditer(text):
            # マッチ範囲を記録（2項パターンの誤検出防止用）
            chain_ranges.append((m.start(), m.end()))

            first_str = m.group(1)
            chain_str = m.group(2)    # " × 1.20 × 0.218 × 10" 部分
            result_str = m.group(3)

            first = self._parse_number(first_str)
            stated = self._parse_number(result_str)
            if first is None or stated is None:
                continue

            # 連鎖部分をパース: [(演算子, 数値), ...]
            pairs = _chain_op_pattern.findall(chain_str)
            if len(pairs) < 2:
                continue

            # 左から順に計算
            expected = first
            valid = True
            for op_str, num_str in pairs:
                num = self._parse_number(num_str)
                if num is None:
                    valid = False
                    break
                op = op_str.translate(_FULLWIDTH_MAP)
                if op == '+':
                    expected += num
                elif op == '-':
                    expected -= num
                elif op == '*':
                    expected *= num
                elif op == '/':
                    if num == 0:
                        valid = False
                        break
                    expected /= num
                else:
                    valid = False
                    break

            if not valid:
                continue

            is_correct = abs(expected - stated) < 0.01

            if not is_correct:
                finding = ArithmeticFinding(
                    position=base_offset + m.start(),
                    original=m.group(0),
                    expected=expected,
                    stated=stated,
                    is_correct=False,
                    message=f"計算誤り: {m.group(0)} → 正しくは {expected:g}"
                )
                findings.append(finding.__dict__)

        return chain_ranges, findings

    def _check_equations(self, text: str, base_offset: int,
                         exclude_ranges: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
        """四則演算式を検出し正誤を判定する（2項演算）

        Args:
            exclude_ranges: [(start, end), ...] この範囲内のマッチはスキップする
                            （連鎖演算で既に検証済みの範囲）
        """
        results = []
        for m in self._EQUATION_PATTERN.finditer(text):
            # 連鎖演算の範囲内ならスキップ（誤検出防止）
            if exclude_ranges:
                match_start = m.start()
                match_end = m.end()
                skip = False
                for ex_start, ex_end in exclude_ranges:
                    if match_start >= ex_start and match_end <= ex_end:
                        skip = True
                        break
                if skip:
                    continue
            left_str, op_str, right_str, result_str = m.groups()
            left = self._parse_number(left_str)
            right = self._parse_number(right_str)
            stated = self._parse_number(result_str)

            if left is None or right is None or stated is None:
                continue

            # 演算子を正規化
            op = op_str.translate(_FULLWIDTH_MAP)
            if op == '*':
                # × も * に変換済み
                pass

            # 計算実行
            try:
                if op == '+':
                    expected = left + right
                elif op == '-':
                    expected = left - right
                elif op in ('*',):
                    expected = left * right
                elif op == '/':
                    if right == 0:
                        continue
                    expected = left / right
                else:
                    continue
            except Exception:
                continue

            # 浮動小数点の誤差を考慮（0.01以内は一致とみなす）
            is_correct = abs(expected - stated) < 0.01

            if not is_correct:
                finding = ArithmeticFinding(
                    position=base_offset + m.start(),
                    original=m.group(0),
                    expected=expected,
                    stated=stated,
                    is_correct=False,
                    message=f"計算誤り: {m.group(0)} → 正しくは {left_str}{op_str}{right_str}＝{expected:g}"
                )
                results.append(finding.__dict__)

        return results

    def _check_subtotals(self, text: str, base_offset: int) -> List[Dict[str, Any]]:
        """計算書の小計行を検証する

        PDFの計算書パターン:
          コンクリート V＝ 1.20 × 1.20 × 0.218 × 10 ＝ 3.139   ← 名称行（セクション開始）
                       -0.80 × 0.80 × 0.218 × 10 ＝ -1.395  ← 追加式
          計 ＝ 1.744                                           ← 小計 (3.139+(-1.395)=1.744)

        式の結果値（＝の右側）を蓄積し、「計 ＝ X」の X と合算を照合する。
        """
        results = []
        lines = text.split('\n')
        collected_results: List[float] = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # 1. 小計行の検出: 「計 ＝ 1.744」
            subtotal_match = self._SUBTOTAL_PATTERN.search(line_stripped)
            if subtotal_match and collected_results:
                stated_subtotal = self._parse_number(subtotal_match.group(2))
                if stated_subtotal is not None:
                    expected_subtotal = sum(collected_results)
                    is_correct = abs(expected_subtotal - stated_subtotal) < 0.01

                    if not is_correct:
                        line_offset = sum(len(lines[j]) + 1 for j in range(i))
                        finding = ArithmeticFinding(
                            position=base_offset + line_offset,
                            original=f"{subtotal_match.group(1)} ＝ {subtotal_match.group(2)}",
                            expected=expected_subtotal,
                            stated=stated_subtotal,
                            is_correct=False,
                            message=(
                                f"小計不一致: {subtotal_match.group(1)}＝{stated_subtotal:g} "
                                f"→ 各式の合計は {expected_subtotal:g}"
                            ),
                            finding_type="subtotal"
                        )
                        results.append(finding.__dict__)

                collected_results = []
                continue

            # 2. 名称行（セクション開始）の判定 — eq_resultより先に判定する
            #    先頭が日本語文字等で「V＝」「A＝」「W＝」等の変数代入を含む行
            #    → 蓄積をリセットし、その行自体の式結果だけ収集開始
            is_section_start = (
                line_stripped
                and not line_stripped[0].isdigit()
                and line_stripped[0] != '-'
                and bool(re.search(r'[A-Za-z][=＝]', line_stripped))
            )
            if is_section_start:
                collected_results = []
                # この行自体に式結果があれば新セクションの最初の値として収集
                eq_match = self._EQUATION_RESULT_PATTERN.search(line_stripped)
                if eq_match:
                    val = self._parse_number(eq_match.group(1))
                    if val is not None:
                        collected_results.append(val)
                continue

            # 3. 式結果の収集: 演算子を含み「＝ 数値」で終わる行
            eq_result_match = self._EQUATION_RESULT_PATTERN.search(line_stripped)
            if eq_result_match:
                val = self._parse_number(eq_result_match.group(1))
                if val is not None:
                    collected_results.append(val)

        return results

    def _check_rounding(self, text: str, base_offset: int) -> List[Dict[str, Any]]:
        """数量列の丸め値が計算結果と整合しているか検証する

        計算書のパターン:
          計 ＝ 1.744 ｍ3 1.7      ← 1.744 → 1.7（四捨五入 or 切捨で一致 ✅）
          ＝ 47.760 kg 47.8        ← 47.760 → 47.8（四捨五入で一致 ✅）
          計 ＝ 17.440 式(ｍ 2 ) ( 17.4 )  ← 括弧付き数量

        丸め方式の判定（いずれかに一致すればOK）:
          - 四捨五入
          - 切り捨て
          - 切り上げ
        """
        results = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for m in self._ROUNDING_PATTERN.finditer(line_stripped):
                exact_str = m.group(1)
                # group(2) = unit, group(3) = quantity
                quantity_str = m.group(3)

                exact = self._parse_number(exact_str)
                quantity = self._parse_number(quantity_str)

                if exact is None or quantity is None:
                    continue

                # 数量の小数桁数を推定
                quantity_norm = quantity_str.translate(_FULLWIDTH_MAP).replace(',', '')
                if '.' in quantity_norm:
                    decimals = len(quantity_norm.split('.')[1])
                else:
                    decimals = 0

                # 丸め方式チェック（いずれかに一致すればOK）
                factor = 10 ** decimals
                rounded_half = round(exact, decimals)
                rounded_floor = math.floor(abs(exact) * factor) / factor
                if exact < 0:
                    rounded_floor = -rounded_floor
                rounded_ceil = math.ceil(abs(exact) * factor) / factor
                if exact < 0:
                    rounded_ceil = -rounded_ceil

                is_ok = (
                    abs(quantity - rounded_half) < 0.001
                    or abs(quantity - rounded_floor) < 0.001
                    or abs(quantity - rounded_ceil) < 0.001
                )

                if not is_ok:
                    line_offset = sum(len(lines[j]) + 1 for j in range(i))
                    finding = ArithmeticFinding(
                        position=base_offset + line_offset,
                        original=f"{exact_str} → {quantity_str}",
                        expected=round(exact, decimals),
                        stated=quantity,
                        is_correct=False,
                        message=(
                            f"丸め不一致: {exact_str} → 数量 {quantity_str} "
                            f"（四捨五入={rounded_half:g}, "
                            f"切捨={rounded_floor:g}, "
                            f"切上={rounded_ceil:g}）"
                        ),
                        finding_type="rounding"
                    )
                    results.append(finding.__dict__)

        return results

    def _check_table_sums(self, text: str, base_offset: int) -> List[Dict[str, Any]]:
        """表の合計行と個別行の整合性を検証する

        テキスト中の連続する数値行と「合計」行を検出し、
        個別の数値の合算が合計値と一致するか検証する。
        """
        results = []
        lines = text.split('\n')
        current_numbers: List[float] = []
        current_start_line = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                # 空行でリセット
                current_numbers = []
                current_start_line = i + 1
                continue

            # 合計行の検出
            total_match = self._TOTAL_PATTERN.search(line_stripped)
            if total_match and current_numbers:
                stated_total = self._parse_number(total_match.group(2))
                if stated_total is not None:
                    expected_total = sum(current_numbers)
                    is_correct = abs(expected_total - stated_total) < 0.01

                    if not is_correct:
                        # 行位置をオフセットに変換（概算）
                        line_offset = sum(len(lines[j]) + 1 for j in range(i))
                        finding = TableSumFinding(
                            position=base_offset + line_offset,
                            table_context=f"行{current_start_line + 1}〜{i + 1}",
                            column_label=total_match.group(1),
                            expected_sum=expected_total,
                            stated_sum=stated_total,
                            is_correct=False,
                            message=(
                                f"表の合計不一致: {total_match.group(1)}={stated_total:g} "
                                f"→ 個別値の合計は {expected_total:g}"
                            )
                        )
                        results.append(finding.__dict__)

                current_numbers = []
                current_start_line = i + 1
                continue

            # 数値行の検出（タブ区切りやコロン区切りの行から数値を抽出）
            row_match = self._TABLE_ROW_PATTERN.match(line_stripped)
            if row_match:
                val = self._parse_number(row_match.group(2))
                if val is not None:
                    current_numbers.append(val)
            else:
                # 数値のみの行
                num_only = re.match(r'^\s*([\d,，\.．]+)\s*$', line_stripped)
                if num_only:
                    val = self._parse_number(num_only.group(1))
                    if val is not None:
                        current_numbers.append(val)
                else:
                    # 表の行ではない → リセット
                    current_numbers = []
                    current_start_line = i + 1

        return results
