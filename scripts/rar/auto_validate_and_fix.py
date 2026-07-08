#!/usr/bin/env python3
"""
RAR形式JSON自動検証・修正スクリプト

NotebookLMが生成したJSONを自動的に検証し、必要に応じて修正します。
Phase 1で発見された問題を自動修正します。
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed: bool = False
    fixed_content: Optional[str] = None


class RARJSONValidator:
    """RAR形式JSON検証・修正クラス"""

    # 利用可能なファイル名リスト（Phase 1で確認済み）
    VALID_SOURCES = {
        "【日本道路協会】道路構造令の解説と運用（令和3年）_R3年3月-5.pdf",
        "03_交通施設の技術基準とその解説（愛知_設計手引き）.pdf",
        "【日本道路協会】舗装設計施工指針（令和4年度版）_R4年11月.pdf",
        "（参考）04_道路（愛知_設計手引き）.pdf",
        "【日本道路協会】道路橋示方書・同解説（Ⅴ耐震設計編）_R9年11月.pdf",
    }

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_and_fix(self, input_path: str, output_path: Optional[str] = None) -> ValidationResult:
        """
        JSONファイルを検証し、必要に応じて修正

        Args:
            input_path: 入力JSONファイルパス
            output_path: 出力JSONファイルパス（Noneの場合は入力と同じディレクトリに_fixed.jsonを作成）

        Returns:
            ValidationResult: 検証結果
        """
        self.errors = []
        self.warnings = []

        print(f"📄 検証開始: {input_path}")

        # ファイル読み込み
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"ファイル読み込みエラー: {e}"],
                warnings=[]
            )

        # Step 1: JSON形式の検証と修正
        print("🔍 Step 1: JSON形式の検証...")
        parsed_data, format_fixed = self._fix_json_format(content)

        if parsed_data is None:
            return ValidationResult(
                is_valid=False,
                errors=self.errors,
                warnings=self.warnings
            )

        # Step 2: データ構造の検証
        print("🔍 Step 2: データ構造の検証...")
        structure_valid = self._validate_structure(parsed_data)

        # Step 3: Chain-of-Thought品質の検証
        print("🔍 Step 3: Chain-of-Thought品質の検証...")
        self._validate_cot_quality(parsed_data)

        # Step 4: 引用の検証
        print("🔍 Step 4: 引用の検証...")
        self._validate_citations(parsed_data)

        # Step 5: ファイル名の検証
        print("🔍 Step 5: ファイル名の検証...")
        self._validate_source_files(parsed_data)

        # 修正が必要な場合は出力
        if format_fixed or self.warnings:
            if output_path is None:
                input_pathobj = Path(input_path)
                output_path = str(input_pathobj.parent / f"{input_pathobj.stem}_fixed.json")

            # 修正版を保存
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 修正版を保存: {output_path}")
            fixed_content = output_path
        else:
            fixed_content = None

        # 結果サマリー
        print("\n" + "="*80)
        print("📊 検証結果サマリー")
        print("="*80)
        print(f"総エントリー数: {len(parsed_data)}")
        print(f"エラー数: {len(self.errors)}")
        print(f"警告数: {len(self.warnings)}")

        if self.errors:
            print("\n❌ エラー:")
            for error in self.errors[:10]:  # 最大10件表示
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings[:10]:  # 最大10件表示
                print(f"  - {warning}")

        is_valid = len(self.errors) == 0
        print(f"\n最終判定: {'✅ 合格' if is_valid else '❌ 不合格'}")

        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            fixed=format_fixed or len(self.warnings) > 0,
            fixed_content=fixed_content
        )

    def _fix_json_format(self, content: str) -> Tuple[Optional[List[Dict]], bool]:
        """
        JSON形式の修正

        Phase 1で発見された問題を自動修正:
        - 複数行JSONオブジェクト → JSON配列
        - LaTeX記号のエスケープ
        - オブジェクト間のカンマ
        """
        fixed = False

        # LaTeX記号のエスケープ修正
        original_content = content
        content = re.sub(r'(?<!\\)\\sigma', r'\\\\sigma', content)
        content = re.sub(r'(?<!\\)\\alpha', r'\\\\alpha', content)
        content = re.sub(r'(?<!\\)\\beta', r'\\\\beta', content)
        content = re.sub(r'(?<!\\)\\gamma', r'\\\\gamma', content)

        if content != original_content:
            print("  ✓ LaTeX記号をエスケープしました")
            fixed = True

        # まずは通常のJSON配列として解析を試みる
        try:
            data = json.loads(content)
            if isinstance(data, list):
                print("  ✓ 正しいJSON配列形式です")
                return data, fixed
            else:
                self.errors.append("JSONがリスト形式ではありません")
                return None, fixed
        except json.JSONDecodeError:
            print("  ⚠️  標準JSON解析失敗、手動パーサーを使用...")

        # 手動パーサー（Phase 1で使用したロジック）
        try:
            entries = []
            depth = 0
            current_obj = ""

            for char in content:
                if char == '{':
                    depth += 1
                    current_obj += char
                elif char == '}':
                    current_obj += char
                    depth -= 1
                    if depth == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            entries.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError as e:
                            self.warnings.append(f"オブジェクト解析エラー: {str(e)[:50]}")
                            current_obj = ""
                elif depth > 0:
                    current_obj += char

            if entries:
                print(f"  ✓ 手動パーサーで{len(entries)}件のオブジェクトを抽出しました")
                return entries, True
            else:
                self.errors.append("手動パーサーでもオブジェクトを抽出できませんでした")
                return None, fixed

        except Exception as e:
            self.errors.append(f"JSON解析エラー: {e}")
            return None, fixed

    def _validate_structure(self, data: List[Dict]) -> bool:
        """データ構造の検証"""
        required_fields = ['id', 'instruction', 'documents', 'output']
        output_fields = ['chain_of_thought', 'final_answer', 'citations']

        for i, entry in enumerate(data):
            # 必須フィールドの確認
            for field in required_fields:
                if field not in entry:
                    self.errors.append(f"エントリー {i}: 必須フィールド '{field}' が欠損")

            # outputフィールドの内部構造確認
            if 'output' in entry:
                for field in output_fields:
                    if field not in entry['output']:
                        self.errors.append(f"エントリー {i}: output.{field} が欠損")

            # documentsが配列であることを確認
            if 'documents' in entry:
                if not isinstance(entry['documents'], list):
                    self.errors.append(f"エントリー {i}: documentsがリスト形式ではありません")
                elif len(entry['documents']) == 0:
                    self.errors.append(f"エントリー {i}: documentsが空です")

        return len(self.errors) == 0

    def _validate_cot_quality(self, data: List[Dict]) -> None:
        """Chain-of-Thought品質の検証"""
        for i, entry in enumerate(data):
            if 'output' not in entry or 'chain_of_thought' not in entry['output']:
                continue

            cot = entry['output']['chain_of_thought']

            # 長さチェック
            if len(cot) < 50:
                self.warnings.append(f"エントリー {i} ({entry.get('id', 'unknown')}): CoTが短すぎます（{len(cot)}文字）")

            # ステップ化チェック
            has_steps = bool(re.search(r'[1-9]\.|①|②|③', cot))
            if not has_steps:
                self.warnings.append(f"エントリー {i} ({entry.get('id', 'unknown')}): CoTがステップ化されていません")

            # 推論語チェック
            reasoning_words = ['から', 'ため', 'により', 'したがって', 'これにより', 'そのため']
            has_reasoning = any(word in cot for word in reasoning_words)
            if not has_reasoning:
                self.warnings.append(f"エントリー {i} ({entry.get('id', 'unknown')}): CoTに推論語が含まれていません")

    def _validate_citations(self, data: List[Dict]) -> None:
        """引用の検証"""
        for i, entry in enumerate(data):
            if 'output' not in entry or 'citations' not in entry['output']:
                continue

            citations = entry['output']['citations']

            if not citations:
                self.warnings.append(f"エントリー {i} ({entry.get('id', 'unknown')}): 引用が空です")
                continue

            for j, citation in enumerate(citations):
                # 引用文の長さチェック
                if 'quote' in citation:
                    quote_len = len(citation['quote'])
                    if quote_len < 20:
                        self.warnings.append(
                            f"エントリー {i} 引用 {j}: 引用文が短すぎます（{quote_len}文字）"
                        )
                    elif quote_len > 200:
                        self.warnings.append(
                            f"エントリー {i} 引用 {j}: 引用文が長すぎます（{quote_len}文字）"
                        )

    def _validate_source_files(self, data: List[Dict]) -> None:
        """ファイル名の検証"""
        for i, entry in enumerate(data):
            if 'documents' not in entry:
                continue

            for j, doc in enumerate(entry['documents']):
                if 'source' not in doc:
                    self.errors.append(f"エントリー {i} 文書 {j}: sourceフィールドが欠損")
                    continue

                source = doc['source']
                if source not in self.VALID_SOURCES:
                    self.errors.append(
                        f"エントリー {i} 文書 {j}: 無効なファイル名 '{source}'"
                    )

            # 引用のファイル名も検証
            if 'output' in entry and 'citations' in entry['output']:
                for j, citation in enumerate(entry['output']['citations']):
                    if 'source' in citation:
                        source = citation['source']
                        if source not in self.VALID_SOURCES:
                            self.errors.append(
                                f"エントリー {i} 引用 {j}: 無効なファイル名 '{source}'"
                            )


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用方法: python auto_validate_and_fix.py <input_json_file> [output_json_file]")
        print("\n例:")
        print("  python auto_validate_and_fix.py RARdata_batch1.jsonl")
        print("  python auto_validate_and_fix.py RARdata_batch1.jsonl RARdata_batch1_fixed.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    validator = RARJSONValidator()
    result = validator.validate_and_fix(input_path, output_path)

    # 終了コード
    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
