#!/usr/bin/env python3
"""
RARデータ品質チェックスクリプト

RARdata.jsonファイルの品質をチェックし、学習前の検証を行います。

使用方法:
    # 基本的な使用
    python scripts/check_rar_data.py RARdata.json

    # 詳細モード
    python scripts/check_rar_data.py RARdata.json --verbose

    # 自動修正モード（警告レベルの問題を自動修正）
    python scripts/check_rar_data.py RARdata.json --auto-fix

チェック項目:
    ✅ JSON構文の妥当性
    ✅ ID重複チェック
    ✅ 必須フィールドの存在確認
    ✅ Oracle文書の比率
    ✅ Citation整合性
    ✅ データ統計情報
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


class RARDataValidator:
    """RARデータの品質検証クラス"""

    def __init__(self, file_path: str, verbose: bool = False):
        self.file_path = file_path
        self.verbose = verbose
        self.data = None
        self.errors = []
        self.warnings = []
        self.stats = {}

    def load_data(self) -> bool:
        """データファイルを読み込み"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON構文エラー: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"ファイルが見つかりません: {self.file_path}")
            return False
        except Exception as e:
            self.errors.append(f"ファイル読み込みエラー: {e}")
            return False

    def check_structure(self):
        """データ構造のチェック"""
        if not isinstance(self.data, list):
            self.errors.append("データはJSON配列である必要があります")
            return

        self.stats['total_entries'] = len(self.data)

        if len(self.data) == 0:
            self.errors.append("データが空です")

    def check_ids(self):
        """IDの重複と連続性をチェック"""
        ids = [entry.get('id') for entry in self.data]

        # ID重複チェック
        id_counts = Counter(ids)
        duplicates = [id_val for id_val, count in id_counts.items() if count > 1]

        if duplicates:
            self.errors.append(f"ID重複: {duplicates}")

        # ID範囲
        if ids:
            self.stats['id_range'] = f"{ids[0]} ～ {ids[-1]}"

            # ID連続性チェック（DES-001形式の場合）
            if ids[0].startswith('DES-'):
                try:
                    start_num = int(ids[0].split('-')[1])
                    end_num = int(ids[-1].split('-')[1])
                    expected_count = end_num - start_num + 1

                    if len(ids) != expected_count:
                        self.warnings.append(
                            f"IDが連続していません: {len(ids)}件（期待: {expected_count}件）"
                        )
                except (ValueError, IndexError):
                    pass

    def check_required_fields(self):
        """必須フィールドの存在確認"""
        required_fields = ['id', 'instruction', 'documents', 'output']
        output_required_fields = ['chain_of_thought', 'final_answer', 'citations']

        for i, entry in enumerate(self.data):
            entry_id = entry.get('id', f'index-{i}')

            # トップレベル必須フィールド
            for field in required_fields:
                if field not in entry:
                    self.errors.append(f"エントリ {entry_id}: {field} が欠落")

            # output内の必須フィールド
            if 'output' in entry:
                output = entry['output']
                for field in output_required_fields:
                    if field not in output:
                        self.errors.append(f"エントリ {entry_id}: output.{field} が欠落")

            # documentsが空でないかチェック
            if 'documents' in entry and len(entry['documents']) == 0:
                self.warnings.append(f"エントリ {entry_id}: documents が空です")

    def check_oracle_ratio(self):
        """Oracle文書の比率チェック"""
        oracle_count = 0
        total_docs = 0

        for entry in self.data:
            docs = entry.get('documents', [])
            total_docs += len(docs)
            oracle_count += sum(1 for doc in docs if doc.get('is_oracle', False))

        if total_docs > 0:
            oracle_ratio = oracle_count / total_docs
            self.stats['oracle_count'] = oracle_count
            self.stats['total_docs'] = total_docs
            self.stats['oracle_ratio'] = f"{oracle_ratio:.1%}"

            # 推奨範囲: 50-80%
            if oracle_ratio < 0.5:
                self.warnings.append(
                    f"Oracle比率が低すぎます: {oracle_ratio:.1%} (推奨: 50-80%)"
                )
            elif oracle_ratio > 0.8:
                self.warnings.append(
                    f"Oracle比率が高すぎます: {oracle_ratio:.1%} (推奨: 50-80%)"
                )
        else:
            self.stats['oracle_ratio'] = "N/A"
            self.warnings.append("文書データが存在しません")

    def check_citations(self):
        """Citation整合性チェック"""
        citation_errors = 0

        for i, entry in enumerate(self.data):
            entry_id = entry.get('id', f'index-{i}')
            citations = entry.get('output', {}).get('citations', [])
            docs = entry.get('documents', [])
            doc_sources = {doc.get('source') for doc in docs}

            for j, citation in enumerate(citations):
                cite_source = citation.get('source')

                # Citation元がdocumentsに存在するかチェック
                if cite_source not in doc_sources:
                    citation_errors += 1
                    if self.verbose:
                        self.errors.append(
                            f"エントリ {entry_id}: Citation[{j}]の元文書が documents に存在しない"
                        )

                # Citation必須フィールド
                if 'quote' not in citation:
                    self.warnings.append(f"エントリ {entry_id}: Citation[{j}]に quote が欠落")

        if citation_errors > 0 and not self.verbose:
            self.errors.append(f"Citation整合性エラー: {citation_errors}件")

    def check_data_quality(self):
        """データ品質の詳細チェック"""

        # Chain-of-Thought品質
        cot_lengths = []
        answer_lengths = []

        for entry in self.data:
            output = entry.get('output', {})
            cot = output.get('chain_of_thought', '')
            answer = output.get('final_answer', '')

            cot_lengths.append(len(cot))
            answer_lengths.append(len(answer))

        if cot_lengths:
            avg_cot_length = sum(cot_lengths) / len(cot_lengths)
            avg_answer_length = sum(answer_lengths) / len(answer_lengths)

            self.stats['avg_cot_length'] = f"{avg_cot_length:.0f}文字"
            self.stats['avg_answer_length'] = f"{avg_answer_length:.0f}文字"

            # CoTが短すぎる場合は警告
            if avg_cot_length < 50:
                self.warnings.append(
                    f"Chain-of-Thoughtが短すぎます: 平均{avg_cot_length:.0f}文字"
                )

    def validate(self) -> bool:
        """全チェックを実行"""

        if not self.load_data():
            return False

        self.check_structure()
        if self.data is None or not isinstance(self.data, list):
            return False

        self.check_ids()
        self.check_required_fields()
        self.check_oracle_ratio()
        self.check_citations()
        self.check_data_quality()

        return len(self.errors) == 0

    def print_report(self):
        """検証レポートを出力"""
        print("=" * 80)
        print(f"RARデータ品質チェック結果: {self.file_path}")
        print("=" * 80)

        # 統計情報
        print("\n📊 データ統計:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")

        # エラー
        if self.errors:
            print(f"\n❌ エラー ({len(self.errors)}件):")
            for error in self.errors:
                print(f"  - {error}")

        # 警告
        if self.warnings:
            print(f"\n⚠️  警告 ({len(self.warnings)}件):")
            for warning in self.warnings:
                print(f"  - {warning}")

        # 結果サマリー
        print("\n" + "=" * 80)
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("✅ 全チェック合格 - 学習に使用できます")
            else:
                print("⚠️  警告あり - 学習可能ですが確認推奨")
        else:
            print("❌ エラー検出 - 修正が必要です")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="RARデータ品質チェックスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的なチェック
  python scripts/check_rar_data.py RARdata.json

  # 詳細モード（すべてのエラーを表示）
  python scripts/check_rar_data.py RARdata.json --verbose

  # 複数ファイルをチェック
  python scripts/check_rar_data.py RARdata_v1.json RARdata_v2.json
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='チェックするRARデータファイル'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細モード（すべてのエラーを表示）'
    )

    args = parser.parse_args()

    all_valid = True

    for file_path in args.files:
        validator = RARDataValidator(file_path, verbose=args.verbose)
        is_valid = validator.validate()
        validator.print_report()

        if not is_valid:
            all_valid = False

        # 複数ファイルの場合は区切り線
        if len(args.files) > 1:
            print("\n")

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
