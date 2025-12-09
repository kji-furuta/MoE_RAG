#!/usr/bin/env python3
"""
RAR形式データの品質検証スクリプト

検証項目:
1. JSONフォーマット妥当性
2. 必須フィールドの存在確認
3. Chain-of-Thoughtの品質評価
4. 引用情報の整合性
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

def validate_rar_format(entry: Dict) -> Tuple[bool, List[str]]:
    """RAR形式の必須フィールドを検証"""
    errors = []

    # 必須フィールド確認
    required_fields = ['id', 'instruction', 'documents', 'output']
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    if 'documents' in entry:
        if not isinstance(entry['documents'], list) or len(entry['documents']) == 0:
            errors.append("'documents' must be a non-empty list")
        else:
            for i, doc in enumerate(entry['documents']):
                if 'source' not in doc or 'content' not in doc or 'is_oracle' not in doc:
                    errors.append(f"Document {i} missing required fields")

    if 'output' in entry:
        output = entry['output']
        required_output_fields = ['chain_of_thought', 'final_answer', 'citations']
        for field in required_output_fields:
            if field not in output:
                errors.append(f"Missing output field: {field}")

        if 'citations' in output:
            if not isinstance(output['citations'], list):
                errors.append("'citations' must be a list")
            for i, cite in enumerate(output['citations']):
                if 'source' not in cite or 'quote' not in cite:
                    errors.append(f"Citation {i} missing required fields")

    return len(errors) == 0, errors

def assess_cot_quality(cot: str) -> Dict[str, any]:
    """Chain-of-Thoughtの品質を評価"""
    metrics = {
        'length': len(cot),
        'has_steps': '1.' in cot or '2.' in cot or '①' in cot,
        'has_reasoning': any(keyword in cot for keyword in ['から', 'ため', 'により', 'したがって', 'よって']),
        'word_count': len(cot.split()),
        'score': 0.0
    }

    # スコアリング
    score = 0.0
    if metrics['length'] > 20:
        score += 0.3
    if metrics['has_steps']:
        score += 0.4
    if metrics['has_reasoning']:
        score += 0.3

    metrics['score'] = score
    return metrics

def validate_citations(entry: Dict) -> Tuple[bool, List[str]]:
    """引用情報の整合性を検証"""
    errors = []

    if 'output' not in entry or 'citations' not in entry['output']:
        return True, []

    citations = entry['output']['citations']
    documents = entry.get('documents', [])

    # 引用元が実際に存在するドキュメントか確認
    doc_sources = {doc['source'] for doc in documents}
    for i, cite in enumerate(citations):
        if cite['source'] not in doc_sources:
            errors.append(f"Citation {i} references non-existent source: {cite['source']}")

        # 引用文が空でないか確認
        if not cite.get('quote', '').strip():
            errors.append(f"Citation {i} has empty quote")

    return len(errors) == 0, errors

def main(jsonl_path: str):
    """メイン検証処理"""
    print("=" * 80)
    print("RAR形式データ品質検証")
    print("=" * 80)

    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"❌ エラー: ファイルが見つかりません: {jsonl_path}")
        sys.exit(1)

    total_entries = 0
    valid_entries = 0
    format_errors = []
    citation_errors = []
    cot_scores = []

    print(f"\n📁 ファイル: {jsonl_path}")
    print("検証中...\n")

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                total_entries += 1

                # フォーマット検証
                is_valid, errors = validate_rar_format(entry)
                if not is_valid:
                    format_errors.extend([f"Line {line_num} ({entry.get('id', 'unknown')}): {err}" for err in errors])

                # CoT品質評価
                if 'output' in entry and 'chain_of_thought' in entry['output']:
                    cot_metrics = assess_cot_quality(entry['output']['chain_of_thought'])
                    cot_scores.append(cot_metrics['score'])

                # 引用検証
                cite_valid, cite_errors = validate_citations(entry)
                if not cite_valid:
                    citation_errors.extend([f"Line {line_num} ({entry.get('id', 'unknown')}): {err}" for err in cite_errors])

                if is_valid and cite_valid:
                    valid_entries += 1

            except json.JSONDecodeError as e:
                format_errors.append(f"Line {line_num}: JSON parse error: {e}")

    # 結果サマリー
    print("=" * 80)
    print("📊 検証結果サマリー")
    print("=" * 80)
    print(f"総エントリー数: {total_entries}")
    print(f"有効エントリー数: {valid_entries}")
    print(f"無効エントリー数: {total_entries - valid_entries}")
    print(f"成功率: {valid_entries/total_entries*100:.1f}%" if total_entries > 0 else "N/A")

    # CoT品質スコア
    if cot_scores:
        avg_cot_score = sum(cot_scores) / len(cot_scores)
        print(f"\n🧠 Chain-of-Thought平均スコア: {avg_cot_score:.2f}/1.00")

        score_distribution = Counter([round(s, 1) for s in cot_scores])
        print("スコア分布:")
        for score in sorted(score_distribution.keys(), reverse=True):
            count = score_distribution[score]
            bar = "█" * int(count / total_entries * 50)
            print(f"  {score:.1f}: {bar} ({count}件)")

    # エラー詳細（最初の10件のみ表示）
    if format_errors:
        print(f"\n⚠️ フォーマットエラー ({len(format_errors)}件):")
        for err in format_errors[:10]:
            print(f"  - {err}")
        if len(format_errors) > 10:
            print(f"  ... 他{len(format_errors) - 10}件")

    if citation_errors:
        print(f"\n⚠️ 引用エラー ({len(citation_errors)}件):")
        for err in citation_errors[:10]:
            print(f"  - {err}")
        if len(citation_errors) > 10:
            print(f"  ... 他{len(citation_errors) - 10}件")

    # 判定
    print("\n" + "=" * 80)
    if valid_entries == total_entries and avg_cot_score >= 0.7:
        print("✅ 検証合格: RAR形式データは品質基準を満たしています")
        print(f"   - フォーマット妥当性: 100%")
        print(f"   - CoT品質スコア: {avg_cot_score:.2f} (基準: ≥0.70)")
        return 0
    elif valid_entries / total_entries >= 0.95:
        print("⚠️ 条件付き合格: 一部エラーがありますが使用可能です")
        print(f"   - 有効率: {valid_entries/total_entries*100:.1f}% (基準: ≥95%)")
        return 0
    else:
        print("❌ 検証失敗: データ修正が必要です")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python validate_rar_data.py <RARdata.jsonl>")
        sys.exit(1)

    exit_code = main(sys.argv[1])
    sys.exit(exit_code)
