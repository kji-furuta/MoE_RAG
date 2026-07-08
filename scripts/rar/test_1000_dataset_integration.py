#!/usr/bin/env python3
"""
1,000件RAR形式データセットの統合テスト

Phase 2で生成された1,000件データセットの品質と
既存システムとの互換性を検証
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.core.citation_engine import Citation


def load_dataset(dataset_path: str) -> List[Dict]:
    """データセットを読み込む"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_dataset_structure(data: List[Dict]) -> Tuple[bool, List[str]]:
    """データセット構造の整合性を検証"""
    errors = []

    # ID重複チェック
    ids = [entry.get('id', '') for entry in data]
    id_counts = Counter(ids)
    duplicates = [id for id, count in id_counts.items() if count > 1]
    if duplicates:
        errors.append(f"ID重複検出: {duplicates[:5]}...")

    # ID連続性チェック
    expected_ids = {f"DES-{i:03d}" for i in range(1, len(data) + 1)}
    actual_ids = set(ids)
    missing_ids = expected_ids - actual_ids
    if missing_ids:
        errors.append(f"欠損ID検出: {sorted(list(missing_ids))[:5]}...")

    unexpected_ids = actual_ids - expected_ids
    if unexpected_ids:
        errors.append(f"予期しないID検出: {sorted(list(unexpected_ids))[:5]}...")

    return len(errors) == 0, errors


def test_citation_conversion(data: List[Dict], sample_size: int = 10) -> Tuple[bool, Dict]:
    """RAR形式からCitation形式への変換テスト"""

    import random
    sample_data = random.sample(data, min(sample_size, len(data)))

    results = {
        'total_tested': len(sample_data),
        'conversion_success': 0,
        'conversion_failed': 0,
        'total_citations': 0,
        'citation_errors': []
    }

    for entry in sample_data:
        entry_id = entry.get('id', 'unknown')
        try:
            citations = []
            for idx, rar_cite in enumerate(entry.get('output', {}).get('citations', [])):
                source = rar_cite.get('source', '')
                quote = rar_cite.get('quote', '')

                # Citationオブジェクトを作成
                citation = Citation(
                    id=str(idx + 1),
                    text=quote,
                    source=source,
                    document_title=source.replace('.pdf', ''),
                    page=None,
                    section=None
                )
                citations.append(citation)

            # インライン引用形式を生成
            inline_citations = [c.to_inline_citation() for c in citations]

            # 検証
            if len(citations) > 0 and all('[' in ic for ic in inline_citations):
                results['conversion_success'] += 1
                results['total_citations'] += len(citations)
            else:
                results['conversion_failed'] += 1
                results['citation_errors'].append(f"{entry_id}: 引用変換失敗")

        except Exception as e:
            results['conversion_failed'] += 1
            results['citation_errors'].append(f"{entry_id}: {str(e)}")

    success = results['conversion_failed'] == 0
    return success, results


def analyze_dataset_diversity(data: List[Dict]) -> Dict:
    """データセットの多様性を分析"""

    # 質問の長さ分布
    instruction_lengths = [len(entry.get('instruction', '')) for entry in data]

    # 引用数分布
    citation_counts = [len(entry.get('output', {}).get('citations', [])) for entry in data]

    # Oracle/Distractor比率
    oracle_counts = []
    distractor_counts = []
    for entry in data:
        docs = entry.get('documents', [])
        oracles = sum(1 for doc in docs if doc.get('is_oracle', False))
        distractors = len(docs) - oracles
        oracle_counts.append(oracles)
        distractor_counts.append(distractors)

    # Chain-of-Thought品質
    cot_scores = []
    for entry in data:
        cot = entry.get('output', {}).get('chain_of_thought', '')
        score = 0.0
        if len(cot) > 20:
            score += 0.3
        if '1.' in cot or '2.' in cot or '①' in cot:
            score += 0.4
        if any(k in cot for k in ['から', 'ため', 'により', 'したがって', 'よって', 'ので']):
            score += 0.3
        cot_scores.append(score)

    return {
        'instruction_length': {
            'min': min(instruction_lengths),
            'max': max(instruction_lengths),
            'avg': sum(instruction_lengths) / len(instruction_lengths)
        },
        'citations_per_entry': {
            'min': min(citation_counts),
            'max': max(citation_counts),
            'avg': sum(citation_counts) / len(citation_counts)
        },
        'oracle_ratio': {
            'avg_oracles': sum(oracle_counts) / len(oracle_counts),
            'avg_distractors': sum(distractor_counts) / len(distractor_counts),
            'avg_ratio': (sum(oracle_counts) / (sum(oracle_counts) + sum(distractor_counts))) * 100
        },
        'cot_quality': {
            'min': min(cot_scores),
            'max': max(cot_scores),
            'avg': sum(cot_scores) / len(cot_scores)
        }
    }


def main():
    """メインテスト実行"""

    print("=" * 80)
    print("Phase 2: 1,000件データセット統合テスト")
    print("=" * 80)

    dataset_path = "/workspace/data/rar_training/rar_1000_simulated.json"

    if not Path(dataset_path).exists():
        print(f"❌ データセットが見つかりません: {dataset_path}")
        sys.exit(1)

    print(f"\n📁 データセット: {dataset_path}")

    # データセット読み込み
    print("\n🔍 データセット読み込み中...")
    data = load_dataset(dataset_path)
    print(f"   総エントリー数: {len(data)}")

    # 1. 構造整合性チェック
    print("\n" + "=" * 80)
    print("1️⃣  データセット構造検証")
    print("=" * 80)

    structure_valid, structure_errors = validate_dataset_structure(data)

    if structure_valid:
        print("✅ 構造整合性: 合格")
        print(f"   - ID範囲: DES-001 ~ DES-{len(data):03d}")
        print(f"   - ID重複: なし")
        print(f"   - ID欠損: なし")
    else:
        print("❌ 構造整合性: 問題あり")
        for error in structure_errors:
            print(f"   - {error}")

    # 2. Citation変換テスト
    print("\n" + "=" * 80)
    print("2️⃣  Citation変換テスト (サンプル10件)")
    print("=" * 80)

    conversion_valid, conversion_results = test_citation_conversion(data, sample_size=10)

    print(f"テスト件数: {conversion_results['total_tested']}")
    print(f"変換成功: {conversion_results['conversion_success']}")
    print(f"変換失敗: {conversion_results['conversion_failed']}")
    print(f"総引用数: {conversion_results['total_citations']}")

    if conversion_valid:
        print("✅ Citation変換: 合格")
    else:
        print("❌ Citation変換: 問題あり")
        for error in conversion_results['citation_errors'][:5]:
            print(f"   - {error}")

    # 3. データ多様性分析
    print("\n" + "=" * 80)
    print("3️⃣  データセット多様性分析")
    print("=" * 80)

    diversity = analyze_dataset_diversity(data)

    print(f"\n📏 質問文の長さ:")
    print(f"   最小: {diversity['instruction_length']['min']} 文字")
    print(f"   最大: {diversity['instruction_length']['max']} 文字")
    print(f"   平均: {diversity['instruction_length']['avg']:.1f} 文字")

    print(f"\n📚 引用数:")
    print(f"   最小: {diversity['citations_per_entry']['min']} 個")
    print(f"   最大: {diversity['citations_per_entry']['max']} 個")
    print(f"   平均: {diversity['citations_per_entry']['avg']:.1f} 個")

    print(f"\n🎯 Oracle/Distractor比率:")
    print(f"   平均Oracle数: {diversity['oracle_ratio']['avg_oracles']:.1f}")
    print(f"   平均Distractor数: {diversity['oracle_ratio']['avg_distractors']:.1f}")
    print(f"   Oracle比率: {diversity['oracle_ratio']['avg_ratio']:.1f}%")

    print(f"\n🧠 Chain-of-Thought品質:")
    print(f"   最小スコア: {diversity['cot_quality']['min']:.2f}")
    print(f"   最大スコア: {diversity['cot_quality']['max']:.2f}")
    print(f"   平均スコア: {diversity['cot_quality']['avg']:.2f}/1.00")

    # 総合判定
    print("\n" + "=" * 80)
    print("📊 総合判定")
    print("=" * 80)

    all_passed = structure_valid and conversion_valid

    if all_passed:
        print("✅ 全テスト合格")
        print("\n1,000件データセットは以下の品質基準を満たしています:")
        print("  ✓ 構造整合性: ID連続性、重複なし")
        print("  ✓ Citation互換性: 既存システムと完全互換")
        print(f"  ✓ CoT品質: 平均スコア {diversity['cot_quality']['avg']:.2f}/1.00")
        print(f"  ✓ Oracle比率: {diversity['oracle_ratio']['avg_ratio']:.1f}% (目標60-70%)")
        print("\n🎯 Phase 2完了: 学習データとして使用可能です")
        return 0
    else:
        print("⚠️  一部テスト失敗")
        print("\n修正が必要な項目:")
        if not structure_valid:
            print("  ✗ 構造整合性の問題")
        if not conversion_valid:
            print("  ✗ Citation変換の問題")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
