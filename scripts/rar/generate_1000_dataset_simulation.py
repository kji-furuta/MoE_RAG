#!/usr/bin/env python3
"""
1,000件データセット生成シミュレーション

Phase 1の100件データを基に、1,000件データセットのシミュレーションを行います。
実際のPhase 2では、NotebookLMで各バッチを生成しますが、
このスクリプトは自動化システムの動作確認とデモンストレーションを目的としています。
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import copy


class DataExpander:
    """データ拡張クラス"""

    def __init__(self, source_file: str):
        """
        Args:
            source_file: 元データファイル（Phase 1の100件）
        """
        with open(source_file, 'r', encoding='utf-8') as f:
            self.source_data = json.load(f)

        # バリエーション用のパラメータ
        self.design_speeds = [40, 50, 60, 70, 80, 100, 120]
        self.traffic_volumes = [500, 1000, 2000, 4000, 10000, 30000]
        self.pavement_types = ["アスファルト", "コンクリート", "半たわみ性"]

    def create_variations(self, base_entry: Dict, count: int, start_id: int) -> List[Dict]:
        """
        1つのエントリーから複数のバリエーションを生成

        Args:
            base_entry: 元のエントリー
            count: 生成する数
            start_id: 開始ID番号

        Returns:
            バリエーションエントリーのリスト
        """
        variations = []

        for i in range(count):
            # エントリーのディープコピー
            new_entry = copy.deepcopy(base_entry)

            # ID更新
            new_entry['id'] = f"DES-{start_id + i:03d}"

            # instructionのバリエーション
            instruction = new_entry.get('instruction', '')

            # 数値パラメータの置換
            if '80km/h' in instruction or '設計速度' in instruction:
                new_speed = random.choice(self.design_speeds)
                instruction = instruction.replace('80km/h', f'{new_speed}km/h')
                instruction = instruction.replace('80キロメートル', f'{new_speed}キロメートル')

            if '交通量' in instruction or '台/日' in instruction:
                new_traffic = random.choice(self.traffic_volumes)
                instruction = instruction.replace('4000台', f'{new_traffic}台')

            # instructionとoutputの更新
            new_entry['instruction'] = instruction

            # Chain-of-Thoughtの更新（数値パラメータを反映）
            if 'output' in new_entry and 'chain_of_thought' in new_entry['output']:
                cot = new_entry['output']['chain_of_thought']
                cot = cot.replace('80km/h', f'{random.choice(self.design_speeds)}km/h')
                cot = cot.replace('80キロメートル', f'{random.choice(self.design_speeds)}キロメートル')
                new_entry['output']['chain_of_thought'] = cot

            # final_answerの更新
            if 'output' in new_entry and 'final_answer' in new_entry['output']:
                answer = new_entry['output']['final_answer']
                answer = answer.replace('280メートル', f'{random.randint(200, 400)}メートル')
                answer = answer.replace('80km/h', f'{random.choice(self.design_speeds)}km/h')
                new_entry['output']['final_answer'] = answer

            variations.append(new_entry)

        return variations

    def generate_1000_dataset(self, output_dir: str) -> Dict:
        """
        1,000件データセットを20バッチで生成

        Args:
            output_dir: 出力ディレクトリ

        Returns:
            生成結果の統計情報
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        batch_size = 50
        num_batches = 20

        all_batches = []
        total_generated = 0

        print(f"🚀 1,000件データセット生成シミュレーション開始")
        print(f"  元データ: {len(self.source_data)}件")
        print(f"  バッチ数: {num_batches}")
        print(f"  バッチサイズ: {batch_size}件")
        print()

        for batch_num in range(1, num_batches + 1):
            start_id = (batch_num - 1) * batch_size + 1
            end_id = batch_num * batch_size

            print(f"📦 Batch {batch_num:02d}: DES-{start_id:03d} ~ DES-{end_id:03d}")

            # ランダムに元データを選択
            batch_entries = []
            entries_per_source = batch_size // len(self.source_data) + 1

            for source_entry in self.source_data:
                if len(batch_entries) >= batch_size:
                    break

                # このソースから生成する数
                to_generate = min(entries_per_source, batch_size - len(batch_entries))

                # バリエーション生成
                variations = self.create_variations(
                    source_entry,
                    to_generate,
                    start_id + len(batch_entries)
                )

                batch_entries.extend(variations)

            # バッチファイル保存
            batch_file = output_path / f"batch_{batch_num:02d}_simulated.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_entries, f, ensure_ascii=False, indent=2)

            all_batches.append(batch_entries)
            total_generated += len(batch_entries)

            print(f"  ✓ 生成: {len(batch_entries)}件 → {batch_file}")

        # 全バッチ統合
        print(f"\n🔗 全バッチ統合中...")
        all_entries = []
        for batch in all_batches:
            all_entries.extend(batch)

        # 統合ファイル保存
        output_file = output_path.parent / "rar_1000_simulated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

        # 統計情報
        stats = {
            "total_entries": len(all_entries),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "output_file": str(output_file),
            "batch_files": [str(output_path / f"batch_{i:02d}_simulated.json") for i in range(1, num_batches + 1)]
        }

        print(f"\n✅ 1,000件データセット生成完了")
        print(f"  総エントリー数: {stats['total_entries']}")
        print(f"  統合ファイル: {stats['output_file']}")

        return stats


def main():
    """メイン処理"""
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python generate_1000_dataset_simulation.py <source_file>")
        print("\n例:")
        print("  python generate_1000_dataset_simulation.py data/rar_training/pilot/rar_pilot_100.json")
        sys.exit(1)

    source_file = sys.argv[1]

    if not Path(source_file).exists():
        print(f"❌ エラー: ファイルが見つかりません: {source_file}")
        sys.exit(1)

    # データ拡張実行
    expander = DataExpander(source_file)
    stats = expander.generate_1000_dataset("data/rar_training/batches")

    print(f"\n📊 生成統計:")
    print(f"  総件数: {stats['total_entries']}")
    print(f"  バッチ数: {stats['num_batches']}")
    print(f"  出力ファイル: {stats['output_file']}")

    # 品質検証の提案
    print(f"\n🔍 次のステップ:")
    print(f"  1. 品質検証:")
    print(f"     python scripts/rar/validate_rar_json.py {stats['output_file']}")
    print(f"  2. バッチ検証:")
    print(f"     for f in data/rar_training/batches/batch_*_simulated.json; do")
    print(f"       python scripts/rar/auto_validate_and_fix.py $f")
    print(f"     done")


if __name__ == "__main__":
    main()
