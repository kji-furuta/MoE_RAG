#!/usr/bin/env python3
"""
RAR形式データ生成バッチ処理システム

1,000件のRAR形式データを段階的に生成・検証・統合するシステム
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import subprocess


class RARBatchProcessor:
    """RAR形式データのバッチ処理マネージャー"""

    def __init__(self, output_dir: str = "data/rar_training/batches"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = 50  # 1バッチあたりのエントリー数
        self.target_total = 1000  # 目標総数

        # 統計情報
        self.stats = {
            "total_generated": 0,
            "total_valid": 0,
            "batches_completed": 0,
            "batches_failed": 0,
            "errors": [],
            "warnings": []
        }

    def generate_batch_instructions(self, batch_number: int) -> str:
        """
        バッチ生成用の指示書を作成

        Args:
            batch_number: バッチ番号（1始まり）

        Returns:
            生成指示のMarkdownテキスト
        """
        start_id = (batch_number - 1) * self.batch_size + 1
        end_id = batch_number * self.batch_size

        instructions = f"""
# RAR形式データ生成指示書 - Batch {batch_number}

## 📋 生成範囲
- **バッチ番号**: {batch_number}
- **ID範囲**: DES-{start_id:03d} ~ DES-{end_id:03d}
- **生成数**: {self.batch_size}件

## 🎯 重要な注意事項

### 前バッチとの重複防止
{'**初回バッチ**: 新規生成' if batch_number == 1 else f'**前バッチ (DES-{start_id-self.batch_size:03d} ~ DES-{start_id-1:03d}) と質問が重複しないように**'}

### ID命名規則
- **開始ID**: DES-{start_id:03d}
- **終了ID**: DES-{end_id:03d}
- **連続性**: 必ず連番で生成してください

## 📝 NotebookLMプロンプト

以下のプロンプトを使用してください:

---

[notebooklm_prompt_template_v2.md の内容をここに貼り付け]

**重要な変更点**:
- **生成数**: 50件
- **IDフォーマット**: `DES-{start_id:03d}` から連番
{'- **重複回避**: 前回生成した質問（設計速度、曲線半径、舗装厚さなど）とは異なる視点の質問を生成' if batch_number > 1 else ''}

---

## 🔄 生成後の処理フロー

1. NotebookLMで生成
2. 生成結果を `batch_{batch_number:02d}_raw.json` として保存
3. 自動検証スクリプトを実行:
   ```bash
   python scripts/rar/auto_validate_and_fix.py \\
     data/rar_training/batches/batch_{batch_number:02d}_raw.json \\
     data/rar_training/batches/batch_{batch_number:02d}_validated.json
   ```
4. 検証結果を確認
5. 必要に応じて手動修正

## ✅ 品質チェックリスト

### 自動検証項目
- [ ] JSON配列形式が正しい
- [ ] {self.batch_size}件すべてのエントリーが存在
- [ ] ID範囲が DES-{start_id:03d} ~ DES-{end_id:03d}
- [ ] 全エントリーに必須フィールド存在
- [ ] Chain-of-Thoughtがステップ化されている
- [ ] ファイル名が有効リストに含まれる

### 手動確認項目
- [ ] 質問の内容が技術的に妥当
- [ ] Chain-of-Thoughtの論理性が高い
- [ ] 引用が回答の根拠として適切
{'- [ ] 前バッチと質問が重複していない' if batch_number > 1 else ''}

## 📊 進捗状況

- **総生成数**: {(batch_number - 1) * self.batch_size} / {self.target_total}
- **達成率**: {(batch_number - 1) * self.batch_size / self.target_total * 100:.1f}%
- **残り**: {self.target_total - (batch_number - 1) * self.batch_size}件

---

**作成日**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}
"""
        return instructions

    def save_batch_instructions(self, batch_number: int) -> str:
        """バッチ指示書をファイルに保存"""
        instructions = self.generate_batch_instructions(batch_number)

        output_file = self.output_dir / f"batch_{batch_number:02d}_instructions.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(instructions)

        print(f"📄 バッチ{batch_number}の指示書を保存: {output_file}")
        return str(output_file)

    def validate_batch(self, batch_file: str) -> Dict:
        """
        バッチファイルを検証

        Args:
            batch_file: バッチJSONファイルパス

        Returns:
            検証結果の辞書
        """
        print(f"\n🔍 バッチファイル検証: {batch_file}")

        try:
            # 自動検証スクリプトを実行
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/rar/auto_validate_and_fix.py",
                    batch_file
                ],
                capture_output=True,
                text=True
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def merge_batches(self, batch_files: List[str], output_file: str) -> Dict:
        """
        複数のバッチファイルを統合

        Args:
            batch_files: バッチファイルのリスト
            output_file: 出力ファイルパス

        Returns:
            統合結果の辞書
        """
        print(f"\n🔗 バッチファイル統合開始...")
        print(f"  入力ファイル数: {len(batch_files)}")

        all_entries = []
        seen_ids = set()
        duplicates = []

        for batch_file in sorted(batch_files):
            if not Path(batch_file).exists():
                print(f"  ⚠️  ファイルが見つかりません: {batch_file}")
                continue

            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)

                # 重複チェック
                for entry in batch_data:
                    entry_id = entry.get('id', 'unknown')
                    if entry_id in seen_ids:
                        duplicates.append(entry_id)
                    else:
                        seen_ids.add(entry_id)
                        all_entries.append(entry)

                print(f"  ✓ {batch_file}: {len(batch_data)}件読み込み")

            except Exception as e:
                print(f"  ❌ {batch_file}: エラー - {e}")
                self.stats["errors"].append(f"{batch_file}: {e}")

        # 統合ファイル保存
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

        # 統計情報
        result = {
            "total_entries": len(all_entries),
            "unique_ids": len(seen_ids),
            "duplicates": duplicates,
            "output_file": str(output_path)
        }

        print(f"\n📊 統合結果:")
        print(f"  総エントリー数: {result['total_entries']}")
        print(f"  ユニークID数: {result['unique_ids']}")
        print(f"  重複ID数: {len(duplicates)}")
        if duplicates:
            print(f"  重複ID: {', '.join(duplicates[:10])}")
        print(f"  出力ファイル: {result['output_file']}")

        return result

    def generate_progress_report(self, current_batch: int) -> str:
        """進捗レポートを生成"""
        completed = (current_batch - 1) * self.batch_size
        progress = completed / self.target_total * 100

        report = f"""
# RAR形式データ生成 進捗レポート

**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}

## 📊 進捗状況

| 項目 | 値 |
|------|------|
| 目標総数 | {self.target_total}件 |
| 完了数 | {completed}件 |
| 達成率 | {progress:.1f}% |
| 完了バッチ | {current_batch - 1}/{self.target_total // self.batch_size} |
| 残りバッチ | {self.target_total // self.batch_size - (current_batch - 1)} |

## 📈 バッチ別状況

"""
        for i in range(1, current_batch):
            status = "✅ 完了" if i < current_batch else "🔄 処理中"
            report += f"- Batch {i:02d} (DES-{(i-1)*self.batch_size + 1:03d} ~ DES-{i*self.batch_size:03d}): {status}\n"

        report += f"\n## 🎯 次のステップ\n\n"
        report += f"1. Batch {current_batch:02d}の生成指示書を確認\n"
        report += f"2. NotebookLMでDES-{(current_batch-1)*self.batch_size + 1:03d} ~ DES-{current_batch*self.batch_size:03d}を生成\n"
        report += f"3. 検証スクリプトで品質チェック\n"
        report += f"4. 次のバッチへ進行\n"

        return report


def main():
    """メイン処理"""
    processor = RARBatchProcessor()

    if len(sys.argv) < 2:
        print("RAR形式データ生成バッチ処理システム")
        print("\n使用方法:")
        print("  1. 指示書生成: python batch_processing_system.py generate <batch_number>")
        print("  2. バッチ検証: python batch_processing_system.py validate <batch_file>")
        print("  3. バッチ統合: python batch_processing_system.py merge <output_file> <batch1> <batch2> ...")
        print("  4. 進捗レポート: python batch_processing_system.py report <current_batch>")
        print("\n例:")
        print("  python batch_processing_system.py generate 1")
        print("  python batch_processing_system.py validate data/rar_training/batches/batch_01_raw.json")
        print("  python batch_processing_system.py merge data/rar_training/rar_1000.json batch_*.json")
        print("  python batch_processing_system.py report 5")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        batch_number = int(sys.argv[2])
        output_file = processor.save_batch_instructions(batch_number)
        print(f"\n✅ 指示書を生成しました: {output_file}")
        print(f"\n次のステップ:")
        print(f"  1. {output_file} を確認")
        print(f"  2. NotebookLMで指示に従ってデータ生成")
        print(f"  3. 生成結果を data/rar_training/batches/batch_{batch_number:02d}_raw.json として保存")

    elif command == "validate":
        batch_file = sys.argv[2]
        result = processor.validate_batch(batch_file)
        print(result["stdout"])
        if not result["success"]:
            print(f"\n❌ 検証失敗")
            print(result["stderr"])
            sys.exit(1)
        print(f"\n✅ 検証成功")

    elif command == "merge":
        output_file = sys.argv[2]
        batch_files = sys.argv[3:]
        result = processor.merge_batches(batch_files, output_file)
        print(f"\n✅ 統合完了: {result['total_entries']}件")

    elif command == "report":
        current_batch = int(sys.argv[2])
        report = processor.generate_progress_report(current_batch)

        report_file = processor.output_dir / f"progress_report_batch{current_batch:02d}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(report)
        print(f"\n📄 レポート保存: {report_file}")

    else:
        print(f"❌ 未知のコマンド: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
