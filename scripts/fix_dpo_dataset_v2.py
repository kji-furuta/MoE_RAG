#!/usr/bin/env python3
"""
DPOデータセットのJSONLフォーマット修正スクリプト v2
複数行にまたがるJSONオブジェクトを適切に処理
"""
import json
from pathlib import Path

def fix_dpo_dataset_v2(input_path: str, output_path: str = None):
    """
    複数行にまたがるJSONを正しいJSONL形式に変換

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス（Noneの場合は入力ファイルを上書き）
    """
    if output_path is None:
        output_path = input_path

    input_file = Path(input_path)
    temp_file = Path(str(output_path) + ".tmp")

    # 元のファイルを読み込み
    with input_file.open('r', encoding='utf-8') as f:
        content = f.read()

    # JSONオブジェクトを抽出（複数行対応）
    objects = []
    current_obj_lines = []
    brace_count = 0

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # ブレースカウント
        brace_count += line.count('{') - line.count('}')
        current_obj_lines.append(line)

        # オブジェクト完了
        if brace_count == 0 and current_obj_lines:
            obj_str = ' '.join(current_obj_lines)
            try:
                obj = json.loads(obj_str)
                objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse object: {e}")
                print(f"Lines: {current_obj_lines[:3]}")
            current_obj_lines = []

    print(f"Parsed {len(objects)} objects from input file")

    # 一時ファイルに正しいJSONL形式で書き出し
    with temp_file.open('w', encoding='utf-8') as f:
        valid_count = 0
        for i, obj in enumerate(objects):
            try:
                # DPO必須フィールドのチェックと抽出
                dpo_obj = {
                    "prompt": str(obj["prompt"]),
                    "chosen": str(obj["chosen"]),
                    "rejected": str(obj["rejected"])
                }
                # 1行1JSONとして書き出し（ensure_ascii=Falseで日本語を保持）
                json.dump(dpo_obj, f, ensure_ascii=False)
                f.write('\n')
                valid_count += 1
            except (KeyError, TypeError) as e:
                print(f"Warning: Skipping object {i} due to missing field: {e}")
                continue

    # 一時ファイルを本ファイルに置き換え
    temp_file.replace(output_path)

    print(f"Successfully wrote {valid_count} valid records to {output_path}")

    # 検証
    print("\n=== Validation ===")
    with Path(output_path).open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        print(f"Total lines: {len(lines)}")

        if lines:
            # 最初の3件をサンプル表示
            for i in range(min(3, len(lines))):
                obj = json.loads(lines[i])
                print(f"\nSample {i+1}:")
                print(f"  Fields: {list(obj.keys())}")
                print(f"  Prompt: {obj['prompt'][:50]}...")
                print(f"  Chosen: {obj['chosen'][:50]}...")
                print(f"  Rejected: {obj['rejected'][:50]}...")

    return valid_count

if __name__ == "__main__":
    # 元のファイルから修正
    source = "/workspace/data/dpo/request_dpo.jsonl"
    target = "/workspace/data/dpo/preference_dataset.jsonl"

    print(f"Converting: {source} -> {target}")
    count = fix_dpo_dataset_v2(source, target)
    print(f"\n✅ Conversion complete! {count} records ready for DPO training.")
