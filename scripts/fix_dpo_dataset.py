#!/usr/bin/env python3
"""
DPOデータセットのJSONLフォーマット修正スクリプト
"""
import json
import re
from pathlib import Path

def fix_dpo_dataset(input_path: str, output_path: str = None):
    """
    不正なJSONフォーマットを正しいJSONL形式に変換

    Args:
        input_path: 入力ファイルパス
        output_path: 出力ファイルパス（Noneの場合は入力ファイルを上書き）
    """
    if output_path is None:
        output_path = input_path

    input_file = Path(input_path)
    output_file = Path(output_path)

    # ファイルを読み込む
    with input_file.open('r', encoding='utf-8') as f:
        content = f.read()

    # 各JSON オブジェクトを抽出（改行とスペースで区切られている）
    # 正規表現で { ... } のパターンを抽出
    json_objects = []

    # ブレース対応のシンプルなパーサー
    depth = 0
    current_obj = []

    for char in content:
        if char == '{':
            depth += 1
            current_obj.append(char)
        elif char == '}':
            current_obj.append(char)
            depth -= 1
            if depth == 0 and current_obj:
                obj_str = ''.join(current_obj).strip()
                if obj_str:
                    try:
                        # JSONとしてパース
                        obj = json.loads(obj_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse object: {e}")
                        print(f"Content: {obj_str[:100]}...")
                current_obj = []
        elif depth > 0:
            current_obj.append(char)

    print(f"Found {len(json_objects)} JSON objects")

    # 正しいJSONL形式で書き出し（DPO標準の3フィールド）
    with output_file.open('w', encoding='utf-8') as f:
        for i, obj in enumerate(json_objects):
            # DPO必須フィールドのみ抽出
            try:
                dpo_obj = {
                    "prompt": obj["prompt"],
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"]
                }
                # 1行1JSONオブジェクトとして書き出し
                json.dump(dpo_obj, f, ensure_ascii=False)
                f.write('\n')
            except KeyError as e:
                print(f"Warning: Missing field in object {i}: {e}")

    print(f"Successfully wrote {len(json_objects)} records to {output_file}")

    # 検証
    print("\nValidation:")
    with output_file.open('r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        if lines:
            first = json.loads(lines[0])
            print(f"First record fields: {list(first.keys())}")
            print(f"Sample: {json.dumps(first, ensure_ascii=False)[:100]}...")

if __name__ == "__main__":
    input_path = "/workspace/data/dpo/preference_dataset.jsonl"
    print(f"Fixing DPO dataset: {input_path}")
    fix_dpo_dataset(input_path)
    print("Done!")
