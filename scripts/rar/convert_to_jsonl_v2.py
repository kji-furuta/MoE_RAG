#!/usr/bin/env python3
"""
複数行フォーマットJSONをJSONL形式に変換（改良版）
"""

import json
import re

def convert_multiline_json_to_jsonl(input_path, output_path):
    """
    連続する複数行JSONオブジェクトをJSONL形式に変換

    入力例:
    {
      "id": "001",
      ...
    }
    {
      "id": "002",
      ...
    }

    出力例:
    {"id": "001", ...}
    {"id": "002", ...}
    """

    print(f"📥 入力: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 方法1: 正規表現でトップレベルの{}を抽出
    # パターン: { から始まり、対応する } まで（ネストを考慮）
    entries = []
    depth = 0
    current_obj = ""
    in_string = False
    escape = False

    for i, char in enumerate(content):
        if escape:
            escape = False
            current_obj += char
            continue

        if char == '\\':
            escape = True
            current_obj += char
            continue

        if char == '"' and not escape:
            in_string = not in_string
            current_obj += char
            continue

        if in_string:
            current_obj += char
            continue

        # 文字列外の括弧カウント
        if char == '{':
            depth += 1
            current_obj += char
        elif char == '}':
            current_obj += char
            depth -= 1

            # トップレベルのオブジェクト終了
            if depth == 0 and current_obj.strip():
                try:
                    obj = json.loads(current_obj.strip())
                    entries.append(obj)
                except json.JSONDecodeError as e:
                    print(f"⚠️  パースエラー: {current_obj[:100]}... エラー: {e}")
                current_obj = ""
        else:
            current_obj += char

    print(f"✅ 解析完了: {len(entries)}件のエントリー")

    # JSONL形式で出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"📤 出力: {output_path}")
    print(f"✅ 変換完了")

    return len(entries)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("使用方法: python convert_to_jsonl_v2.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    count = convert_multiline_json_to_jsonl(sys.argv[1], sys.argv[2])
    print(f"\n✅ {count}件のエントリーをJSONL形式に変換しました")
