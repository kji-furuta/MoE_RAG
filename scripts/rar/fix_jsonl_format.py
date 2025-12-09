#!/usr/bin/env python3
"""
整形されたJSON配列をJSONL形式に変換

入力: 複数行フォーマットのJSONファイル
出力: 1行1オブジェクトのJSONLファイル
"""

import json
import sys
from pathlib import Path

def convert_to_jsonl(input_path: str, output_path: str = None):
    """JSON配列またはマルチラインJSONをJSONL形式に変換"""

    if output_path is None:
        output_path = input_path.replace('.jsonl', '_fixed.jsonl')

    print(f"📥 入力: {input_path}")
    print(f"📤 出力: {output_path}")

    # ファイル全体を読み込んで結合
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # JSON配列として解析を試みる
    try:
        # まず配列として解析を試みる
        if content.strip().startswith('['):
            data = json.loads(content)
            print(f"✅ JSON配列として解析成功: {len(data)}件のエントリー")
        else:
            # 複数行にわたる個別JSONオブジェクトとして解析
            entries = []
            current_obj = ""
            brace_count = 0
            in_string = False
            escape_next = False

            for char in content:
                # 文字列内のチェック（エスケープ処理）
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string

                # 括弧カウント（文字列外のみ）
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                current_obj += char

                # オブジェクト完了時にパース
                if brace_count == 0 and current_obj.strip() and current_obj.strip().startswith('{'):
                    try:
                        obj = json.loads(current_obj.strip())
                        entries.append(obj)
                        current_obj = ""
                    except json.JSONDecodeError:
                        # パース失敗した場合は蓄積継続
                        pass

            data = entries
            print(f"✅ 個別オブジェクトとして解析: {len(data)}件のエントリー")

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析エラー: {e}")
        return False

    # JSONL形式で出力
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ 変換完了: {len(data)}件のエントリーをJSONL形式で出力")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python fix_jsonl_format.py <input.jsonl> [output.jsonl]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_to_jsonl(input_file, output_file)
    sys.exit(0 if success else 1)
