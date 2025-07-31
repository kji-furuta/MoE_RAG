#!/usr/bin/env python3
"""
ファインチューニング用データの前処理スクリプト

使用方法:
    python scripts/prepare_training_data.py --input data/raw/sample_training_data.jsonl --output data/processed/training_data.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """JSONLファイルに保存する"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def format_for_training(raw_data: List[Dict[str, Any]], format_type: str = "text") -> List[Dict[str, Any]]:
    """
    データをファインチューニング用の形式に変換
    
    Args:
        raw_data: 生データ
        format_type: 出力形式 ("text", "instruction", "chat")
    """
    processed_data = []
    
    for item in raw_data:
        if format_type == "text":
            # シンプルなテキスト形式
            if "text" in item:
                processed_data.append({"text": item["text"]})
            elif "instruction" in item and "output" in item:
                # instruction形式をtext形式に変換
                text = f"質問: {item['instruction']}\n回答: {item['output']}"
                processed_data.append({"text": text})
                
        elif format_type == "instruction":
            # Instruction形式
            if "instruction" in item:
                processed_item = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                }
                processed_data.append(processed_item)
            elif "text" in item and "質問:" in item["text"] and "回答:" in item["text"]:
                # テキスト形式をinstruction形式に変換
                parts = item["text"].split("\n回答:")
                if len(parts) == 2:
                    instruction = parts[0].replace("質問:", "").strip()
                    output = parts[1].strip()
                    processed_data.append({
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    })
                    
        elif format_type == "chat":
            # チャット形式
            if "instruction" in item:
                processed_item = {
                    "conversations": [
                        {"from": "human", "value": item["instruction"]},
                        {"from": "assistant", "value": item["output"]}
                    ]
                }
                processed_data.append(processed_item)
    
    return processed_data

def validate_data(data: List[Dict[str, Any]]) -> bool:
    """データの妥当性を検証"""
    if not data:
        print("エラー: データが空です")
        return False
    
    # 各レコードの検証
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"エラー: レコード {i+1} が辞書形式ではありません")
            return False
        
        # 必要なフィールドの確認
        if "text" not in item and "instruction" not in item and "conversations" not in item:
            print(f"エラー: レコード {i+1} に必要なフィールドがありません")
            return False
    
    print(f"検証完了: {len(data)} レコードが正常です")
    return True

def main():
    parser = argparse.ArgumentParser(description="ファインチューニング用データの前処理")
    parser.add_argument("--input", required=True, help="入力JSONLファイルのパス")
    parser.add_argument("--output", required=True, help="出力JSONLファイルのパス")
    parser.add_argument("--format", choices=["text", "instruction", "chat"], default="text", 
                       help="出力データの形式")
    parser.add_argument("--validate", action="store_true", help="データの妥当性を検証")
    
    args = parser.parse_args()
    
    # データの読み込み
    print(f"データを読み込み中: {args.input}")
    raw_data = load_jsonl(args.input)
    print(f"読み込み完了: {len(raw_data)} レコード")
    
    # データの変換
    print(f"データを {args.format} 形式に変換中...")
    processed_data = format_for_training(raw_data, args.format)
    print(f"変換完了: {len(processed_data)} レコード")
    
    # データの検証
    if args.validate:
        if not validate_data(processed_data):
            return
    
    # データの保存
    print(f"データを保存中: {args.output}")
    save_jsonl(processed_data, args.output)
    print("処理完了!")
    
    # 統計情報の表示
    print("\n=== 統計情報 ===")
    print(f"入力レコード数: {len(raw_data)}")
    print(f"出力レコード数: {len(processed_data)}")
    print(f"出力形式: {args.format}")
    
    # サンプルデータの表示
    if processed_data:
        print("\n=== サンプルデータ ===")
        print(json.dumps(processed_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()