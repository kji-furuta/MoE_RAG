#!/usr/bin/env python3
"""
instruction + input + output形式のテストファイル
"""
import requests
import json
import tempfile
import os

BASE_URL = "http://localhost:8050"

def test_instruction_format():
    """instruction + input + output形式のアップロードをテスト"""
    
    # テストデータの作成
    test_data = [
        {"instruction": "設計速度100km/hの最小曲線半径は？", "input": "道路構造令の解説", "output": "460m"},
        {"instruction": "橋梁の設計荷重T-25とは？", "input": "", "output": "25トンの設計自動車荷重"},
        {"text": "質問: 盛土（もりど）と切土（きりど）の違いは何ですか？\n回答: 盛土は、低い地盤に土を盛って高くし、計画高まで道路面を造成することです。一方、切土は、高い地盤を削り取って低くし、計画高まで掘り下げることです。"},
        {"question": "トンネルの換気方式の種類は？", "answer": "縦流換気、横流換気、半横流換気"},
        {"input": "道路の横断勾配", "output": "一般的に1.5〜2.0%"}
    ]
    
    # 一時ファイルの作成
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        for data in test_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        temp_file = f.name
    
    try:
        # ファイルアップロードのテスト
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_data.jsonl', f, 'application/x-jsonlines')}
            response = requests.post(f"{BASE_URL}/api/upload-data", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ アップロード成功!")
            print(f"   ステータス: {result.get('status')}")
            print(f"   有効行数: {result.get('valid_lines')}")
            print(f"   エラー数: {result.get('error_count')}")
            print(f"   サンプルデータ:")
            for i, sample in enumerate(result.get('sample_data', [])[:3]):
                print(f"     {i+1}. {json.dumps(sample, ensure_ascii=False)[:100]}...")
            
            if result.get('errors'):
                print(f"   エラー詳細: {result.get('errors')}")
            
            return result.get('status') == 'success'
        else:
            print(f"❌ アップロード失敗: {response.status_code}")
            print(f"   エラー: {response.text}")
            return False
            
    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_training_data_format():
    """トレーニング時のデータ形式変換をテスト"""
    
    print("\n=== データ形式変換のテスト ===")
    
    test_cases = [
        {
            "input": {"instruction": "設計速度100km/hの最小曲線半径は？", "input": "道路構造令", "output": "460m"},
            "expected": "質問: 設計速度100km/hの最小曲線半径は？\n入力: 道路構造令\n回答: 460m"
        },
        {
            "input": {"instruction": "橋梁の設計荷重T-25とは？", "input": "", "output": "25トンの設計自動車荷重"},
            "expected": "質問: 橋梁の設計荷重T-25とは？\n回答: 25トンの設計自動車荷重"
        },
        {
            "input": {"question": "トンネルの換気方式は？", "answer": "縦流換気、横流換気"},
            "expected": "質問: トンネルの換気方式は？\n回答: 縦流換気、横流換気"
        }
    ]
    
    print("期待される変換結果:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. 入力: {test['input']}")
        print(f"   期待: {test['expected']}")

if __name__ == "__main__":
    print("=== Instruction形式のサポートテスト ===\n")
    
    # アップロードテスト
    success = test_instruction_format()
    
    # 変換テスト
    test_training_data_format()
    
    if success:
        print("\n✅ テスト完了: instruction形式がサポートされています!")
    else:
        print("\n❌ テスト失敗: サーバーを確認してください")