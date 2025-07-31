#!/usr/bin/env python3
"""
オンラインでモデルの存在をチェックするスクリプト
"""

import json
import requests
import time

def load_models():
    """available_models.jsonからモデルリストを読み込み"""
    with open('available_models.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['available_models']

def check_model_online(model_name):
    """オンラインでモデルの存在をチェック"""
    try:
        # Hugging FaceのモデルページURL
        url = f"https://huggingface.co/{model_name}"
        
        print(f"チェック中: {model_name}")
        print(f"  URL: {url}")
        
        # HTTPリクエストでページの存在を確認
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return True, "モデルが存在します"
        elif response.status_code == 404:
            return False, "モデルが見つかりません"
        elif response.status_code == 403:
            return False, "アクセス権限がありません（プライベートモデル）"
        else:
            return False, f"HTTPエラー: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "タイムアウト"
    except requests.exceptions.ConnectionError:
        return False, "接続エラー"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def main():
    """メイン処理"""
    print("=== オンラインモデルチェック ===\n")
    
    models = load_models()
    results = []
    
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}]")
        success, message = check_model_online(model)
        results.append({
            'model': model,
            'success': success,
            'message': message
        })
        print(f"  結果: {'✅ 存在' if success else '❌ 不存在'}")
        print(f"  メッセージ: {message}")
        print()
        
        # API制限を避けるため少し待機
        time.sleep(0.5)
    
    # サマリー
    success_count = sum(1 for r in results if r['success'])
    print(f"=== 結果サマリー ===")
    print(f"存在: {success_count}/{len(models)}")
    print(f"不存在: {len(models) - success_count}/{len(models)}")
    
    # 存在しないモデルを表示
    failed_models = [r for r in results if not r['success']]
    if failed_models:
        print("\n=== 存在しないモデル ===")
        for model in failed_models:
            print(f"• {model['model']}: {model['message']}")

if __name__ == "__main__":
    main() 