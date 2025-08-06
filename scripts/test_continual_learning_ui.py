#!/usr/bin/env python3
"""
継続学習管理システムUIテストスクリプト

ベースモデル選択のプルダウンメニューが正しく表示されるかをテストします。
"""

import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_api_endpoint():
    """APIエンドポイントのテスト"""
    print("🔍 APIエンドポイントのテスト")
    
    try:
        response = requests.get("http://localhost:8050/api/continual-learning/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ API正常: {len(models)}個のモデルを取得")
            for i, model in enumerate(models[:3]):
                print(f"  {i+1}. {model['name']} ({model['type']})")
            return True
        else:
            print(f"❌ APIエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API接続エラー: {e}")
        return False

def test_web_page():
    """Webページのテスト"""
    print("\n🌐 Webページのテスト")
    
    try:
        response = requests.get("http://localhost:8050/continual", timeout=10)
        if response.status_code == 200:
            print("✅ 継続学習管理ページにアクセス成功")
            
            # HTMLの内容を確認
            if "baseModel" in response.text:
                print("✅ baseModel要素が存在します")
            else:
                print("❌ baseModel要素が見つかりません")
                
            if "モデルを選択してください" in response.text:
                print("✅ デフォルトオプションが存在します")
            else:
                print("❌ デフォルトオプションが見つかりません")
                
            return True
        else:
            print(f"❌ Webページエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Webページ接続エラー: {e}")
        return False

def test_browser_interaction():
    """ブラウザでの動作テスト"""
    print("\n🌐 ブラウザでの動作テスト")
    
    try:
        # Chromeオプションを設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # ヘッドレスモード
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # WebDriverを初期化
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # ページにアクセス
            driver.get("http://localhost:8050/continual")
            print("✅ ページにアクセス成功")
            
            # ページが読み込まれるまで待機
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "baseModel"))
            )
            print("✅ baseModel要素が読み込まれました")
            
            # セレクト要素を取得
            select_element = driver.find_element(By.ID, "baseModel")
            options = select_element.find_elements(By.TAG_NAME, "option")
            
            print(f"✅ 選択肢の数: {len(options)}")
            
            # 選択肢の内容を確認
            for i, option in enumerate(options[:5]):  # 最初の5つを表示
                print(f"  {i+1}. {option.text}")
            
            # JavaScriptの実行をテスト
            driver.execute_script("""
                // モデル一覧を取得するテスト
                fetch('/api/continual-learning/models')
                    .then(response => response.json())
                    .then(models => {
                        console.log('取得したモデル数:', models.length);
                        return models;
                    })
                    .catch(error => {
                        console.error('APIエラー:', error);
                    });
            """)
            
            # 少し待機してJavaScriptの実行を確認
            time.sleep(3)
            
            # 再度選択肢を確認
            select_element = driver.find_element(By.ID, "baseModel")
            options = select_element.find_elements(By.TAG_NAME, "option")
            
            print(f"✅ JavaScript実行後の選択肢の数: {len(options)}")
            
            if len(options) > 1:
                print("✅ ベースモデルの選択肢が正しく表示されています")
                return True
            else:
                print("❌ ベースモデルの選択肢が表示されていません")
                return False
                
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"❌ ブラウザテストエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("=" * 60)
    print("🚀 継続学習管理システムUIテスト")
    print("=" * 60)
    
    # 各テストを実行
    tests = [
        ("APIエンドポイント", test_api_endpoint),
        ("Webページ", test_web_page),
        ("ブラウザでの動作", test_browser_interaction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テストでエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print("=" * 60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n総合評価: {success_count}/{len(results)} テストが成功")
    
    if success_count == len(results):
        print("🎉 すべてのテストが成功しました！")
        print("継続学習管理システムのベースモデル選択は正常に動作しています。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("ブラウザで http://localhost:8050/continual にアクセスして確認してください。")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 