#!/usr/bin/env python3
"""
AI_FT_3 システム状態レポート

継続学習管理システムとRAGシステムの統合状態を確認します。
"""

import requests
import subprocess
import json
import sys
from datetime import datetime

def check_container_status():
    """コンテナの状態を確認"""
    print("🔍 コンテナ状態の確認")
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ai-ft-container"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "ai-ft-container" in result.stdout:
            print("✅ AI-FTコンテナが実行中です")
            return True
        else:
            print("❌ AI-FTコンテナが見つかりません")
            return False
    except Exception as e:
        print(f"❌ Docker確認エラー: {e}")
        return False

def check_web_server():
    """Webサーバーの状態を確認"""
    print("\n🌐 Webサーバーの確認")
    
    try:
        response = requests.get("http://localhost:8050/", timeout=5)
        if response.status_code == 200:
            print("✅ Webサーバーが正常に動作しています")
            return True
        else:
            print(f"❌ Webサーバーエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Webサーバー接続エラー: {e}")
        return False

def check_continual_learning_system():
    """継続学習管理システムの確認"""
    print("\n🔄 継続学習管理システムの確認")
    
    # API確認
    try:
        response = requests.get("http://localhost:8050/api/continual-learning/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 継続学習API正常: {len(models)}個のモデルを検出")
            
            # ベースモデルの確認
            base_models = [m for m in models if m.get('type') == 'base']
            finetuned_models = [m for m in models if m.get('type') == 'finetuned']
            
            print(f"  - ベースモデル: {len(base_models)}個")
            print(f"  - ファインチューニング済みモデル: {len(finetuned_models)}個")
            
            return True
        else:
            print(f"❌ 継続学習APIエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 継続学習API接続エラー: {e}")
        return False

def check_rag_system():
    """RAGシステムの確認"""
    print("\n🔍 RAGシステムの確認")
    
    try:
        response = requests.get("http://localhost:8050/rag/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ RAGシステムが正常に動作しています")
            print(f"  - サービス: {health_data.get('service', 'N/A')}")
            print(f"  - 状態: {health_data.get('status', 'N/A')}")
            return True
        else:
            print(f"❌ RAGシステムエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ RAGシステム接続エラー: {e}")
        return False

def check_model_management():
    """モデル管理システムの確認"""
    print("\n📊 モデル管理システムの確認")
    
    try:
        response = requests.get("http://localhost:8050/api/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print("✅ モデル管理APIが正常に動作しています")
            
            # 利用可能モデルの確認
            available_models = models_data.get('available_models', [])
            saved_models = models_data.get('saved_models', [])
            
            print(f"  - 利用可能モデル: {len(available_models)}個")
            print(f"  - 保存済みモデル: {len(saved_models)}個")
            
            return True
        else:
            print(f"❌ モデル管理APIエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ モデル管理API接続エラー: {e}")
        return False

def check_system_info():
    """システム情報の確認"""
    print("\n💻 システム情報の確認")
    
    try:
        response = requests.get("http://localhost:8050/api/system-info", timeout=5)
        if response.status_code == 200:
            system_info = response.json()
            print("✅ システム情報APIが正常に動作しています")
            
            # GPU情報
            gpu_info = system_info.get('gpu_info', {})
            if gpu_info:
                print(f"  - GPU: {gpu_info.get('name', 'N/A')}")
                print(f"  - メモリ使用率: {gpu_info.get('memory_usage', 'N/A')}%")
            
            # メモリ情報
            memory_info = system_info.get('memory_info', {})
            if memory_info:
                print(f"  - RAM使用率: {memory_info.get('usage_percent', 'N/A')}%")
            
            return True
        else:
            print(f"❌ システム情報APIエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ システム情報API接続エラー: {e}")
        return False

def generate_report():
    """包括的なシステムレポートを生成"""
    print("=" * 60)
    print("🚀 AI_FT_3 システム状態レポート")
    print("=" * 60)
    print(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 各システムの確認
    checks = [
        ("コンテナ状態", check_container_status),
        ("Webサーバー", check_web_server),
        ("継続学習管理システム", check_continual_learning_system),
        ("RAGシステム", check_rag_system),
        ("モデル管理システム", check_model_management),
        ("システム情報", check_system_info)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}の確認中にエラー: {e}")
            results.append((name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📋 システム状態サマリー")
    print("=" * 60)
    
    success_count = 0
    for name, result in results:
        status = "✅ 正常" if result else "❌ 異常"
        print(f"{name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n総合評価: {success_count}/{len(results)} システムが正常")
    
    if success_count == len(results):
        print("🎉 すべてのシステムが正常に動作しています！")
        print("\n📱 アクセスURL:")
        print("  - メインダッシュボード: http://localhost:8050/")
        print("  - ファインチューニング: http://localhost:8050/finetune")
        print("  - 継続学習管理: http://localhost:8050/continual")
        print("  - RAGシステム: http://localhost:8050/rag")
        print("  - モデル管理: http://localhost:8050/models")
    else:
        print("⚠️  一部のシステムに問題があります。")
        print("詳細なトラブルシューティングは以下を参照してください:")
        print("  - docs/CONTINUAL_LEARNING_TROUBLESHOOTING.md")
    
    print("\n" + "=" * 60)

def main():
    """メイン関数"""
    try:
        generate_report()
    except KeyboardInterrupt:
        print("\n\n中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 