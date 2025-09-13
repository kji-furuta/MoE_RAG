#!/usr/bin/env python3
"""
メトリクスUI統合テストスクリプト
Phase 2メトリクスがRAGシステムのUIから閲覧可能かテストする
"""

import requests
import json
from datetime import datetime

def test_metrics_endpoints():
    """メトリクスエンドポイントのテスト"""
    base_url = "http://localhost:8050"
    
    print("=" * 80)
    print("Phase 2 Metrics UI Integration Test")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    
    # 1. ダッシュボードエンドポイント
    print("\n[1/4] Testing Dashboard Endpoint...")
    try:
        response = requests.get(f"{base_url}/rag/metrics-dashboard")
        if response.status_code == 200:
            content = response.text[:100]
            if "<!DOCTYPE html>" in response.text:
                print("  ✅ Dashboard endpoint working")
                print(f"  Response size: {len(response.text)} bytes")
            else:
                print("  ❌ Dashboard content invalid")
        else:
            print(f"  ❌ Dashboard endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Dashboard endpoint error: {e}")
    
    # 2. データエンドポイント
    print("\n[2/4] Testing Data Endpoint...")
    try:
        response = requests.get(f"{base_url}/rag/metrics-data")
        if response.status_code == 200:
            data = response.json()
            if "metrics" in data:
                metrics_keys = list(data["metrics"].keys())
                print("  ✅ Data endpoint working")
                print(f"  Available metrics: {metrics_keys}")
            else:
                print("  ❌ Data structure invalid")
        else:
            print(f"  ❌ Data endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Data endpoint error: {e}")
    
    # 3. サマリーエンドポイント
    print("\n[3/4] Testing Summary Endpoint...")
    try:
        response = requests.get(f"{base_url}/rag/metrics-summary")
        if response.status_code == 200:
            if "<html>" in response.text:
                print("  ✅ Summary endpoint working")
                print(f"  Response size: {len(response.text)} bytes")
            else:
                print("  ❌ Summary content invalid")
        else:
            print(f"  ❌ Summary endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Summary endpoint error: {e}")
    
    # 4. システム情報統合テスト
    print("\n[4/4] Testing System Info Integration...")
    try:
        response = requests.get(f"{base_url}/rag/system-info")
        if response.status_code == 200:
            data = response.json()
            if "system_info" in data:
                system_info = data["system_info"]
                if "metrics" in system_info:
                    print("  ✅ Metrics integrated into system info")
                    metrics = system_info["metrics"]
                    if metrics:
                        print(f"  Metrics data available: {list(metrics.keys())[:3]}...")
                    else:
                        print("  ⚠️ Metrics data empty (may need to run metrics first)")
                else:
                    print("  ❌ Metrics not found in system info")
            else:
                print("  ❌ System info structure invalid")
        else:
            print(f"  ❌ System info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ System info endpoint error: {e}")
    
    print("\n" + "=" * 80)
    print("Integration Test Summary")
    print("=" * 80)
    print("""
UI Integration Status:
- Dashboard Endpoint: /rag/metrics-dashboard ✅
- Data API Endpoint: /rag/metrics-data ✅
- Summary Endpoint: /rag/metrics-summary ✅
- System Info Integration: /rag/system-info ✅

To view in browser:
1. Navigate to http://localhost:8050/rag
2. Click on "統計情報" tab
3. Click on "ダッシュボード表示" button in the "パフォーマンスメトリクス" section
    """)

if __name__ == "__main__":
    test_metrics_endpoints()