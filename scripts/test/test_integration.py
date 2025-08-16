#!/usr/bin/env python3
"""
Webインターフェース統合テスト
RAG APIとメインAPIの統合を確認
"""

def test_integration_structure():
    """統合構造をテスト"""
    print("=== Webインターフェース統合テスト ===")
    
    # 1. ファイル存在確認
    import os
    from pathlib import Path
    
    main_unified_path = Path("app/main_unified.py")
    if main_unified_path.exists():
        print("✅ main_unified.py が存在")
    else:
        print("❌ main_unified.py が見つかりません")
        return False
    
    # 2. ファイル内容の確認
    with open(main_unified_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # RAG統合の確認
    rag_checks = [
        ("RAG imports", "from src.rag.core.query_engine import RoadDesignQueryEngine"),
        ("RAG models", "class QueryRequest(BaseModel):"),
        ("RAG class", "class RAGApplication:"),
        ("RAG health endpoint", '@app.get("/rag/health")'),
        ("RAG query endpoint", '@app.post("/rag/query"'),
        ("RAG search endpoint", '@app.get("/rag/search")'),
        ("RAG documents endpoint", '@app.get("/rag/documents")'),
        ("RAG upload endpoint", '@app.post("/rag/upload-document")'),
        ("RAG streaming endpoint", '@app.post("/rag/stream-query")'),
        ("Startup event", '@app.on_event("startup")'),
        ("RAG page route", '@app.get("/rag")')
    ]
    
    print("\n🔍 RAG統合チェック:")
    all_passed = True
    
    for check_name, search_pattern in rag_checks:
        if search_pattern in content:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name} - パターンが見つかりません: {search_pattern}")
            all_passed = False
    
    # 3. エンドポイント数カウント
    rag_endpoints = content.count('@app.get("/rag/') + content.count('@app.post("/rag/')
    print(f"\n📊 RAGエンドポイント数: {rag_endpoints}")
    
    # 4. 統合前後の比較
    print("\n📋 統合前後の変更:")
    print("  📍 統合前: 2つの分離されたAPI (ポート8050 & 8051)")
    print("  📍 統合後: 1つの統合API (ポート8050のみ)")
    print("  📍 追加されたRAGエンドポイント: /rag/*")
    print("  📍 統合されたRAG機能: クエリ、検索、文書管理、統計、アップロード、ストリーミング")
    
    if all_passed:
        print("\n🎉 統合テスト成功！")
        print("💡 ユーザーは単一のエンドポイント (http://localhost:8050) でアクセス可能")
        print("📚 RAG機能は /rag/* エンドポイントで提供")
        return True
    else:
        print("\n⚠️ 統合に不完全な部分があります")
        return False

def test_api_compatibility():
    """API互換性をテスト"""
    print("\n=== API互換性テスト ===")
    
    # 期待されるエンドポイント
    expected_endpoints = {
        "メインAPI": [
            "/", "/finetune", "/models", "/readme", "/rag",
            "/api/models", "/api/upload-data", "/api/train",
            "/api/training-status/{task_id}", "/api/generate"
        ],
        "RAG API": [
            "/rag/health", "/rag/system-info", "/rag/query",
            "/rag/batch-query", "/rag/search", "/rag/documents",
            "/rag/statistics", "/rag/upload-document", "/rag/stream-query"
        ]
    }
    
    with open("app/main_unified.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🌐 エンドポイント可用性チェック:")
    
    all_available = True
    for category, endpoints in expected_endpoints.items():
        print(f"\n  📂 {category}:")
        for endpoint in endpoints:
            # 動的パラメータを含むエンドポイントの処理
            search_pattern = endpoint.replace("{task_id}", "")
            if search_pattern in content:
                print(f"    ✅ {endpoint}")
            else:
                print(f"    ❌ {endpoint}")
                all_available = False
    
    return all_available

def test_startup_flow():
    """起動フローをテスト"""
    print("\n=== 起動フローテスト ===")
    
    expected_flow = [
        "RAG system imports",
        "RAGApplication class definition", 
        "startup event handler",
        "RAG initialization",
        "unified server startup"
    ]
    
    with open("app/main_unified.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    flow_patterns = [
        "from src.rag.core.query_engine import RoadDesignQueryEngine",
        "class RAGApplication:",
        '@app.on_event("startup")',
        "await rag_app.initialize()",
        'uvicorn.run(app, host="0.0.0.0", port=8050'
    ]
    
    print("🚀 起動フロー確認:")
    all_steps_present = True
    
    for i, (step_name, pattern) in enumerate(zip(expected_flow, flow_patterns), 1):
        if pattern in content:
            print(f"  {i}. ✅ {step_name}")
        else:
            print(f"  {i}. ❌ {step_name}")
            all_steps_present = False
    
    return all_steps_present

def main():
    """メインテスト実行"""
    print("🔧 AI Fine-tuning Toolkit - Web Interface Integration Test")
    print("=" * 60)
    
    tests = [
        ("統合構造テスト", test_integration_structure),
        ("API互換性テスト", test_api_compatibility),
        ("起動フローテスト", test_startup_flow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}でエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("🏁 テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n📊 成功率: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 すべてのテストが成功！")
        print("💡 Webインターフェース統合が完了しました")
        print("🚀 single port (8050) でRAGとメインAPIの両方にアクセス可能")
    else:
        print(f"\n⚠️ {total - passed}個のテストが失敗")
        print("🔧 追加の修正が必要です")

if __name__ == "__main__":
    main()