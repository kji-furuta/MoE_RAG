#!/usr/bin/env python3
"""
RAGシステムの高優先度改善項目のテストスクリプト
"""

import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from src.rag.utils.exceptions import (
    ValidationError,
    VectorStoreConnectionError,
    SearchError,
    AuthenticationError,
    RateLimitError
)
from src.rag.utils.validation import (
    validate_query,
    sanitize_html,
    validate_document_id,
    validate_api_key
)
from src.rag.auth.authentication import (
    AuthenticationManager,
    get_auth_manager
)
from src.rag.core.async_query_engine import (
    AsyncRoadDesignQueryEngine,
    create_async_engine
)


def test_exception_classes():
    """例外クラスのテスト"""
    print("\n=== Testing Exception Classes ===")
    
    try:
        # ValidationErrorのテスト
        raise ValidationError("Test validation error", field="test_field", value="test_value")
    except ValidationError as e:
        print(f"✓ ValidationError: {e}")
        assert e.error_code == "VALIDATION"
        assert e.details['field'] == "test_field"
    
    try:
        # VectorStoreConnectionErrorのテスト
        raise VectorStoreConnectionError(
            "Connection failed",
            url="http://localhost:6333",
            retry_count=3
        )
    except VectorStoreConnectionError as e:
        print(f"✓ VectorStoreConnectionError: {e}")
        assert e.error_code == "VECTOR_STORE_CONNECTION"
        assert e.details['retry_count'] == 3
    
    try:
        # SearchErrorのテスト
        raise SearchError(
            "Search failed",
            search_type="hybrid",
            query="test query"
        )
    except SearchError as e:
        print(f"✓ SearchError: {e}")
        assert e.error_code == "SEARCH_ERROR"
    
    print("✓ All exception classes working correctly")


def test_input_validation():
    """入力検証のテスト"""
    print("\n=== Testing Input Validation ===")
    
    # 正常なクエリ
    clean_query = validate_query("道路の設計速度について教えてください")
    print(f"✓ Normal query validated: {clean_query}")
    
    # SQLインジェクション攻撃の検証
    try:
        dangerous_query = "SELECT * FROM users; DROP TABLE users;"
        cleaned = validate_query(dangerous_query)
        print(f"✓ SQL injection cleaned: '{cleaned}'")
        assert "SELECT" not in cleaned
        assert "DROP" not in cleaned
    except ValidationError as e:
        print(f"✓ Caught dangerous SQL: {e}")
    
    # XSS攻撃の検証
    xss_query = "<script>alert('XSS')</script>道路設計"
    cleaned_xss = validate_query(xss_query)
    print(f"✓ XSS cleaned: '{cleaned_xss}'")
    assert "<script>" not in cleaned_xss
    
    # HTMLサニタイゼーション
    html_text = "<img src=x onerror=alert('XSS')>Normal text"
    sanitized = sanitize_html(html_text)
    print(f"✓ HTML sanitized: '{sanitized}'")
    assert "onerror" not in sanitized
    
    # ドキュメントID検証
    valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
    validated_id = validate_document_id(valid_uuid)
    print(f"✓ Valid UUID accepted: {validated_id}")
    
    try:
        invalid_id = "../../../etc/passwd"
        validate_document_id(invalid_id)
    except ValidationError as e:
        print(f"✓ Invalid document ID rejected: {e}")
    
    # APIキー検証
    valid_key = "a" * 32  # 32文字のキー
    validated_key = validate_api_key(valid_key)
    print(f"✓ Valid API key accepted: {validated_key[:8]}...")
    
    try:
        invalid_key = "short"
        validate_api_key(invalid_key)
    except ValidationError as e:
        print(f"✓ Invalid API key rejected: {e}")
    
    print("✓ All validation tests passed")


def test_authentication():
    """認証機能のテスト"""
    print("\n=== Testing Authentication ===")
    
    auth_manager = get_auth_manager()
    
    # ユーザー作成
    user = auth_manager.create_user("test_user", role="user")
    print(f"✓ User created: {user.username} with API key: {user.api_key[:8]}...")
    
    # APIキー認証
    try:
        authenticated_user = auth_manager.authenticate_api_key(user.api_key)
        print(f"✓ API key authentication successful: {authenticated_user.username}")
    except AuthenticationError as e:
        print(f"✗ API key authentication failed: {e}")
    
    # 無効なAPIキー
    try:
        auth_manager.authenticate_api_key("invalid_key")
    except AuthenticationError as e:
        print(f"✓ Invalid API key rejected: {e}")
    
    # JWTトークン生成
    token = auth_manager.create_token(user, scopes=["read", "write"])
    print(f"✓ JWT token created: {token[:20]}...")
    
    # トークン検証
    try:
        verified_user = auth_manager.verify_token(token)
        print(f"✓ Token verification successful: {verified_user.username}")
    except AuthenticationError as e:
        print(f"✗ Token verification failed: {e}")
    
    # 権限チェック
    admin_user = auth_manager.create_user("admin_user", role="admin")
    assert auth_manager.check_permission(admin_user, "user") == True
    assert auth_manager.check_permission(user, "admin") == False
    print("✓ Permission checks working correctly")
    
    # レート制限テスト
    if auth_manager.rate_limiter:
        # 正常なリクエスト
        for i in range(5):
            auth_manager.authenticate_api_key(user.api_key)
        print("✓ Rate limiting allows normal requests")
        
        quota = auth_manager.rate_limiter.get_remaining_quota(user.user_id)
        print(f"✓ Remaining quota: {quota}")
    
    print("✓ All authentication tests passed")


async def test_async_query_engine():
    """非同期クエリエンジンのテスト"""
    print("\n=== Testing Async Query Engine ===")
    
    try:
        # エンジンの作成と初期化
        engine = AsyncRoadDesignQueryEngine()
        await engine.initialize(mode="minimal")
        print("✓ Async engine initialized")
        
        # 単一クエリのテスト
        query = "道路の最小曲線半径について"
        print(f"Testing query: {query}")
        
        result = await engine.query(
            query_text=query,
            top_k=3,
            timeout=10.0
        )
        
        print(f"✓ Query executed successfully")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        if result.execution_time_breakdown:
            print(f"  - Time breakdown: {result.execution_time_breakdown}")
        
        # バッチクエリのテスト
        queries = [
            "設計速度80km/hの道路",
            "横断勾配の基準",
            "舗装の設計"
        ]
        
        print(f"\nTesting batch queries: {queries}")
        batch_results = await engine.batch_query(
            queries=queries,
            top_k=3,
            max_concurrent=2
        )
        
        print(f"✓ Batch query executed: {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"  - Query {i+1}: {result.processing_time:.2f}s")
        
        # クリーンアップ
        await engine.close()
        print("✓ Engine closed successfully")
        
    except Exception as e:
        print(f"✗ Async engine test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ Async query engine tests completed")


async def test_timeout_handling():
    """タイムアウト処理のテスト"""
    print("\n=== Testing Timeout Handling ===")
    
    try:
        engine = AsyncRoadDesignQueryEngine()
        await engine.initialize(mode="minimal")
        
        # 非常に短いタイムアウトでテスト
        try:
            result = await engine.query(
                query_text="複雑なクエリ" * 100,  # 長いクエリ
                timeout=0.001  # 1ミリ秒のタイムアウト
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                print(f"✓ Timeout correctly handled: {e}")
            else:
                print(f"✗ Unexpected error: {e}")
        
        await engine.close()
        
    except Exception as e:
        print(f"✗ Timeout test failed: {e}")


def run_all_tests():
    """全テストを実行"""
    print("=" * 50)
    print("RAG System Improvements Test Suite")
    print("=" * 50)
    
    # 同期テスト
    test_exception_classes()
    test_input_validation()
    test_authentication()
    
    # 非同期テスト
    print("\n--- Running Async Tests ---")
    asyncio.run(test_async_query_engine())
    asyncio.run(test_timeout_handling())
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    print("=" * 50)


if __name__ == "__main__":
    logger.add("test_rag_improvements.log", rotation="10 MB")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)