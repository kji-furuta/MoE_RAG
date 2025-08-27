#!/usr/bin/env python3
"""
RAGシステムの高優先度改善項目の簡易テストスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_exception_classes():
    """例外クラスのテスト"""
    print("\n=== Testing Exception Classes ===")
    
    from src.rag.utils.exceptions import (
        ValidationError,
        VectorStoreConnectionError,
        SearchError,
        AuthenticationError,
        RateLimitError,
        LLMMemoryError,
        QueryTimeoutError
    )
    
    # ValidationErrorのテスト
    try:
        raise ValidationError("Test validation error", field="test_field", value="test_value")
    except ValidationError as e:
        print(f"✓ ValidationError: {e}")
        assert e.error_code == "VALIDATION"
        assert e.details['field'] == "test_field"
    
    # VectorStoreConnectionErrorのテスト
    try:
        raise VectorStoreConnectionError(
            "Connection failed",
            url="http://localhost:6333",
            retry_count=3
        )
    except VectorStoreConnectionError as e:
        print(f"✓ VectorStoreConnectionError: {e}")
        assert e.error_code == "VECTOR_STORE_CONNECTION"
        assert e.details['retry_count'] == 3
    
    # SearchErrorのテスト
    try:
        raise SearchError(
            "Search failed",
            search_type="hybrid",
            query="test query"
        )
    except SearchError as e:
        print(f"✓ SearchError: {e}")
        assert e.error_code == "SEARCH_ERROR"
    
    # AuthenticationErrorのテスト
    try:
        raise AuthenticationError("Invalid credentials")
    except AuthenticationError as e:
        print(f"✓ AuthenticationError: {e}")
        assert e.error_code == "AUTH_ERROR"
    
    # RateLimitErrorのテスト
    try:
        raise RateLimitError("Too many requests", limit=100, window_seconds=60)
    except RateLimitError as e:
        print(f"✓ RateLimitError: {e}")
        assert e.error_code == "RATE_LIMIT"
        assert e.details['limit'] == 100
    
    # LLMMemoryErrorのテスト
    try:
        raise LLMMemoryError("Out of memory", required_memory=8, available_memory=4)
    except LLMMemoryError as e:
        print(f"✓ LLMMemoryError: {e}")
        assert e.error_code == "LLM_MEMORY"
        assert e.details['required_memory_gb'] == 8
    
    # QueryTimeoutErrorのテスト
    try:
        raise QueryTimeoutError("Query timed out", timeout_seconds=30, query="long query")
    except QueryTimeoutError as e:
        print(f"✓ QueryTimeoutError: {e}")
        assert e.error_code == "QUERY_TIMEOUT"
        assert e.details['timeout_seconds'] == 30
    
    print("✓ All exception classes working correctly")


def test_input_validation():
    """入力検証のテスト"""
    print("\n=== Testing Input Validation ===")
    
    from src.rag.utils.validation import (
        validate_query,
        sanitize_html,
        validate_document_id,
        validate_api_key,
        ValidationError
    )
    
    # 正常なクエリ
    clean_query = validate_query("道路の設計速度について教えてください")
    print(f"✓ Normal query validated: {clean_query}")
    
    # SQLインジェクション攻撃の検証
    dangerous_query = "SELECT * FROM users; DROP TABLE users;"
    cleaned = validate_query(dangerous_query)
    print(f"✓ SQL injection cleaned: '{cleaned}'")
    assert "SELECT" not in cleaned.upper()
    assert "DROP" not in cleaned.upper()
    
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
    
    # 無効なドキュメントID
    try:
        invalid_id = "../../../etc/passwd"
        validate_document_id(invalid_id)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"✓ Invalid document ID rejected: {e}")
    
    # APIキー検証
    valid_key = "a" * 32  # 32文字のキー
    validated_key = validate_api_key(valid_key)
    print(f"✓ Valid API key accepted: {validated_key[:8]}...")
    
    # 無効なAPIキー
    try:
        invalid_key = "short"
        validate_api_key(invalid_key)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"✓ Invalid API key rejected: {e}")
    
    # 長すぎるクエリ
    try:
        long_query = "a" * 3000  # 3000文字
        validate_query(long_query)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"✓ Long query rejected: {e}")
    
    # 空のクエリ
    try:
        validate_query("", allow_empty=False)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        print(f"✓ Empty query rejected: {e}")
    
    print("✓ All validation tests passed")


def test_authentication_basic():
    """認証機能の基本テスト（外部依存なし）"""
    print("\n=== Testing Authentication (Basic) ===")
    
    # 必要な依存関係をモック
    import sys
    import types
    
    # jwtモジュールのモック
    jwt_mock = types.ModuleType('jwt')
    
    def encode_mock(payload, secret, algorithm):
        import json
        import base64
        return base64.b64encode(json.dumps(payload).encode()).decode()
    
    def decode_mock(token, secret, algorithms):
        import json
        import base64
        return json.loads(base64.b64decode(token))
    
    jwt_mock.encode = encode_mock
    jwt_mock.decode = decode_mock
    jwt_mock.ExpiredSignatureError = Exception
    jwt_mock.InvalidTokenError = Exception
    
    sys.modules['jwt'] = jwt_mock
    
    # 認証モジュールをインポート
    from src.rag.auth.authentication import (
        AuthenticationManager,
        User,
        AuthToken,
        RateLimiter
    )
    
    # AuthenticationManagerのテスト
    auth_manager = AuthenticationManager(
        secret_key="test_secret_key",
        token_expiry_hours=1,
        enable_rate_limiting=True
    )
    print("✓ AuthenticationManager created")
    
    # ユーザー作成
    user = auth_manager.create_user("test_user", role="user")
    print(f"✓ User created: {user.username} with API key: {user.api_key[:8]}...")
    assert user.username == "test_user"
    assert user.role == "user"
    assert user.api_key is not None
    
    # APIキー認証
    authenticated_user = auth_manager.authenticate_api_key(user.api_key)
    print(f"✓ API key authentication successful: {authenticated_user.username}")
    assert authenticated_user.user_id == user.user_id
    
    # 無効なAPIキー
    try:
        from src.rag.utils.exceptions import AuthenticationError
        auth_manager.authenticate_api_key("invalid_key_12345")
        assert False, "Should have raised AuthenticationError"
    except AuthenticationError as e:
        print(f"✓ Invalid API key rejected: {e}")
    
    # 権限チェック
    admin_user = auth_manager.create_user("admin_user", role="admin")
    assert auth_manager.check_permission(admin_user, "user") == True
    assert auth_manager.check_permission(admin_user, "admin") == True
    assert auth_manager.check_permission(user, "admin") == False
    assert auth_manager.check_permission(user, "user") == True
    print("✓ Permission checks working correctly")
    
    # レート制限テスト
    rate_limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
    
    # 正常なリクエスト
    for i in range(4):
        rate_limiter.check_rate_limit("user1")
    print("✓ Rate limiter allows normal requests")
    
    # クォータ確認
    quota = rate_limiter.get_remaining_quota("user1")
    assert quota['per_minute'] == 1  # 5 - 4 = 1
    print(f"✓ Remaining quota calculated correctly: {quota}")
    
    # レート制限超過
    try:
        from src.rag.utils.exceptions import RateLimitError
        for i in range(3):  # これで合計7リクエスト（制限は5）
            rate_limiter.check_rate_limit("user1")
        assert False, "Should have raised RateLimitError"
    except RateLimitError as e:
        print(f"✓ Rate limit exceeded correctly: {e}")
    
    print("✓ All authentication tests passed")


def test_error_code_consistency():
    """エラーコードの一貫性テスト"""
    print("\n=== Testing Error Code Consistency ===")
    
    from src.rag.utils.exceptions import (
        RAGException,
        ValidationError,
        VectorStoreError,
        SearchError,
        GenerationError,
        ModelLoadError,
        ConfigurationError
    )
    
    # エラーコードが正しく設定されているか確認
    errors_to_test = [
        (ValidationError("test", field="f", value="v"), "VALIDATION"),
        (VectorStoreError("test", operation="op"), "VECTOR_STORE"),
        (SearchError("test", search_type="hybrid"), "SEARCH_ERROR"),
        (GenerationError("test", model_name="model"), "GENERATION_ERROR"),
        (ModelLoadError("test", model_name="model"), "MODEL_LOAD"),
        (ConfigurationError("test", config_file="file.yaml"), "CONFIG_ERROR"),
    ]
    
    for error, expected_code in errors_to_test:
        assert error.error_code == expected_code, f"Expected {expected_code}, got {error.error_code}"
        print(f"✓ {error.__class__.__name__} has correct error code: {expected_code}")
    
    # 詳細情報が正しく含まれているか確認
    ve = ValidationError("test", field="test_field", value="test_value")
    assert ve.details['field'] == "test_field"
    assert ve.details['value'] == "test_value"
    print("✓ Error details are correctly stored")
    
    # __str__メソッドが正しく動作するか確認
    error_str = str(ve)
    assert "test" in error_str
    assert "VALIDATION" in error_str
    assert "field" in error_str
    print(f"✓ Error string representation works: {error_str[:50]}...")
    
    print("✓ All error code consistency tests passed")


def run_all_tests():
    """全テストを実行"""
    print("=" * 50)
    print("RAG System Improvements Test Suite (Simplified)")
    print("=" * 50)
    
    test_exception_classes()
    test_input_validation()
    test_authentication_basic()
    test_error_code_consistency()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    print("=" * 50)
    print("\nSummary of improvements implemented:")
    print("1. ✓ Specific exception classes with error codes")
    print("2. ✓ Enhanced input validation and sanitization")
    print("3. ✓ SQL injection and XSS protection")
    print("4. ✓ Authentication and authorization system")
    print("5. ✓ Rate limiting mechanism")
    print("6. ✓ API key and JWT token support")


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)