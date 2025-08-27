"""
RAGシステムの認証・認可メカニズム
"""

import time
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import jwt
import logging
logger = logging.getLogger(__name__)

from ..utils.exceptions import AuthenticationError, RateLimitError


@dataclass
class User:
    """ユーザーエンティティ"""
    user_id: str
    username: str
    role: str = "user"
    api_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_access: datetime = field(default_factory=datetime.now)


@dataclass
class AuthToken:
    """認証トークン"""
    token: str
    user_id: str
    expires_at: datetime
    scopes: List[str] = field(default_factory=list)


class AuthenticationManager:
    """認証管理クラス"""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry_hours: int = 24,
        enable_rate_limiting: bool = True
    ):
        """
        初期化
        
        Args:
            secret_key: JWT署名用の秘密鍵
            token_expiry_hours: トークン有効期限（時間）
            enable_rate_limiting: レート制限を有効にするか
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry_hours = token_expiry_hours
        self.enable_rate_limiting = enable_rate_limiting
        
        # ユーザーストア（本番環境ではデータベースを使用）
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.tokens: Dict[str, AuthToken] = {}  # token -> AuthToken
        
        # レート制限用のカウンター
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        
        # デフォルトの管理者ユーザーを作成
        self._create_default_admin()
    
    def _create_default_admin(self):
        """デフォルトの管理者ユーザーを作成"""
        admin_api_key = secrets.token_urlsafe(32)
        admin_user = User(
            user_id="admin",
            username="admin",
            role="admin",
            api_key=admin_api_key
        )
        self.users["admin"] = admin_user
        self.api_keys[admin_api_key] = "admin"
        logger.info(f"Default admin created with API key: {admin_api_key[:8]}...")
    
    def create_user(
        self,
        username: str,
        role: str = "user",
        generate_api_key: bool = True
    ) -> User:
        """
        新規ユーザー作成
        
        Args:
            username: ユーザー名
            role: ユーザーロール
            generate_api_key: APIキーを生成するか
        
        Returns:
            作成されたユーザー
        """
        user_id = hashlib.sha256(username.encode()).hexdigest()[:16]
        
        if user_id in self.users:
            raise ValueError(f"User {username} already exists")
        
        api_key = None
        if generate_api_key:
            api_key = secrets.token_urlsafe(32)
            self.api_keys[api_key] = user_id
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            api_key=api_key
        )
        
        self.users[user_id] = user
        logger.info(f"User created: {username} (role: {role})")
        
        return user
    
    def authenticate_api_key(self, api_key: str) -> User:
        """
        APIキーによる認証
        
        Args:
            api_key: APIキー
        
        Returns:
            認証されたユーザー
        
        Raises:
            AuthenticationError: 認証失敗時
        """
        if api_key not in self.api_keys:
            raise AuthenticationError("Invalid API key")
        
        user_id = self.api_keys[api_key]
        user = self.users.get(user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        # アクセス時刻を更新
        user.last_access = datetime.now()
        
        # レート制限チェック
        if self.rate_limiter:
            self.rate_limiter.check_rate_limit(user_id)
        
        return user
    
    def create_token(self, user: User, scopes: List[str] = None) -> str:
        """
        JWTトークンの生成
        
        Args:
            user: ユーザー
            scopes: 権限スコープ
        
        Returns:
            JWTトークン
        """
        expires_at = datetime.now() + timedelta(hours=self.token_expiry_hours)
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role,
            'scopes': scopes or [],
            'exp': expires_at.timestamp(),
            'iat': datetime.now().timestamp()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # トークンを保存
        auth_token = AuthToken(
            token=token,
            user_id=user.user_id,
            expires_at=expires_at,
            scopes=scopes or []
        )
        self.tokens[token] = auth_token
        
        return token
    
    def verify_token(self, token: str) -> User:
        """
        JWTトークンの検証
        
        Args:
            token: JWTトークン
        
        Returns:
            認証されたユーザー
        
        Raises:
            AuthenticationError: トークン検証失敗時
        """
        try:
            # トークンをデコード
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            user_id = payload.get('user_id')
            if not user_id or user_id not in self.users:
                raise AuthenticationError("Invalid token: user not found")
            
            user = self.users[user_id]
            
            # レート制限チェック
            if self.rate_limiter:
                self.rate_limiter.check_rate_limit(user_id)
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def check_permission(self, user: User, required_role: str) -> bool:
        """
        権限チェック
        
        Args:
            user: ユーザー
            required_role: 必要なロール
        
        Returns:
            権限があるか
        """
        role_hierarchy = {
            'admin': 3,
            'power_user': 2,
            'user': 1,
            'guest': 0
        }
        
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def revoke_token(self, token: str):
        """トークンの無効化"""
        if token in self.tokens:
            del self.tokens[token]
            logger.info("Token revoked")
    
    def revoke_api_key(self, api_key: str):
        """APIキーの無効化"""
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            del self.api_keys[api_key]
            
            if user_id in self.users:
                self.users[user_id].api_key = None
            
            logger.info(f"API key revoked for user: {user_id}")


class RateLimiter:
    """レート制限管理クラス"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        初期化
        
        Args:
            requests_per_minute: 分あたりのリクエスト上限
            requests_per_hour: 時間あたりのリクエスト上限
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # ユーザーごとのリクエスト履歴
        self.request_history: Dict[str, List[float]] = defaultdict(list)
    
    def check_rate_limit(self, user_id: str):
        """
        レート制限チェック
        
        Args:
            user_id: ユーザーID
        
        Raises:
            RateLimitError: レート制限超過時
        """
        current_time = time.time()
        history = self.request_history[user_id]
        
        # 古いエントリを削除
        history = [t for t in history if current_time - t < 3600]
        
        # 分あたりのチェック
        recent_minute = [t for t in history if current_time - t < 60]
        if len(recent_minute) >= self.requests_per_minute:
            raise RateLimitError(
                "Rate limit exceeded (per minute)",
                limit=self.requests_per_minute,
                window_seconds=60
            )
        
        # 時間あたりのチェック
        if len(history) >= self.requests_per_hour:
            raise RateLimitError(
                "Rate limit exceeded (per hour)",
                limit=self.requests_per_hour,
                window_seconds=3600
            )
        
        # リクエストを記録
        history.append(current_time)
        self.request_history[user_id] = history
    
    def get_remaining_quota(self, user_id: str) -> Dict[str, int]:
        """
        残りクォータの取得
        
        Args:
            user_id: ユーザーID
        
        Returns:
            残りリクエスト数
        """
        current_time = time.time()
        history = self.request_history.get(user_id, [])
        
        recent_minute = [t for t in history if current_time - t < 60]
        recent_hour = [t for t in history if current_time - t < 3600]
        
        return {
            'per_minute': self.requests_per_minute - len(recent_minute),
            'per_hour': self.requests_per_hour - len(recent_hour)
        }


# シングルトンインスタンス
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """認証マネージャーのシングルトンインスタンスを取得"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


def require_auth(required_role: str = "user"):
    """
    認証が必要なエンドポイント用のデコレータ
    
    Args:
        required_role: 必要なロール
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # リクエストから認証情報を取得
            # FastAPIの場合の例
            from fastapi import Header, HTTPException
            
            auth_header = kwargs.get('authorization')
            if not auth_header:
                raise HTTPException(status_code=401, detail="Authorization required")
            
            # Bearer トークンまたはAPIキーを解析
            auth_manager = get_auth_manager()
            
            try:
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    user = auth_manager.verify_token(token)
                elif auth_header.startswith("ApiKey "):
                    api_key = auth_header[7:]
                    user = auth_manager.authenticate_api_key(api_key)
                else:
                    raise HTTPException(status_code=401, detail="Invalid authorization format")
                
                # 権限チェック
                if not auth_manager.check_permission(user, required_role):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # ユーザー情報をkwargsに追加
                kwargs['current_user'] = user
                
                return func(*args, **kwargs)
                
            except AuthenticationError as e:
                raise HTTPException(status_code=401, detail=str(e))
            except RateLimitError as e:
                raise HTTPException(status_code=429, detail=str(e))
        
        return wrapper
    return decorator