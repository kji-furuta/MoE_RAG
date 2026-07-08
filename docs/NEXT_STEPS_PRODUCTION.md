# 次期実装計画 - セキュリティ強化とプロダクション準備

## 📅 実装ロードマップ

### Phase 3: セキュリティ強化（1週間）

#### Week 1: 基本セキュリティ
1. **認証システム**
   - JWT認証の実装
   - APIキー管理
   - セッション管理

2. **データ保護**
   - 入力検証強化
   - SQLインジェクション対策
   - XSS対策

3. **アクセス制御**
   - CORS設定の厳格化
   - Rate Limiting
   - IP制限

### Phase 4: 監視システム（1週間）

#### Week 2: モニタリング基盤
1. **メトリクス可視化**
   - Prometheusセットアップ
   - Grafanaダッシュボード作成
   - カスタムメトリクス定義

2. **ログ管理**
   - 構造化ログ
   - ログローテーション
   - エラー追跡

3. **アラート設定**
   - 閾値設定
   - 通知チャネル設定
   - エスカレーションルール

### Phase 5: パフォーマンス最適化（2週間）

#### Week 3-4: 最適化実装
1. **キャッシング戦略**
   - Redis統合
   - クエリキャッシュ
   - 結果キャッシュ

2. **データベース最適化**
   - インデックス最適化
   - クエリ最適化
   - 接続プーリング

3. **スケーラビリティ**
   - 水平スケーリング対応
   - ロードバランシング
   - 非同期タスクキュー

## 🎯 即座に実装すべき項目

### 1. 環境変数による設定管理

```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Security
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False
    
    class Config:
        env_file = ".env"
```

### 2. 基本的な認証実装

```python
# app/auth/jwt_auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
```

### 3. Prometheusメトリクス統合

```python
# app/monitoring/prometheus.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# メトリクス定義
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
active_users = Gauge('active_users', 'Number of active users')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 4. Docker Compose プロダクション構成

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: ai-ft-rag:prod
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
```

## 📊 期待される成果

### 実装後の改善指標

| 項目 | 現在 | 目標 | 改善率 |
|------|------|------|--------|
| **セキュリティスコア** | 60% | 95% | +58% |
| **レスポンス時間** | 500ms | 200ms | 60%短縮 |
| **可用性** | 95% | 99.9% | +4.9% |
| **同時接続数** | 100 | 1000 | 10倍 |
| **エラー率** | 1% | 0.1% | 90%削減 |

## 🔨 実装開始用スクリプト

```bash
#!/bin/bash
# start_phase3.sh

echo "Starting Phase 3: Production Preparation"

# 1. Create directory structure
mkdir -p app/auth app/monitoring config/prod tests/security

# 2. Install security dependencies
pip install python-jose[cryptography] passlib[bcrypt] python-multipart
pip install prometheus-client redis hiredis

# 3. Generate secret key
python -c "import secrets; print(f'SECRET_KEY={secrets.token_urlsafe(32)}')" > .env

# 4. Setup monitoring
docker-compose -f docker-compose.monitoring.yml up -d

echo "Phase 3 setup complete!"
```

## 推奨事項

1. **まずセキュリティ対策を実装**（必須）
2. **次に監視システムを構築**（運用に必須）
3. **その後パフォーマンス最適化**（スケール時に必要）

これらの実装により、エンタープライズレベルの本番環境に対応可能なシステムになります。

どの項目から始めたいですか？
