# 開発ガイドライン

## 開発原則

### 1. Docker優先開発
- **必須**: すべての開発はDockerコンテナ内で実施
- **理由**: 環境差異の排除、GPU設定の統一
- **例外**: 簡単なコード編集のみローカル可

### 2. ポート管理
- **統合API**: 8050番ポート（main_unified.py）
- **他のポート使用禁止**: 競合を避けるため
- **WebSocket**: 同一ポートで `/ws/*` パス使用

### 3. エラーハンドリング
```python
# 良い例
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"具体的なエラー: {e}")
    return ErrorResponse(detail=str(e))
except Exception as e:
    logger.exception("予期しないエラー")
    raise HTTPException(status_code=500)

# 悪い例
try:
    result = risky_operation()
except:
    pass  # エラーを握りつぶさない
```

## コード追加・変更時の注意点

### 新機能追加時
1. 既存パターンの確認
   - 類似機能のコードを参照
   - 命名規則の統一
   - インターフェースの一貫性

2. 設定ファイルの更新
   - `config/*.yaml` に設定追加
   - 環境変数は `.env.example` に記載
   - デフォルト値の設定

3. テストの追加
   - 単体テストを `tests/` に追加
   - 統合テストの更新確認

### API変更時
1. 後方互換性の考慮
   - 既存エンドポイントの維持
   - バージョニング（必要な場合）
   - 非推奨警告の追加

2. ドキュメント更新
   - OpenAPI仕様の自動生成確認
   - README.mdの更新（重要変更時）

### モデル関連の変更
1. メモリ使用量の確認
   ```python
   # GPU メモリ監視
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

2. 量子化設定の確認
   - モデルサイズに応じた自動量子化
   - CPUオフロードの設定

## パフォーマンス最適化

### バッチ処理
```python
# 良い例：バッチ処理
def process_batch(items: List[Item], batch_size: int = 32):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield process_items(batch)

# 悪い例：逐次処理
for item in items:
    process_item(item)  # 非効率
```

### 非同期処理の活用
```python
# 並列実行
results = await asyncio.gather(
    async_task1(),
    async_task2(),
    async_task3()
)
```

### キャッシング
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param):
    # 重い計算処理
    return result
```

## セキュリティガイドライン

### 1. 認証情報の管理
- **禁止**: ハードコーディング
- **必須**: 環境変数使用
- **推奨**: シークレット管理ツール

### 2. 入力検証
```python
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('top_k must be between 1 and 100')
        return v
```

### 3. SQLインジェクション対策
```python
# 良い例：パラメータ化クエリ
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# 悪い例：文字列連結
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # 危険
```

## Git運用

### ブランチ戦略
- **main**: 本番環境
- **develop**: 開発環境
- **feature/***: 機能開発
- **fix/***: バグ修正
- **hotfix/***: 緊急修正

### コミットメッセージ
```
<type>: <subject>

<body>

<footer>
```

Types:
- feat: 新機能
- fix: バグ修正
- docs: ドキュメント
- style: フォーマット
- refactor: リファクタリング
- test: テスト
- chore: その他

例：
```
feat: MoEルーターに動的エキスパート選択を追加

- エントロピーベースの信頼度計算を実装
- top-kパラメータを動的に調整
- パフォーマンスが15%向上
```

## デバッグ手法

### ログレベルの使い分け
```python
logger.debug("詳細なデバッグ情報")
logger.info("通常の処理フロー")
logger.warning("警告（処理は継続）")
logger.error("エラー（リカバリ可能）")
logger.critical("致命的エラー")
```

### プロファイリング
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 計測対象のコード
expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10
```

## リソース管理

### コンテキストマネージャーの使用
```python
# 良い例
with open('file.txt', 'r') as f:
    content = f.read()

# GPUメモリの解放
with torch.cuda.device(0):
    # GPU処理
    pass
# 自動的にメモリ解放
```

### メモリリーク対策
```python
# 循環参照の回避
import weakref

class Node:
    def __init__(self, parent=None):
        self.parent = weakref.ref(parent) if parent else None
```

## 継続的改善

### コードレビューチェックリスト
- [ ] 命名規則の遵守
- [ ] 適切なエラーハンドリング
- [ ] テストの追加・更新
- [ ] ドキュメントの更新
- [ ] パフォーマンスへの影響確認
- [ ] セキュリティの考慮

### リファクタリング指針
1. **DRY原則**: 重複コードの排除
2. **SOLID原則**: 設計原則の遵守
3. **KISS原則**: シンプルさの維持
4. **YAGNI原則**: 必要になるまで実装しない