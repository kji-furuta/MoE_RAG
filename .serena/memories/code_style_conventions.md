# コードスタイルと規約

## フォーマッティング
- **Black**: line-length=88、target-version=py38
- **isort**: profile=black、line_length=88
- **インデント**: スペース4つ

## 命名規則
- **クラス名**: PascalCase (例: `MoERouter`, `RAGQueryEngine`)
- **関数名**: snake_case (例: `process_query`, `train_model`)
- **定数**: UPPER_SNAKE_CASE (例: `MAX_TOKENS`, `DEFAULT_BATCH_SIZE`)
- **プライベート**: アンダースコア接頭辞 (例: `_internal_method`)

## 型ヒント
```python
from typing import Optional, Dict, Any, List, Union

def process_query(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True
) -> Dict[str, Any]:
    pass
```

## Docstring規約
```python
class MoEModel:
    """MoE (Mixture of Experts) モデルクラス
    
    専門分野別のエキスパートモデルを管理し、
    クエリに応じて適切なエキスパートを選択する。
    """
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """順伝播処理
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        pass
```

## ファイル構造
- **1ファイル1クラス/機能**: 単一責任の原則
- **__init__.py**: パッケージのエクスポート管理
- **設定ファイル**: YAML形式で`config/`ディレクトリに配置

## エラーハンドリング
```python
try:
    result = process_complex_task()
except SpecificError as e:
    logger.error(f"特定のエラーが発生: {e}")
    raise
except Exception as e:
    logger.exception(f"予期しないエラー: {e}")
    return default_value
```

## ロギング
```python
import logging
logger = logging.getLogger(__name__)

logger.info("処理開始")
logger.debug(f"詳細情報: {details}")
logger.warning("警告メッセージ")
logger.error("エラーメッセージ")
```

## 非同期処理
```python
async def async_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """非同期処理の例"""
    async with aiohttp.ClientSession() as session:
        response = await session.get(url)
        return await response.json()
```

## テスト規約
- **テストファイル**: `test_*.py` または `*_test.py`
- **テスト関数**: `test_` プレフィックス
- **フィクスチャ**: pytestフィクスチャを活用
- **モック**: `unittest.mock` または `pytest-mock`

## コメント規約
- **インラインコメント**: 必要最小限、複雑なロジックのみ
- **TODOコメント**: `# TODO(username): 説明`
- **FIXMEコメント**: `# FIXME: 問題の説明`
- **日本語コメント**: 技術用語や説明で使用可能