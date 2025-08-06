# コードスタイルとコンベンション

## Python コードスタイル
- **フォーマッター**: Black (line-length=88)
- **インポート整理**: isort (profile=black)
- **型ヒント**: Optional使用、BaseModelでPydanticモデル定義
- **ドキュメント**: 日本語コメント許可、クラス・関数にdocstring

## 命名規則
- **クラス**: PascalCase (例: LoRAFinetuningTrainer)
- **関数/メソッド**: snake_case (例: prepare_model)
- **定数**: UPPER_SNAKE_CASE (例: MAX_LENGTH)
- **プライベート**: アンダースコアプレフィックス (例: _internal_method)

## プロジェクト構造
```
src/
  training/    # ファインチューニング関連
  rag/         # RAGシステム
  utils/       # ユーティリティ
  data/        # データ処理
  evaluation/  # 評価メトリクス
app/           # Webインターフェース
scripts/       # ユーティリティスクリプト
docker/        # Docker設定
configs/       # 設定ファイル
```

## 設計パターン
- **Config管理**: dataclassまたはPydantic BaseModel
- **非同期処理**: FastAPI async/await
- **エラーハンドリング**: try-except with logging
- **GPU最適化**: gradient_checkpointing, mixed_precision