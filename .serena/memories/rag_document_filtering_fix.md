# RAGシステム文書フィルタリング修正

## 問題の概要
RAGシステムで文書選択フィルタリングが機能しない問題を修正しました。

## 根本原因
1. **MetadataManager** (`src/rag/indexing/metadata_manager.py`)
   - 文書にUUID形式の`id`を割り当てる
   - 例: "248a7cae-cea7-4766-995f-924768931246"

2. **フロントエンド** (`templates/rag.html`)
   - 文書リストAPIから受け取った`doc.id`（UUID）を`document_ids`として送信

3. **ベクトルストア** (`src/rag/indexing/vector_store.py`)
   - 以前は`filename`でマッチングを試みていた（不一致）

## 修正内容

### 1. インデックス作成時の修正 (`scripts/rag/index_documents.py`)
```python
# doc_idをメタデータに追加
document_metadata_dict = doc_metadata.to_dict()
document_metadata_dict['doc_id'] = doc_metadata.id  # MetadataManagerのIDを追加
```

### 2. ベクトルストア検索の修正 (`src/rag/indexing/vector_store.py`)
```python
# doc_idでマッチング（MetadataManagerのID）
doc_id_condition = FieldCondition(
    key="doc_id",
    match=MatchValue(value=doc_id)
)
```

## データフロー
1. 文書アップロード → MetadataManagerがUUID生成
2. インデックス作成 → doc_id（UUID）をベクトルストアのメタデータに保存
3. 文書リスト取得 → MetadataManagerからid（UUID）を取得
4. 検索実行 → document_ids（UUID）でフィルタリング
5. ベクトルストア → doc_idフィールドでマッチング

## 重要な注意事項
- 既存の文書は再インデックスが必要（doc_idフィールドがないため）
- 新規アップロードされた文書は自動的に正しく動作する
- 後方互換性のため`original_id`でのマッチングも維持

## テスト方法
1. 新しい文書をアップロード
2. 「ハイブリッド検索・質問応答」タブで特定の文書を選択
3. 検索を実行し、選択した文書のみから結果が返されることを確認
4. 引用・参考資料が正しく表示されることを確認