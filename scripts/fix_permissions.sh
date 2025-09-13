#!/bin/bash

# データベースとストレージの権限を修正するスクリプト
# root権限で実行することを想定

echo "=" 
echo "データベース権限修正スクリプト"
echo "="

# メタデータディレクトリ
echo "1. メタデータディレクトリの権限修正..."
mkdir -p /workspace/metadata
chown -R root:root /workspace/metadata
chmod -R 777 /workspace/metadata
echo "✅ /workspace/metadata: 権限設定完了"

# Qdrantデータディレクトリ  
echo "2. Qdrantデータディレクトリの権限修正..."
mkdir -p /workspace/qdrant_data
chown -R root:root /workspace/qdrant_data
chmod -R 777 /workspace/qdrant_data
echo "✅ /workspace/qdrant_data: 権限設定完了"

# アウトプットディレクトリ
echo "3. アウトプットディレクトリの権限修正..."
mkdir -p /workspace/outputs/rag_index
chown -R root:root /workspace/outputs
chmod -R 777 /workspace/outputs
echo "✅ /workspace/outputs: 権限設定完了"

# ドキュメントディレクトリ
echo "4. ドキュメントディレクトリの権限修正..."
mkdir -p /workspace/data/documents
chown -R root:root /workspace/data/documents
chmod -R 777 /workspace/data/documents
echo "✅ /workspace/data/documents: 権限設定完了"

# 一時アップロードディレクトリ
echo "5. 一時アップロードディレクトリの権限修正..."
mkdir -p /workspace/temp_uploads
chown -R root:root /workspace/temp_uploads
chmod -R 777 /workspace/temp_uploads
echo "✅ /workspace/temp_uploads: 権限設定完了"

# ログディレクトリ
echo "6. ログディレクトリの権限修正..."
mkdir -p /workspace/logs
chown -R root:root /workspace/logs
chmod -R 777 /workspace/logs
echo "✅ /workspace/logs: 権限設定完了"

echo ""
echo "="
echo "✅ すべての権限設定が完了しました"
echo "="