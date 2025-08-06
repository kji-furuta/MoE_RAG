#!/bin/bash
# container_structure_inspector.sh
# コンテナ内部構造を詳細に調査するスクリプト

echo "=== AI_FT_3 コンテナ内部構造調査 ==="
echo "実行日時: $(date)"
echo ""

CONTAINER_NAME="ai-ft-container"
OUTPUT_DIR="docs/container_inspection"

# 出力ディレクトリの作成
mkdir -p "$OUTPUT_DIR"

# 1. コンテナが実行中か確認
echo "1. コンテナ状態の確認..."
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "エラー: コンテナ $CONTAINER_NAME が実行されていません。"
    echo ""
    echo "コンテナを起動するには:"
    echo "cd docker && docker-compose up -d"
    exit 1
fi

echo "✓ コンテナが実行中です"
echo ""

# 2. ワークスペースの基本構造
echo "2. /workspace の基本構造を取得中..."
docker exec $CONTAINER_NAME ls -la /workspace/ > "$OUTPUT_DIR/workspace_root.txt"
docker exec $CONTAINER_NAME find /workspace -maxdepth 2 -type d | sort > "$OUTPUT_DIR/workspace_structure.txt"

# 3. 重要ディレクトリの詳細
echo "3. 重要ディレクトリの内容を確認中..."
IMPORTANT_DIRS=(
    "/workspace/src"
    "/workspace/app"
    "/workspace/configs"
    "/workspace/scripts"
    "/workspace/data"
    "/workspace/outputs"
)

for dir in "${IMPORTANT_DIRS[@]}"; do
    dirname=$(basename "$dir")
    echo "   - $dir を調査中..."
    docker exec $CONTAINER_NAME ls -la $dir 2>/dev/null > "$OUTPUT_DIR/dir_${dirname}.txt" || echo "Directory not found" > "$OUTPUT_DIR/dir_${dirname}.txt"
done

# 4. 設定ファイルの存在確認
echo "4. 設定ファイルの存在を確認中..."
cat > "$OUTPUT_DIR/config_files_status.txt" << EOF
=== 設定ファイル存在確認 ===
実行日時: $(date)

EOF

CONFIG_FILES=(
    "/workspace/config/rag_config.yaml"
    "/workspace/configs/available_models.json"
    "/workspace/docker/.env"
    "/workspace/requirements.txt"
    "/workspace/requirements_rag.txt"
)

for file in "${CONFIG_FILES[@]}"; do
    if docker exec $CONTAINER_NAME test -f "$file"; then
        echo "✓ EXISTS: $file" >> "$OUTPUT_DIR/config_files_status.txt"
    else
        echo "✗ MISSING: $file" >> "$OUTPUT_DIR/config_files_status.txt"
    fi
done

# 5. Pythonパッケージ情報
echo "5. インストール済みPythonパッケージを確認中..."
docker exec $CONTAINER_NAME pip list > "$OUTPUT_DIR/python_packages.txt"
docker exec $CONTAINER_NAME pip list | grep -E "torch|transformers|accelerate|qdrant|fastapi" > "$OUTPUT_DIR/key_packages.txt"

# 6. 環境変数
echo "6. 環境変数を取得中..."
docker exec $CONTAINER_NAME env | grep -E "CUDA|PYTHON|HF_|WANDB|PATH" | sort > "$OUTPUT_DIR/environment_vars.txt"

# 7. ボリュームマウント情報
echo "7. ボリュームマウント情報を取得中..."
docker inspect $CONTAINER_NAME | grep -A 20 "Mounts" > "$OUTPUT_DIR/volume_mounts.txt"

# 8. 実行可能スクリプトの確認
echo "8. 実行可能スクリプトを確認中..."
docker exec $CONTAINER_NAME find /workspace -name "*.py" -o -name "*.sh" | grep -E "main|start|run|index" | head -20 > "$OUTPUT_DIR/executable_scripts.txt"

# 9. レポートの生成
echo "9. 調査レポートを生成中..."
cat > "$OUTPUT_DIR/inspection_report.md" << 'REPORT_EOF'
# コンテナ内部構造調査レポート

生成日時: $(date)

## 1. ワークスペース構造

$(cat "$OUTPUT_DIR/workspace_root.txt")

## 2. ディレクトリ構造（上位2階層）

$(head -30 "$OUTPUT_DIR/workspace_structure.txt")
...

## 3. 設定ファイル状態

$(cat "$OUTPUT_DIR/config_files_status.txt")

## 4. 主要Pythonパッケージ

$(cat "$OUTPUT_DIR/key_packages.txt")

## 5. 環境変数（主要なもの）

$(cat "$OUTPUT_DIR/environment_vars.txt")

## 6. 主要な実行ファイル

$(head -10 "$OUTPUT_DIR/executable_scripts.txt")

REPORT_EOF

echo ""
echo "=== 調査完了 ==="
echo "結果は $OUTPUT_DIR に保存されました："
ls -la "$OUTPUT_DIR/"
echo ""
echo "調査レポート: $OUTPUT_DIR/inspection_report.md"
