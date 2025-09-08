#!/bin/bash

echo "================================"
echo "量子化ステータスファイルチェック"
echo "================================"

# /tmpディレクトリの量子化ステータスファイルを探す
echo "ステータスファイル一覧:"
ls -la /tmp/quantization_*.json 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "最新のステータスファイル内容:"
    
    # 最新のファイルを取得
    latest_file=$(ls -t /tmp/quantization_*.json 2>/dev/null | head -1)
    
    if [ -n "$latest_file" ]; then
        echo "ファイル: $latest_file"
        echo "内容:"
        cat "$latest_file" | python -m json.tool
    fi
else
    echo "ステータスファイルが見つかりません"
fi

echo ""
echo "================================"
echo "量子化出力ディレクトリ"
echo "================================"

# 量子化出力をチェック
output_dir="/workspace/outputs/ollama_conversion"

if [ -d "$output_dir" ]; then
    echo "$output_dir の内容:"
    ls -la "$output_dir"
else
    echo "$output_dir が存在しません"
fi

echo ""
echo "================================"
echo "プロセスチェック"
echo "================================"

# 実行中のPythonプロセスを確認
echo "量子化関連のプロセス:"
ps aux | grep -E "qlora_to_ollama|quantiz" | grep -v grep