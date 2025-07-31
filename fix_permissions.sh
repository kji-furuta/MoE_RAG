#!/bin/bash
# Docker内で権限を修正するスクリプト

echo "Fixing permissions for outputs directory..."

# outputsディレクトリの権限を修正
if [ -d "/workspace/outputs" ]; then
    chmod -R 777 /workspace/outputs
    echo "Permissions fixed for existing outputs directory"
else
    mkdir -p /workspace/outputs
    chmod -R 777 /workspace/outputs
    echo "Created and set permissions for outputs directory"
fi

# 現在のユーザーとグループを表示
echo "Current user: $(whoami)"
echo "Current groups: $(groups)"

# outputsディレクトリの権限を確認
echo "Outputs directory permissions:"
ls -la /workspace/ | grep outputs

echo "Permission fix completed!"