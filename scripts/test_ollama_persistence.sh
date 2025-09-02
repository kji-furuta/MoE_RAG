#!/bin/bash

# Ollama モデル永続化テストスクリプト

echo "========================================="
echo "Testing Ollama Model Persistence"
echo "========================================="

# テスト前の状態確認
echo "1. Checking current Ollama models..."
docker exec ai-ft-container ollama list

# Ollamaサービスを再起動してモデルが維持されるか確認
echo -e "\n2. Restarting Ollama service..."
docker exec ai-ft-container pkill ollama
sleep 3
docker exec ai-ft-container bash -c "nohup ollama serve > /dev/null 2>&1 &"
sleep 5

# 初期化スクリプトを実行
echo -e "\n3. Running initialization script..."
docker exec ai-ft-container /workspace/scripts/init_ollama_models.sh

# 最終確認
echo -e "\n4. Final model list:"
docker exec ai-ft-container ollama list

# deepseek-32b-finetunedモデルが存在するか確認
if docker exec ai-ft-container ollama list | grep -q "deepseek-32b-finetuned"; then
    echo -e "\n✅ SUCCESS: deepseek-32b-finetuned model is registered!"
else
    echo -e "\n❌ FAILED: deepseek-32b-finetuned model is not found!"
    echo "Manual registration may be required."
fi

echo "========================================="
echo "Test Complete"
echo "=========================================">