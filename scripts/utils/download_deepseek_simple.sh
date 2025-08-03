#!/bin/bash
# Simple script to download DeepSeek model using huggingface-cli

echo "=== DeepSeek Model Downloader ==="
echo "This will download the cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese model"
echo "Size: approximately 65GB"
echo ""

# Activate virtual environment
source venv/bin/activate

# Set cache directory
export HF_HOME="/home/kjifu/AI_finet/hf_cache"
export HF_HUB_CACHE="/home/kjifu/AI_finet/hf_cache"

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"

# Model details
MODEL_ID="cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"

echo "Downloading to: $HF_HOME"
echo ""

# Download using huggingface-cli
echo "Starting download..."
huggingface-cli download "$MODEL_ID" \
    --cache-dir "$HF_HOME" \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "Download complete!"
echo ""
echo "To verify the download, run:"
echo "source venv/bin/activate && python test_deepseek_fixed.py"