#!/usr/bin/env python3
"""
GPT-NeoXモデルのみをOllamaに登録
"""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    model_path = Path("/workspace/models/gpt-neox-20b.Q4_K_M.gguf")
    model_name = "gpt-neox-20b-base:latest"
    
    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        return
    
    logger.info(f"GPT-NeoX 20Bベースモデルを登録中...")
    logger.info(f"パス: {model_path}")
    logger.info(f"サイズ: {model_path.stat().st_size / (1024**3):.2f} GB")
    
    # Modelfileを作成
    modelfile_content = f"""FROM {model_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM "You are a helpful AI assistant with expertise in road design and civil engineering."
"""
    
    # 一時ファイルに保存
    modelfile_path = Path("/tmp/Modelfile_gpt_neox")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    # Ollamaでモデルを作成
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    logger.info(f"実行: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"✅ モデル '{model_name}' を登録しました")
        logger.info(f"使用例: ollama run {model_name} 'What is road cross slope?'")
    else:
        logger.error(f"❌ 登録失敗: {result.stderr}")

if __name__ == "__main__":
    main()