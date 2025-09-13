#!/usr/bin/env python3
"""
ユーザーがUIで設定した名前でOllamaモデルを登録する
"""

import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_ollama_model(source_name: str, target_name: str):
    """既存のOllamaモデルを新しい名前でコピー"""
    try:
        # Modelfileを作成
        modelfile_content = f"FROM {source_name}"
        
        # 一時的なModelfileを作成
        modelfile_path = Path("/tmp/Modelfile_copy")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        # 新しい名前でモデルを作成
        cmd = ["ollama", "create", target_name, "-f", str(modelfile_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ モデル '{target_name}' を作成しました")
            return True
        else:
            logger.error(f"❌ モデル作成失敗: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"エラー: {e}")
        return False

def main():
    """メイン処理"""
    
    # ユーザーがUIで設定した名前（設定ファイルで参照されている名前）
    user_defined_names = [
        "5_deepseek-32b-finetuned:latest",
        "4_deepseek-32b-finetuned:latest",
        "deepseek-32b-finetuned:latest"
    ]
    
    # 現在存在するモデル
    source_model = "deepseek-32b-japanese:latest"
    
    logger.info("ユーザー定義の名前でモデルを登録します...")
    
    # 最初の名前（5_deepseek-32b-finetuned:latest）を優先的に使用
    target_name = user_defined_names[0]
    
    # 既存のモデルをユーザー定義の名前でコピー
    if copy_ollama_model(source_model, target_name):
        logger.info(f"✅ モデルを '{target_name}' として登録しました")
        
        # 他の名前でもエイリアスを作成（オプション）
        for alias in user_defined_names[1:]:
            if alias != target_name:
                copy_ollama_model(target_name, alias)
        
        # 登録されたモデルを確認
        logger.info("\n現在のOllamaモデル:")
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print(result.stdout)
        
        logger.info("\n✅ 完了！以下のコマンドでモデルをテストできます:")
        logger.info(f"  ollama run {target_name} '道路設計について教えてください'")
        
    else:
        logger.error("❌ モデルの登録に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()