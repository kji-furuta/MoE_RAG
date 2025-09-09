#!/usr/bin/env python3
"""
ダウンロード済みのベースモデルをOllamaに登録する
"""

import subprocess
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_modelfile(model_path: Path, model_name: str) -> str:
    """Modelfileを作成"""
    modelfile_content = f"""FROM {model_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt for Japanese language understanding
SYSTEM "あなたは日本語を理解し、道路設計と土木工学の専門知識を持つAIアシスタントです。"
"""
    return modelfile_content

def register_model_to_ollama(model_path: Path, model_name: str):
    """モデルをOllamaに登録"""
    try:
        # Modelfileを作成
        modelfile_content = create_modelfile(model_path, model_name)
        
        # 一時的なModelfileを保存
        modelfile_path = Path(f"/tmp/Modelfile_{model_name.replace(':', '_')}")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile作成: {modelfile_path}")
        
        # Ollamaでモデルを作成
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ モデル '{model_name}' を正常に登録しました")
            return True
        else:
            logger.error(f"❌ モデル登録失敗: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"エラー: {e}")
        return False

def main():
    """メイン処理"""
    
    # 登録するベースモデル
    models_to_register = [
        {
            "path": "/workspace/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            "name": "deepseek-32b-base:latest",
            "description": "DeepSeek-R1-Distill-Qwen 32B Base Model (Q4_K_M)"
        },
        {
            "path": "/workspace/models/gpt-neox-20b.Q4_K_M.gguf", 
            "name": "gpt-neox-20b-base:latest",
            "description": "GPT-NeoX 20B Base Model (Q4_K_M)"
        }
    ]
    
    # 既存のファインチューニング済みモデルも正しく登録
    finetuned_models = [
        {
            "path": "/workspace/models/5_deepseek-32b-finetuned.gguf",
            "name": "5_deepseek-32b-finetuned:latest",
            "description": "DeepSeek 32B Fine-tuned Model (User defined name)"
        },
        {
            "path": "/workspace/models/deepseek-32b-finetuned.gguf",
            "name": "deepseek-32b-finetuned:latest",
            "description": "DeepSeek 32B Fine-tuned Model"
        },
        {
            "path": "/workspace/models/gpt-neox-20b-finetuned.gguf",
            "name": "gpt-neox-20b-finetuned:latest",
            "description": "GPT-NeoX 20B Fine-tuned Model"
        }
    ]
    
    logger.info("ベースモデルのOllama登録を開始します...")
    
    success_count = 0
    failed_count = 0
    
    # ベースモデルを登録
    for model in models_to_register:
        model_path = Path(model["path"])
        if model_path.exists():
            logger.info(f"\n登録中: {model['description']}")
            logger.info(f"パス: {model_path}")
            logger.info(f"サイズ: {model_path.stat().st_size / (1024**3):.2f} GB")
            
            if register_model_to_ollama(model_path, model["name"]):
                success_count += 1
            else:
                failed_count += 1
        else:
            logger.warning(f"⚠️ モデルファイルが見つかりません: {model_path}")
            failed_count += 1
    
    # ファインチューニング済みモデルも正しく登録
    logger.info("\n\nファインチューニング済みモデルの修正登録...")
    for model in finetuned_models:
        model_path = Path(model["path"])
        if model_path.exists():
            logger.info(f"\n登録中: {model['description']}")
            if register_model_to_ollama(model_path, model["name"]):
                success_count += 1
            else:
                failed_count += 1
    
    # 結果表示
    logger.info("\n" + "="*60)
    logger.info("登録結果:")
    logger.info(f"✅ 成功: {success_count} モデル")
    logger.info(f"❌ 失敗: {failed_count} モデル")
    
    # 登録されたモデルを確認
    logger.info("\n現在のOllamaモデル一覧:")
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print(result.stdout)
    
    if success_count > 0:
        logger.info("\n✅ 使用例:")
        logger.info("  ollama run deepseek-32b-base:latest '道路設計について教えてください'")
        logger.info("  ollama run gpt-neox-20b-base:latest '横断勾配の基準は？'")
        logger.info("  ollama run 5_deepseek-32b-finetuned:latest '道路の設計速度について'")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()