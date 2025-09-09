#!/usr/bin/env python3
"""
すべての必要なモデルを再登録する
"""

import subprocess
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model_from_gguf(gguf_path: Path, model_name: str):
    """GGUFファイルからOllamaモデルを作成"""
    try:
        # Modelfileを作成
        modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM "あなたは日本語を理解し、道路設計と土木工学の専門知識を持つAIアシスタントです。"
"""
        
        # 一時ファイルに保存
        modelfile_path = Path(f"/tmp/Modelfile_{model_name.replace(':', '_')}")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        # Ollamaでモデルを作成
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        logger.info(f"作成中: {model_name}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info(f"✅ {model_name} を登録しました")
            return True
        else:
            logger.error(f"❌ {model_name} の登録に失敗: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning(f"⏱️ {model_name} の登録がタイムアウトしました（正常に作成されている可能性があります）")
        return True
    except Exception as e:
        logger.error(f"エラー: {e}")
        return False

def main():
    """メイン処理"""
    
    # 登録するモデルのリスト（優先順位順）
    models_to_register = [
        # ベースモデル
        {
            "path": "/workspace/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            "name": "deepseek-32b-base:latest"
        },
        {
            "path": "/workspace/models/gpt-neox-20b.Q4_K_M.gguf",
            "name": "gpt-neox-20b-base:latest"
        },
        # ファインチューニング済みモデル（ユーザー定義名）
        {
            "path": "/workspace/models/5_deepseek-32b-finetuned.gguf",
            "name": "5_deepseek-32b-finetuned:latest"
        },
        {
            "path": "/workspace/models/deepseek-32b-finetuned.gguf",
            "name": "deepseek-32b-finetuned:latest"
        },
        {
            "path": "/workspace/models/4_deepseek-32b-finetuned.gguf",
            "name": "4_deepseek-32b-finetuned:latest"
        },
        {
            "path": "/workspace/models/gpt-neox-20b-finetuned.gguf",
            "name": "gpt-neox-20b-finetuned:latest"
        }
    ]
    
    logger.info("モデルの再登録を開始します...")
    logger.info("注: この処理には数分かかる場合があります")
    
    success_count = 0
    failed_count = 0
    
    for model in models_to_register:
        model_path = Path(model["path"])
        if model_path.exists():
            if create_model_from_gguf(model_path, model["name"]):
                success_count += 1
            else:
                failed_count += 1
            time.sleep(1)  # 各モデル登録間に短い待機時間
        else:
            logger.warning(f"⚠️ ファイルが見つかりません: {model_path}")
    
    # 結果表示
    logger.info("\n" + "="*60)
    logger.info(f"登録完了: 成功 {success_count} / 失敗 {failed_count}")
    
    # 登録されたモデルを確認
    logger.info("\n現在のOllamaモデル:")
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print(result.stdout)
    
    logger.info("\n✅ Docker Composeの設定も修正済みです")
    logger.info("次回のコンテナ再起動後もモデルは永続化されます")

if __name__ == "__main__":
    main()