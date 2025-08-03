#!/usr/bin/env python3
"""
WSL用の簡単なモデルダウンロードスクリプト
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def quick_download():
    """簡単なダウンロード"""
    print("=== WSL簡単ダウンロード ===")
    print()
    
    # プロジェクト内のmodelsディレクトリを設定
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    # ディレクトリを作成
    models_dir.mkdir(exist_ok=True)
    
    # 環境変数を設定
    os.environ['HF_HOME'] = str(models_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(models_dir)
    
    print(f"ダウンロード先: {models_dir}")
    print()
    
    # ダウンロードするモデル
    model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    
    print(f"ダウンロード開始: {model_name}")
    print("（時間がかかる場合があります）")
    print()
    
    try:
        # トークナイザーをダウンロード
        print("1. トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(models_dir)
        )
        print("✅ トークナイザー完了")
        
        # モデルをダウンロード
        print("2. モデルをダウンロード中...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=str(models_dir)
        )
        print("✅ モデル完了")
        
        print()
        print("🎉 ダウンロード完了！")
        print(f"保存場所: {models_dir}")
        
        # ファイルサイズを確認
        total_size = 0
        for file_path in models_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_gb = total_size / (1024**3)
        print(f"合計サイズ: {size_gb:.1f}GB")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("インターネット接続とディスク容量を確認してください")

if __name__ == "__main__":
    quick_download() 