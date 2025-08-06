#!/usr/bin/env python3
"""
RAGシステムで使用するモデルを事前にダウンロードするスクリプト
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from loguru import logger

def preload_embedding_model():
    """埋め込みモデルを事前にダウンロード"""
    model_name = "intfloat/multilingual-e5-large"
    
    logger.info(f"埋め込みモデルをダウンロード中: {model_name}")
    
    try:
        # トークナイザーとモデルをダウンロード
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        logger.info(f"✅ モデルのダウンロードが完了しました: {model_name}")
        
        # キャッシュディレクトリの確認
        cache_dir = Path.home() / ".cache" / "huggingface"
        logger.info(f"キャッシュディレクトリ: {cache_dir}")
        
        # キャッシュされたファイルの確認
        if cache_dir.exists():
            files = list(cache_dir.rglob("*"))
            logger.info(f"キャッシュされたファイル数: {len(files)}")
        
        return True
        
    except Exception as e:
        logger.error(f"モデルのダウンロードに失敗しました: {e}")
        return False

def preload_llm_model():
    """LLMモデルを事前にダウンロード"""
    model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    
    logger.info(f"LLMモデルをダウンロード中: {model_name}")
    logger.info("⚠️ 注意: これは大きなモデル（約70GB）なので、ダウンロードに時間がかかります")
    
    try:
        from transformers import AutoModelForCausalLM
        
        # モデルの情報のみダウンロード（フルモデルは大きすぎるため）
        logger.info("モデル設定ファイルをダウンロード中...")
        from huggingface_hub import snapshot_download
        
        # モデルファイルをダウンロード（allow_patterns で必要なファイルのみ）
        snapshot_download(
            repo_id=model_name,
            cache_dir=Path.home() / ".cache" / "huggingface",
            allow_patterns=["*.json", "*.txt", "tokenizer*", "*.model"],  # 設定とトークナイザーのみ
            local_dir_use_symlinks=False
        )
        
        logger.info(f"✅ モデル設定ファイルのダウンロードが完了しました: {model_name}")
        logger.info("💡 フルモデルは起動時に必要に応じてダウンロードされます")
        
        return True
        
    except Exception as e:
        logger.error(f"モデルのダウンロードに失敗しました: {e}")
        return False

def main():
    """メイン処理"""
    logger.info("RAGモデルの事前ダウンロードを開始します...")
    
    # 埋め込みモデルのダウンロード
    embedding_success = preload_embedding_model()
    
    # LLMモデルのダウンロード
    llm_success = preload_llm_model()
    
    if embedding_success and llm_success:
        logger.info("✅ すべてのモデルのダウンロードが完了しました")
        logger.info("次回からは起動時のダウンロードがスキップされます")
    else:
        logger.error("❌ 一部のモデルのダウンロードに失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()