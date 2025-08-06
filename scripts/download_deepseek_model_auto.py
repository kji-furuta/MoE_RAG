#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-32B-Japaneseモデルを自動でダウンロードするスクリプト
※注意: 約70GBのダウンロードが必要です
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger
import torch

def download_deepseek_model():
    """DeepSeekモデルを完全にダウンロード"""
    model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    
    logger.info(f"モデルをダウンロード中: {model_name}")
    logger.info("⚠️ 注意: これは大きなモデル（約70GB）なので、ダウンロードに長時間かかります")
    
    # ディスク容量チェック
    import shutil
    free_space = shutil.disk_usage(".").free / (1024**3)  # GB
    logger.info(f"利用可能なディスク容量: {free_space:.1f} GB")
    
    if free_space < 100:
        logger.error("ディスク容量が不足しています。少なくとも100GBの空き容量が必要です。")
        return False
    
    try:
        # Hugging Faceトークンの確認
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token:
            logger.info("HF_TOKENが設定されています")
        
        # モデルをダウンロード
        cache_dir = Path.home() / ".cache" / "huggingface"
        logger.info(f"キャッシュディレクトリ: {cache_dir}")
        
        logger.info("ダウンロードを開始します...")
        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True  # 中断時の再開をサポート
        )
        
        logger.info(f"✅ モデルのダウンロードが完了しました: {downloaded_path}")
        
        # ダウンロードされたファイルの確認
        import glob
        safetensor_files = glob.glob(str(Path(downloaded_path) / "*.safetensors"))
        logger.info(f"ダウンロードされたモデルファイル数: {len(safetensor_files)}")
        
        # ファイルサイズの合計を計算
        total_size = sum(os.path.getsize(f) for f in safetensor_files) / (1024**3)
        logger.info(f"モデルの合計サイズ: {total_size:.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"モデルのダウンロードに失敗しました: {e}")
        logger.error("Hugging Faceのアクセストークンが必要な場合があります。")
        logger.error("環境変数 HF_TOKEN を設定してください。")
        return False

def main():
    """メイン処理"""
    logger.info("DeepSeek-R1-Distill-Qwen-32B-Japaneseモデルの自動ダウンロードを開始します...")
    logger.info("約70GBのダウンロードを行います。")
    
    success = download_deepseek_model()
    
    if success:
        logger.info("✅ モデルのダウンロードが完了しました")
        logger.info("次回からは起動時のダウンロードがスキップされます")
    else:
        logger.error("❌ モデルのダウンロードに失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()