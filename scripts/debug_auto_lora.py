#!/usr/bin/env python3
"""
auto_lora_to_ollama.pyのデバッグ版
詳細なログ出力とエラーハンドリング付き
"""

import torch
import gc
import os
import sys
import subprocess
import json
import time
import shutil
import traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ログファイル設定
LOG_FILE = "/workspace/logs/quantization_debug.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_message(msg):
    """コンソールとファイルに同時出力"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")
        f.flush()

def check_environment():
    """環境チェック"""
    log_message("=== 環境チェック開始 ===")
    
    # GPU状態
    if torch.cuda.is_available():
        log_message(f"GPU利用可能: {torch.cuda.device_count()}個")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log_message(f"  GPU {i}: {props.name}")
            log_message(f"    メモリ: {props.total_memory / 1024**3:.1f}GB")
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            log_message(f"    使用中: {allocated:.1f}GB")
    else:
        log_message("GPU利用不可")
    
    # メモリ状態
    import psutil
    mem = psutil.virtual_memory()
    log_message(f"システムメモリ: {mem.total / 1024**3:.1f}GB")
    log_message(f"  使用中: {mem.used / 1024**3:.1f}GB")
    log_message(f"  空き: {mem.available / 1024**3:.1f}GB")
    
    # ファイル存在チェック
    lora_path = "/workspace/outputs/lora_20250830_223432"
    if os.path.exists(lora_path):
        log_message(f"✅ LoRAパス存在: {lora_path}")
        files = os.listdir(lora_path)[:5]
        log_message(f"  ファイル例: {files}")
    else:
        log_message(f"❌ LoRAパスが存在しません: {lora_path}")
    
    # Pythonパス確認
    log_message(f"Python実行パス: {sys.executable}")
    log_message(f"Pythonバージョン: {sys.version}")
    
    return True

def test_import():
    """インポートテスト"""
    log_message("=== インポートテスト ===")
    
    try:
        # AI_FT_7のモジュールをインポート
        sys.path.insert(0, '/workspace')
        log_message("パス追加: /workspace")
        
        from app.memory_optimized_loader import (
            get_optimal_quantization_config,
            clear_gpu_memory,
            get_gpu_memory_info
        )
        log_message("✅ memory_optimized_loaderインポート成功")
        
        # テスト実行
        memory_info = get_gpu_memory_info()
        if memory_info:
            log_message(f"  GPUメモリ情報取得成功: {memory_info}")
        
        return True
        
    except Exception as e:
        log_message(f"❌ インポートエラー: {e}")
        log_message(traceback.format_exc())
        return False

def test_model_loading():
    """モデルロードのテスト（軽量）"""
    log_message("=== モデルロードテスト ===")
    
    try:
        from app.memory_optimized_loader import get_optimal_quantization_config
        
        base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
        
        # 量子化設定を取得
        quantization_config, device_map = get_optimal_quantization_config(
            base_model_name,
            available_memory_gb=40  # 仮の値
        )
        
        log_message(f"量子化設定取得成功")
        log_message(f"  device_map: {device_map}")
        
        # 実際のロードはスキップ（メモリ節約）
        log_message("モデルロードはスキップ（テストのため）")
        
        return True
        
    except Exception as e:
        log_message(f"❌ モデルロードテストエラー: {e}")
        log_message(traceback.format_exc())
        return False

def main():
    """メイン処理"""
    log_message("="*60)
    log_message("量子化デバッグスクリプト開始")
    log_message("="*60)
    
    try:
        # 環境チェック
        if not check_environment():
            log_message("環境チェック失敗")
            return 1
        
        # インポートテスト
        if not test_import():
            log_message("インポートテスト失敗")
            return 1
        
        # モデルロードテスト
        if not test_model_loading():
            log_message("モデルロードテスト失敗")
            return 1
        
        log_message("="*60)
        log_message("✅ すべてのテスト成功")
        log_message("="*60)
        
        log_message("\n問題が見つからない場合、以下を確認してください：")
        log_message("1. Dockerコンテナのメモリ制限")
        log_message("2. GPUドライバーの状態")
        log_message("3. ディスク容量")
        
        return 0
        
    except Exception as e:
        log_message(f"❌ 予期しないエラー: {e}")
        log_message(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())