#!/usr/bin/env python
"""
AI_FT_7 システム診断スクリプト
現在の環境状態を詳細に確認
"""

import os
import sys
import torch
import psutil
import subprocess
from pathlib import Path
import json
from datetime import datetime

def check_system_info():
    """システム情報の確認"""
    print("=" * 70)
    print("🖥️ System Information")
    print("=" * 70)
    
    # CPU情報
    print("\n📊 CPU Information:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    # メモリ情報
    mem = psutil.virtual_memory()
    print(f"\n💾 Memory Information:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    
    # ディスク情報
    print(f"\n💿 Disk Information:")
    for partition in psutil.disk_partitions():
        if partition.mountpoint == '/' or '/workspace' in partition.mountpoint:
            usage = psutil.disk_usage(partition.mountpoint)
            print(f"  {partition.mountpoint}:")
            print(f"    Total: {usage.total / (1024**3):.1f} GB")
            print(f"    Free: {usage.free / (1024**3):.1f} GB")
            print(f"    Used: {usage.percent:.1f}%")

def check_gpu_info():
    """GPU情報の確認"""
    print("\n" + "=" * 70)
    print("🎮 GPU Information")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"\n✅ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # 現在のメモリ使用量
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"    Allocated: {allocated:.2f} GB")
                print(f"    Reserved: {reserved:.2f} GB")
    else:
        print("❌ CUDA is not available")

def check_python_packages():
    """主要パッケージのバージョン確認"""
    print("\n" + "=" * 70)
    print("📦 Package Versions")
    print("=" * 70)
    
    packages = [
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "deepspeed",
        "fastapi",
        "uvicorn",
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"  {package}: {version}")
        except ImportError:
            print(f"  {package}: ❌ Not installed")

def check_project_structure():
    """プロジェクト構造の確認"""
    print("\n" + "=" * 70)
    print("📁 Project Structure")
    print("=" * 70)
    
    base_path = Path("/home/kjifu/AI_FT_7")
    
    important_dirs = [
        "src/training",
        "src/rag",
        "app",
        "models",
        "outputs",
        "data",
        "configs",
        "docker",
    ]
    
    for dir_path in important_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            # ディレクトリサイズを計算
            size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            print(f"  ✅ {dir_path}: {size_gb:.2f} GB")
        else:
            print(f"  ❌ {dir_path}: Not found")

def check_model_files():
    """モデルファイルの確認"""
    print("\n" + "=" * 70)
    print("🤖 Model Files")
    print("=" * 70)
    
    models_dir = Path("/home/kjifu/AI_FT_7/models")
    outputs_dir = Path("/home/kjifu/AI_FT_7/outputs")
    
    print("\n📂 Models directory:")
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                print(f"  - {model_path.name}: {size / (1024**3):.2f} GB")
    else:
        print("  Directory not found")
    
    print("\n📂 Outputs directory:")
    if outputs_dir.exists():
        for output_path in outputs_dir.iterdir():
            if output_path.is_dir():
                size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
                print(f"  - {output_path.name}: {size / (1024**3):.2f} GB")
    else:
        print("  Directory not found")

def check_docker_status():
    """Docker状態の確認"""
    print("\n" + "=" * 70)
    print("🐳 Docker Status")
    print("=" * 70)
    
    try:
        # Dockerコンテナの確認
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is running")
            if "ai-ft-container" in result.stdout:
                print("  ✅ AI-FT container is running")
            else:
                print("  ⚠️ AI-FT container is not running")
        else:
            print("❌ Docker is not accessible")
    except FileNotFoundError:
        print("❌ Docker command not found")

def generate_report():
    """診断レポートの生成"""
    print("\n" + "=" * 70)
    print("📋 Diagnosis Summary")
    print("=" * 70)
    
    # ディスク容量チェック
    workspace_usage = psutil.disk_usage('/home/kjifu/AI_FT_7')
    free_gb = workspace_usage.free / (1024**3)
    
    print("\n🔍 Key Findings:")
    
    # ディスク容量の判定
    if free_gb < 20:
        print(f"  ❌ Critical: Low disk space ({free_gb:.1f} GB free)")
        print("     → Cannot perform full fine-tuning of 32B model")
        print("     → Recommend: Use LoRA/DoRA or clean up space")
    elif free_gb < 85:
        print(f"  ⚠️ Warning: Limited disk space ({free_gb:.1f} GB free)")
        print("     → Full fine-tuning may fail")
        print("     → Recommend: Use LoRA/DoRA or AWQ quantization")
    else:
        print(f"  ✅ Sufficient disk space ({free_gb:.1f} GB free)")
    
    # GPU状態の判定
    if torch.cuda.is_available():
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory 
                        for i in range(torch.cuda.device_count()))
        total_vram_gb = total_vram / (1024**3)
        
        if total_vram_gb >= 48:
            print(f"  ✅ Sufficient VRAM ({total_vram_gb:.1f} GB total)")
            print("     → Can run 32B model with optimization")
        else:
            print(f"  ⚠️ Limited VRAM ({total_vram_gb:.1f} GB total)")
            print("     → Recommend: Use quantization or smaller models")
    
    # 推奨事項
    print("\n💡 Recommendations:")
    print("  1. Implement DoRA for better accuracy (+3.7%)")
    print("  2. Use vLLM for faster inference (2.5-3x)")
    print("  3. Apply AWQ quantization to save memory (50%)")
    
    # レポートファイルの保存
    report = {
        "timestamp": datetime.now().isoformat(),
        "disk_free_gb": free_gb,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    report_path = Path("/home/kjifu/AI_FT_7/system_diagnosis.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Report saved to: {report_path}")

def main():
    """メイン実行"""
    print("\n" + "🔧 AI_FT_7 System Diagnosis Tool 🔧".center(70))
    print("=" * 70)
    
    check_system_info()
    check_gpu_info()
    check_python_packages()
    check_project_structure()
    check_model_files()
    check_docker_status()
    generate_report()
    
    print("\n" + "=" * 70)
    print("✅ Diagnosis completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
