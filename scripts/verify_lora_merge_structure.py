#!/usr/bin/env python3
"""
LoRA Merge Process Structure Verification Script
Verifies all components of the DeepSeek model LoRA merge workflow
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

def check_base_model():
    """ベースモデルの確認"""
    print("=" * 60)
    print("1. BASE MODEL CHECK")
    print("=" * 60)
    
    base_model = Path("models/gguf/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf")
    
    if base_model.exists():
        size_gb = base_model.stat().st_size / (1024**3)
        print(f"✅ Base model found: {base_model}")
        print(f"   Size: {size_gb:.2f} GB")
        print(f"   Last modified: {datetime.fromtimestamp(base_model.stat().st_mtime)}")
        return True
    else:
        print(f"❌ Base model missing: {base_model}")
        print("   Run: scripts/init_deepseek_model.sh to download")
        return False

def check_lora_adapters():
    """LoRAアダプターの確認"""
    print("\n" + "=" * 60)
    print("2. LORA ADAPTERS CHECK")
    print("=" * 60)
    
    outputs_dir = Path("outputs")
    lora_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("lora_")])
    
    if lora_dirs:
        print(f"✅ Found {len(lora_dirs)} LoRA adapter directories")
        print("\nLatest 5 adapters:")
        for lora_dir in lora_dirs[-5:]:
            adapter_file = lora_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                size_mb = adapter_file.stat().st_size / (1024**2)
                print(f"  ✅ {lora_dir.name}")
                print(f"     - adapter_model.safetensors: {size_mb:.2f} MB")
                
                # Check training info
                info_file = lora_dir / "training_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                        print(f"     - Base model: {info.get('base_model', 'Unknown')}")
            else:
                print(f"  ⚠️ {lora_dir.name} - missing adapter_model.safetensors")
        
        return True
    else:
        print("❌ No LoRA adapters found in outputs/")
        return False

def check_conversion_script():
    """変換スクリプトの確認"""
    print("\n" + "=" * 60)
    print("3. CONVERSION SCRIPT CHECK")
    print("=" * 60)
    
    script = Path("scripts/apply_lora_to_gguf_improved.py")
    
    if script.exists():
        print(f"✅ Conversion script found: {script}")
        
        # Check for CMake fix
        with open(script, 'r') as f:
            content = f.read()
            if 'DLLAMA_CURL=OFF' in content:
                print("   ✅ CMake CURL fix is applied")
            else:
                print("   ⚠️ CMake CURL fix is missing")
        
        # Check llama.cpp directory
        llama_cpp_dir = Path("llama.cpp")
        if llama_cpp_dir.exists():
            print(f"   ✅ llama.cpp directory exists")
            build_dir = llama_cpp_dir / "build"
            if build_dir.exists():
                print(f"   ✅ llama.cpp build directory exists")
            else:
                print(f"   ⚠️ llama.cpp not built yet - will be built on first run")
        else:
            print(f"   ⚠️ llama.cpp not cloned yet - will be cloned on first run")
        
        return True
    else:
        print(f"❌ Conversion script missing: {script}")
        return False

def check_ollama_integration():
    """Ollama統合の確認"""
    print("\n" + "=" * 60)
    print("4. OLLAMA INTEGRATION CHECK")
    print("=" * 60)
    
    # Check if Ollama is running
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print("✅ Ollama is running")
            if 'deepseek' in result.stdout.lower():
                print("   ✅ DeepSeek model registered in Ollama:")
                for line in result.stdout.split('\n'):
                    if 'deepseek' in line.lower():
                        print(f"      {line.strip()}")
            else:
                print("   ⚠️ DeepSeek model not registered in Ollama")
                print("      Run: ollama create deepseek-32b-japanese -f Modelfile")
        else:
            print("⚠️ Ollama command failed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️ Ollama not running or not installed")
        print("   Start with: ollama serve")
    
    # Check Ollama model directory
    ollama_models = Path.home() / ".ollama/models"
    if ollama_models.exists():
        print(f"   ✅ Ollama models directory exists: {ollama_models}")
    else:
        print(f"   ⚠️ Ollama models directory not found")

def show_merge_workflow():
    """マージワークフローの表示"""
    print("\n" + "=" * 60)
    print("5. COMPLETE MERGE WORKFLOW")
    print("=" * 60)
    
    print("""
STEP 1: Prepare LoRA Adapter
  └─ Train with: python src/training/lora_finetuning.py
  └─ Output: outputs/lora_YYYYMMDD_HHMMSS/adapter_model.safetensors

STEP 2: Merge LoRA with Base Model
  └─ Run: python scripts/apply_lora_to_gguf_improved.py \\
          --base-model models/gguf/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \\
          --lora-adapter outputs/lora_YYYYMMDD_HHMMSS \\
          --output-model outputs/merged_models/deepseek-32b-merged.gguf

STEP 3: Import to Ollama (Optional)
  └─ Create Modelfile:
      FROM outputs/merged_models/deepseek-32b-merged.gguf
      PARAMETER temperature 0.7
  └─ Run: ollama create deepseek-custom -f Modelfile

STEP 4: Use the Model
  └─ Via Ollama: ollama run deepseek-custom
  └─ Via API: POST http://localhost:11434/api/generate
  └─ Via Web UI: http://localhost:8050/
""")

def show_example_command():
    """実行例の表示"""
    print("\n" + "=" * 60)
    print("6. EXAMPLE MERGE COMMAND")
    print("=" * 60)
    
    # Find latest LoRA adapter
    outputs_dir = Path("outputs")
    lora_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("lora_")])
    
    if lora_dirs:
        latest_lora = lora_dirs[-1].name
        print(f"Using latest LoRA adapter: {latest_lora}")
        print("\nExample command:")
        print(f"""
python scripts/apply_lora_to_gguf_improved.py \\
    --base-model models/gguf/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \\
    --lora-adapter outputs/{latest_lora} \\
    --output-model outputs/merged_models/deepseek-32b-{latest_lora}.gguf \\
    --quantize Q4_K_M
""")
    else:
        print("No LoRA adapters found. Train a model first with:")
        print("python src/training/lora_finetuning.py")

def main():
    """メイン処理"""
    print("=" * 60)
    print("DeepSeek Model LoRA Merge Structure Verification")
    print("=" * 60)
    
    # Run all checks
    base_ok = check_base_model()
    lora_ok = check_lora_adapters()
    script_ok = check_conversion_script()
    check_ollama_integration()
    
    # Show workflow
    show_merge_workflow()
    show_example_command()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if base_ok and lora_ok and script_ok:
        print("✅ All components ready for LoRA merge!")
    else:
        print("⚠️ Some components need attention:")
        if not base_ok:
            print("  - Download base model with: scripts/init_deepseek_model.sh")
        if not lora_ok:
            print("  - Train a LoRA adapter with: python src/training/lora_finetuning.py")
        if not script_ok:
            print("  - Restore conversion script: git checkout scripts/apply_lora_to_gguf_improved.py")

if __name__ == "__main__":
    main()