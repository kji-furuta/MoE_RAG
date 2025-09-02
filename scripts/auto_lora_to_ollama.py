#!/usr/bin/env python3
"""
LoRAマージからOllama登録までの完全自動化スクリプト
デュアルGPU環境で32Bモデルを効率的に処理
"""

import torch
import gc
import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def run_command(cmd, description=""):
    """コマンドを実行して結果を表示"""
    if description:
        print(f"\n{description}")
    print(f"実行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ エラー: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def clear_memory():
    """GPUメモリをクリア"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def step1_merge_lora():
    """Step 1: LoRAをマージ（AI_FT_7の量子化システムを使用）"""
    print("Loading base model", flush=True)  # UIの進捗表示用
    print("\n" + "="*60, flush=True)
    print("Step 1: LoRAマージ（AI_FT_7統合）", flush=True)
    print("="*60, flush=True)
    
    try:
        # AI_FT_7のメモリ最適化ローダーをインポート
        print("モジュールインポート中...", flush=True)
        sys.path.insert(0, '/workspace')
        from app.memory_optimized_loader import (
            load_finetuned_model_with_optimization,
            get_optimal_quantization_config,
            clear_gpu_memory,
            get_gpu_memory_info
        )
        print("✅ モジュールインポート成功", flush=True)
    except Exception as e:
        print(f"❌ モジュールインポートエラー: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        return False
    
    # パス設定
    base_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
    lora_path = "/workspace/outputs/lora_20250830_223432"
    output_path = "/workspace/outputs/merged_model"
    
    # 既存のマージモデルがある場合はスキップ
    if os.path.exists(f"{output_path}/model.safetensors.index.json"):
        print("✅ マージ済みモデルが既に存在します。スキップします。")
        return True
    
    # GPU情報表示
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"\nGPUメモリ状況:")
        print(f"  合計: {memory_info['total_gb']:.1f}GB")
        print(f"  空き: {memory_info['free_gb']:.1f}GB")
    
    clear_gpu_memory()
    
    # AI_FT_7の最適化設定を使用
    quantization_config, device_map = get_optimal_quantization_config(
        base_model_name,
        available_memory_gb=memory_info['free_gb'] if memory_info else None
    )
    
    print("\n1. ベースモデルをロード中（AI_FT_7最適化）...", flush=True)
    print(f"   モデル: {base_model_name}", flush=True)
    print(f"   デバイスマップ: {device_map}", flush=True)
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/offload"
        )
        print("✅ ベースモデルロード成功", flush=True)
    except Exception as e:
        print(f"❌ ベースモデルロードエラー: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        return False
    
    # メモリ使用状況
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"GPU使用中: {memory_info['allocated_gb']:.1f}GB")
    
    print("Loading LoRA adapter")  # UI進捗用
    print("\n2. LoRAアダプタをロード中...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("Merging")  # UI進捗用
    print("\n3. LoRAをマージ中...")
    model = model.merge_and_unload()
    
    print("Saving")  # UI進捗用
    print("\n4. マージモデルを保存中...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    # トークナイザも保存
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    # クリーンアップ
    del model, base_model
    clear_gpu_memory()
    
    print("✅ Step 1 完了: マージモデル保存済み")
    return True

def step2_convert_to_gguf():
    """Step 2: GGUF形式に変換"""
    print("Converting to GGUF")  # UI進捗用
    print("\n" + "="*60)
    print("Step 2: GGUF変換")
    print("="*60)
    
    merged_path = "/workspace/outputs/merged_model"
    gguf_path = "/workspace/outputs/model-f16.gguf"
    
    # 既存のGGUFファイルがある場合はスキップ
    if os.path.exists(gguf_path):
        print("✅ GGUFファイルが既に存在します。スキップします。")
        return True
    
    # llama.cppがなければクローン
    if not os.path.exists("/workspace/llama.cpp"):
        print("\n1. llama.cppをクローン中...")
        run_command("cd /workspace && git clone https://github.com/ggerganov/llama.cpp")
        run_command("cd /workspace/llama.cpp && make clean && make LLAMA_CUDA=1 -j8")
    
    print("\n2. GGUF形式に変換中...")
    print("   （この処理には10-15分かかります）")
    
    cmd = f"""cd /workspace/llama.cpp && python convert_hf_to_gguf.py \
        {merged_path} \
        --outfile {gguf_path} \
        --outtype f16"""
    
    if not run_command(cmd):
        print("❌ GGUF変換に失敗しました")
        return False
    
    # ファイルサイズ確認
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"✅ Step 2 完了: GGUF変換済み ({size_gb:.2f} GB)")
        return True
    
    return False

def step3_quantize():
    """Step 3: 量子化（Q4_K_M）"""
    print("Quantizing")  # UI進捗用
    print("\n" + "="*60)
    print("Step 3: 量子化")
    print("="*60)
    
    gguf_f16_path = "/workspace/outputs/model-f16.gguf"
    gguf_q4_path = "/workspace/outputs/deepseek-finetuned-q4_k_m.gguf"
    
    # 既存の量子化ファイルがある場合はスキップ
    if os.path.exists(gguf_q4_path):
        print("✅ 量子化ファイルが既に存在します。スキップします。")
        return True
    
    print("\n量子化中（Q4_K_M）...")
    print("   （この処理には10-15分かかります）")
    
    cmd = f"/workspace/llama.cpp/quantize {gguf_f16_path} {gguf_q4_path} Q4_K_M"
    
    if not run_command(cmd):
        print("❌ 量子化に失敗しました")
        return False
    
    # ファイルサイズ確認
    if os.path.exists(gguf_q4_path):
        size_gb = os.path.getsize(gguf_q4_path) / (1024**3)
        print(f"✅ Step 3 完了: 量子化済み ({size_gb:.2f} GB)")
        
        # 元のf16ファイルを削除（容量節約）
        if os.path.exists(gguf_f16_path):
            os.remove(gguf_f16_path)
            print("   中間ファイル(f16)を削除しました")
        
        return True
    
    return False

def step4_create_modelfile():
    """Step 4: Modelfile作成"""
    print("\n" + "="*60)
    print("Step 4: Modelfile作成")
    print("="*60)
    
    modelfile_path = "/workspace/outputs/Modelfile"
    
    modelfile_content = '''FROM ./deepseek-finetuned-q4_k_m.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"

SYSTEM "あなたは日本の道路設計の専門家です。道路構造令と設計基準に基づいて正確な技術的回答を提供してください。"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}"""
'''
    
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"✅ Step 4 完了: Modelfile作成済み")
    return True

def step5_register_ollama():
    """Step 5: Ollamaに登録"""
    print("\n" + "="*60)
    print("Step 5: Ollamaモデル登録")
    print("="*60)
    
    # Ollamaが起動しているか確認
    print("\n1. Ollamaサービス確認中...")
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("⚠️ Ollamaが起動していません。起動します...")
        subprocess.Popen("ollama serve", shell=True)
        time.sleep(5)  # 起動待ち
    
    # 既存のモデルを削除（あれば）
    print("\n2. 既存モデルを確認中...")
    run_command("ollama rm deepseek-finetuned 2>/dev/null || true")
    
    # モデルを登録
    print("\n3. 新しいモデルを登録中...")
    cmd = "cd /workspace/outputs && ollama create deepseek-finetuned -f Modelfile"
    
    if not run_command(cmd):
        print("❌ Ollama登録に失敗しました")
        return False
    
    print("✅ Step 5 完了: Ollamaモデル登録済み")
    return True

def step6_update_rag_config():
    """Step 6: RAG設定を更新（AI_FT_7統合）"""
    print("\n" + "="*60)
    print("Step 6: AI_FT_7 RAG設定更新")
    print("="*60)
    
    config_path = "/workspace/src/rag/config/rag_config.yaml"
    
    # 設定ファイルをバックアップ
    from datetime import datetime
    backup_path = f"{config_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(config_path, backup_path)
    print(f"設定ファイルをバックアップ: {backup_path}")
    
    # YAMLファイルを読み込み
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # LLM設定を更新
    if 'llm' not in config:
        config['llm'] = {}
    
    config['llm'].update({
        'use_ollama_fallback': True,
        'ollama_model': 'deepseek-finetuned',
        'ollama_host': 'http://localhost:11434',
        'quantization': {
            'enabled': True,
            'method': 'q4_k_m',
            'compute_type': 'float16'
        }
    })
    
    # システム設定も更新
    if 'system' not in config:
        config['system'] = {}
    
    config['system'].update({
        'memory_optimization': True,
        'gpu_memory_fraction': 0.95,
        'use_dual_gpu': True
    })
    
    # YAMLファイルに書き戻し
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("✅ Step 6 完了: AI_FT_7 RAG設定更新済み")
    print(f"  - Ollamaモデル: deepseek-finetuned")
    print(f"  - 量子化: Q4_K_M")
    print(f"  - メモリ最適化: 有効")
    return True

def step7_test_model():
    """Step 7: モデルテスト"""
    print("\n" + "="*60)
    print("Step 7: 動作確認")
    print("="*60)
    
    print("\n1. Ollamaで直接テスト...")
    test_cmd = 'echo "設計速度100km/hの最小曲線半径は？" | ollama run deepseek-finetuned'
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("応答:")
        print(result.stdout[:500])  # 最初の500文字
    
    print("\n2. RAGシステムでテスト...")
    test_cmd = """curl -X POST "http://localhost:8050/rag/query" \
        -H "Content-Type: application/json" \
        -d '{"query": "設計速度100km/hの最小曲線半径は？", "top_k": 3}'"""
    
    result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        try:
            response = json.loads(result.stdout)
            print("RAG応答:")
            print(json.dumps(response, ensure_ascii=False, indent=2)[:500])
        except:
            print("RAGレスポンス:", result.stdout[:500])
    
    print("\n✅ Step 7 完了: テスト完了")
    return True

def main():
    """メイン処理"""
    print("="*60)
    print("LoRA → Ollama 完全自動化パイプライン")
    print("="*60)
    print("\n処理ステップ:")
    print("1. LoRAマージ")
    print("2. GGUF変換")
    print("3. 量子化 (Q4_K_M)")
    print("4. Modelfile作成")
    print("5. Ollama登録")
    print("6. RAG設定更新")
    print("7. 動作確認")
    
    start_time = time.time()
    
    # 各ステップを実行
    steps = [
        ("Step 1: LoRAマージ", step1_merge_lora),
        ("Step 2: GGUF変換", step2_convert_to_gguf),
        ("Step 3: 量子化", step3_quantize),
        ("Step 4: Modelfile作成", step4_create_modelfile),
        ("Step 5: Ollama登録", step5_register_ollama),
        ("Step 6: RAG設定更新", step6_update_rag_config),
        ("Step 7: 動作確認", step7_test_model)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"⚠️ {step_name} で問題が発生しました")
        except Exception as e:
            print(f"❌ {step_name} でエラー: {e}")
            failed_steps.append(step_name)
    
    # 結果サマリー
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("処理完了")
    print("="*60)
    print(f"処理時間: {elapsed_time/60:.1f}分")
    
    if failed_steps:
        print(f"\n⚠️ 以下のステップで問題がありました:")
        for step in failed_steps:
            print(f"  - {step}")
    else:
        print("\n✅ すべてのステップが正常に完了しました！")
        print("\n使用方法:")
        print("1. Ollama経由: ollama run deepseek-finetuned")
        print("2. RAGシステム: http://localhost:8050/rag")
        print("\n生成されたファイル:")
        print("  - /workspace/outputs/merged_model/ (マージ済みモデル)")
        print("  - /workspace/outputs/deepseek-finetuned-q4_k_m.gguf (量子化モデル)")
        print("  - Ollamaモデル: deepseek-finetuned")

if __name__ == "__main__":
    main()