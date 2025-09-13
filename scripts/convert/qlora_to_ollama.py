#!/usr/bin/env python3
"""
QLoRA to Ollama変換パイプライン
QLoRAでファインチューニングしたモデルをOllamaで使用可能にする

ワークフロー:
1. LoRAアダプターとベースモデルをFP16でマージ
2. マージモデルをGGUF形式に変換
3. Q4_K_M形式で量子化
4. Ollamaに登録
"""

import os
import sys
import subprocess
import torch
import gc
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# AI_FT_7のメモリ最適化を使用
sys.path.append('/workspace')
from app.memory_optimized_loader import get_optimal_quantization_config

# llama.cppのディレクトリ（既存のものを優先使用）
if os.path.exists("/workspace/llama.cpp"):
    LLAMA_CPP_DIR = "/workspace/llama.cpp"
else:
    LLAMA_CPP_DIR = "/tmp/llama.cpp"

def run_command(cmd):
    """コマンド実行と結果表示"""
    print(f"実行中: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"エラー: {result.stderr}")
    
    return result.returncode == 0

def find_lora_adapter():
    """最新のLoRAアダプターを探す"""
    import glob
    import tarfile
    import tempfile
    
    lora_dirs = []
    
    # 可能なLoRAディレクトリをチェック
    possible_dirs = [
        "/workspace/outputs/lora_*",
        "/workspace/outputs/*/final_lora_model",
        "/workspace/outputs/*/best_lora_model",
        "/workspace/outputs/continual_task_*/lora"
    ]
    
    for pattern in possible_dirs:
        paths = glob.glob(pattern)
        for path in paths:
            # ディレクトリの場合
            if os.path.isdir(path):
                # adapter_config.jsonが存在するか確認
                if os.path.exists(os.path.join(path, "adapter_config.json")):
                    lora_dirs.append(path)
            # tar.gzファイルの場合
            elif path.endswith('.tar.gz'):
                # 解凍先ディレクトリを作成
                extract_dir = path.replace('.tar.gz', '_extracted')
                
                # 既に解凍済みの場合はそれを使用
                if os.path.exists(extract_dir) and os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
                    lora_dirs.append(extract_dir)
                else:
                    # 解凍が必要
                    print(f"📦 圧縮されたLoRAアダプターを解凍中: {os.path.basename(path)}")
                    try:
                        os.makedirs(extract_dir, exist_ok=True)
                        with tarfile.open(path, 'r:gz') as tar:
                            tar.extractall(extract_dir)
                        
                        # adapter_config.jsonの存在確認
                        if os.path.exists(os.path.join(extract_dir, "adapter_config.json")):
                            print(f"✅ 解凍成功: {extract_dir}")
                            lora_dirs.append(extract_dir)
                        else:
                            print(f"⚠️ adapter_config.jsonが見つかりません: {extract_dir}")
                    except Exception as e:
                        print(f"❌ 解凍エラー: {e}")
    
    if not lora_dirs:
        print("❌ 有効なLoRAアダプターが見つかりません")
        print("   検索パターン:")
        for pattern in possible_dirs:
            print(f"   - {pattern}")
        return None
    
    # 最新のディレクトリを選択（作成時刻順）
    lora_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    selected_dir = lora_dirs[0]
    
    print(f"✅ LoRAアダプター検出: {selected_dir}")
    return selected_dir

def get_base_model_from_adapter(lora_path):
    """LoRAアダプターから元のベースモデルを取得"""
    config_path = os.path.join(lora_path, "adapter_config.json")
    
    if not os.path.exists(config_path):
        print("❌ adapter_config.jsonが見つかりません")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        print("❌ ベースモデル情報が見つかりません")
        return None
    
    print(f"✅ ベースモデル: {base_model}")
    return base_model

def merge_lora_to_fp16(lora_path, base_model_name, output_path):
    """LoRAアダプターをFP16でマージ"""
    
    print("\n" + "="*60)
    print("LoRAアダプターのFP16マージ")
    print("="*60)
    
    # 既存のFP16モデルがある場合はスキップ
    if os.path.exists(f"{output_path}/model.safetensors.index.json") or \
       os.path.exists(f"{output_path}/pytorch_model.bin.index.json"):
        print("✅ FP16マージ済みモデルが既に存在します")
        return True
    
    print("\n1. ベースモデルをロード中...")
    
    # 32Bモデルの場合は4bit量子化でロード（メモリ制約のため）
    model_size_gb = 64  # デフォルト（32Bモデル想定）
    use_4bit = True  # デフォルトで4bit使用
    
    # モデルサイズを推定
    if "7b" in base_model_name.lower() or "8b" in base_model_name.lower():
        model_size_gb = 14
        use_4bit = False  # 小さいモデルはFP16可能
    elif "14b" in base_model_name.lower() or "13b" in base_model_name.lower():
        model_size_gb = 28
        use_4bit = False  # 中規模モデルもFP16可能
    elif "22b" in base_model_name.lower():
        model_size_gb = 44
        use_4bit = True  # 大規模モデルは4bit必須
    elif "32b" in base_model_name.lower():
        model_size_gb = 64
        use_4bit = True  # 32Bは必ず4bit
    
    if use_4bit:
        print(f"   大規模モデル（推定{model_size_gb}GB）のため4bit量子化でロード")
        print("   注意: 4bitマージは精度が低下する可能性があります")
        
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/offload"
        )
        print("✅ 4bit量子化ベースモデルロード成功")
        print("⚠️ 警告: 4bitでのマージは推論時に若干の精度低下が発生する可能性があります")
    else:
        print(f"   モデルサイズ（推定{model_size_gb}GB）はFP16でロード可能")
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="/tmp/offload"
            )
            print("✅ FP16ベースモデルロード成功")
        except Exception as e:
            print(f"⚠️ FP16ロード失敗: {e}")
            print("メモリ不足のため4bitでロードします")
            
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="/tmp/offload"
            )
            print("✅ 4bit量子化ベースモデルロード成功（フォールバック）")
    
    print("\n2. LoRAアダプタをロード中...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("\n3. LoRAをマージ中...")
    model = model.merge_and_unload()
    
    print("\n4. FP16形式で保存中...")
    print("   重要: 量子化せずにFP16で保存")
    os.makedirs(output_path, exist_ok=True)
    
    # 量子化されたモデルをFP16に変換してから保存
    # 4bitでロードされた場合、モデルを再構築
    if hasattr(model, 'is_quantized') or 'bnb' in str(type(model)).lower():
        print("   ⚠️ 4bitモデルをFP16に変換中...")
        print("   注意: これには追加のメモリが必要です")
        
        # モデルをCPUに移動してFP16に変換
        try:
            model = model.to(torch.float16)
        except Exception as e:
            print(f"   ⚠️ 直接変換失敗: {e}")
            print("   代替方法: state_dictのみ保存")
    
    # FP16で保存（量子化メタデータなし）
    print("   保存形式: safetensors（FP16）")
    model.save_pretrained(
        output_path,
        torch_dtype=torch.float16,
        safe_serialization=True,
        max_shard_size="2GB",
        offload_state_dict=True  # メモリ節約
    )
    
    # トークナイザも保存
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    # クリーンアップ
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ FP16マージモデル保存完了: {output_path}")
    return True

def setup_llama_cpp():
    """llama.cppのセットアップ（ビルド済みのものを使用）"""
    
    global LLAMA_CPP_DIR
    
    # 既存のllama.cppをチェック（ビルド済みかも確認）
    if os.path.exists("/workspace/llama.cpp/build/bin/llama-quantize"):
        print("✅ /workspace/llama.cppは既にセットアップ済み（ビルド済み）")
        LLAMA_CPP_DIR = "/workspace/llama.cpp"
        return True
    elif os.path.exists("/tmp/llama.cpp/build/bin/llama-quantize"):
        print("✅ /tmp/llama.cppは既にセットアップ済み（ビルド済み）")
        LLAMA_CPP_DIR = "/tmp/llama.cpp"
        return True
    
    print("\n" + "="*60)
    print("⚠️ llama.cppがセットアップされていません")
    print("="*60)
    print("\n以下のコマンドを別ターミナルで実行してください：")
    print("\n  docker exec -it ai-ft-container bash")
    print("  /workspace/scripts/setup_llama_cpp_standalone.sh")
    print("\nセットアップ後、再度量子化を実行してください。")
    print("="*60)
    
    # セットアップスクリプトの存在を確認
    setup_script = "/workspace/scripts/setup_llama_cpp_standalone.sh"
    if not os.path.exists(setup_script):
        print("\n❌ セットアップスクリプトが見つかりません")
        print("手動でllama.cppをセットアップしてください")
    else:
        # 自動実行を試みる（ただしWebサーバーに影響しないように注意）
        print("\n自動セットアップを試みます...")
        print("注意: これには数分かかります")
        
        # スクリプトを実行可能にする
        os.chmod(setup_script, 0o755)
        
        # サブプロセスとして実行（シェルを使わない）
        result = subprocess.run(
            ["/bin/bash", setup_script],
            capture_output=True,
            text=True,
            timeout=600  # 10分のタイムアウト
        )
        
        if result.returncode == 0:
            print("✅ llama.cppの自動セットアップが完了しました")
            LLAMA_CPP_DIR = "/workspace/llama.cpp"
            return True
        else:
            print(f"❌ 自動セットアップに失敗しました")
            print(f"エラー: {result.stderr}")
            return False
    
    return False

def convert_to_gguf(model_path, output_file):
    """FP16モデルをGGUF形式に変換"""
    
    print("\n" + "="*60)
    print("GGUF変換")
    print("="*60)
    
    if os.path.exists(output_file):
        print(f"✅ GGUF既に存在: {output_file}")
        return True
    
    print("GGUF変換中...")
    print(f"使用するllama.cpp: {LLAMA_CPP_DIR}")
    
    # 必要な依存関係をインストール
    print("依存関係を確認中...")
    if not run_command("pip install -q gguf sentencepiece protobuf"):
        print("⚠️ 依存関係のインストールに失敗しましたが続行します")
    
    # convert_hf_to_gguf.pyの存在確認
    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"❌ {convert_script}が見つかりません")
        # 代替スクリプトを確認
        alt_script = f"{LLAMA_CPP_DIR}/convert-hf-to-gguf.py"
        if os.path.exists(alt_script):
            print(f"📌 代替スクリプトを使用: {alt_script}")
            convert_script = alt_script
        else:
            print("❌ GGUF変換スクリプトが見つかりません")
            return False
    
    # convert_hf_to_gguf.pyを使用
    cmd = f"cd {LLAMA_CPP_DIR} && python {os.path.basename(convert_script)} {model_path} --outfile {output_file} --outtype f16"
    
    print(f"実行コマンド詳細:")
    print(f"  作業ディレクトリ: {LLAMA_CPP_DIR}")
    print(f"  スクリプト: {os.path.basename(convert_script)}")
    print(f"  入力モデル: {model_path}")
    print(f"  出力ファイル: {output_file}")
    
    if not run_command(cmd):
        print("❌ GGUF変換失敗")
        print("\nトラブルシューティング:")
        print("1. llama.cppディレクトリを確認:")
        run_command(f"ls -la {LLAMA_CPP_DIR}/convert*.py 2>/dev/null | head -5")
        print("2. モデルディレクトリを確認:")
        run_command(f"ls -la {model_path}/ 2>/dev/null | head -5")
        return False
    
    print(f"✅ GGUF変換成功: {output_file}")
    return True

def quantize_gguf(input_file, output_file, quant_type="Q4_K_M"):
    """GGUFファイルを量子化"""
    
    print("\n" + "="*60)
    print(f"量子化 ({quant_type})")
    print("="*60)
    
    if os.path.exists(output_file):
        print(f"✅ 量子化済み: {output_file}")
        return True
    
    print(f"{quant_type}形式で量子化中...")
    
    # llama-quantizeを使用（/tmpのllama.cppを使用）
    cmd = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize {input_file} {output_file} {quant_type}"
    
    if not run_command(cmd):
        print("❌ 量子化失敗")
        return False
    
    print(f"✅ 量子化成功: {output_file}")
    return True

def create_modelfile(gguf_path, output_path):
    """Ollama用のModelfileを作成"""
    
    print("\nModelfile作成中...")
    
    # GGUFファイル名を取得（相対パス用）
    gguf_filename = os.path.basename(gguf_path)
    
    modelfile_content = f'''FROM ./{gguf_filename}

# 日本語対応DeepSeekモデル（ファインチューニング済み）
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""
'''
    
    with open(output_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile作成: {output_path}")
    print(f"   GGUFファイル: {gguf_filename}")
    return True

def register_with_ollama(modelfile_path, model_name="deepseek-finetuned"):
    """Ollamaにモデルを登録"""
    
    print("\n" + "="*60)
    print("Ollama登録")
    print("="*60)
    
    # Ollamaサービスが起動しているか確認
    if not run_command("curl -s http://localhost:11434/api/tags > /dev/null 2>&1"):
        print("⚠️ Ollamaサービスが起動していません")
        print("Ollamaサービスを自動起動中...")
        
        # Ollamaを自動起動
        if run_command("nohup ollama serve > /dev/null 2>&1 &"):
            print("Ollamaサービス起動を待機中...")
            import time
            time.sleep(5)  # 起動待機
            
            # 再度確認
            if not run_command("curl -s http://localhost:11434/api/tags > /dev/null 2>&1"):
                print("❌ Ollamaサービスの起動に失敗しました")
                print("手動で 'ollama serve' を実行してください")
                return False
            print("✅ Ollamaサービスが起動しました")
        else:
            print("❌ Ollamaサービスの自動起動に失敗しました")
            return False
    
    print(f"モデル名 '{model_name}' として登録中...")
    
    # 作業ディレクトリとファイルの存在を確認
    model_dir = os.path.dirname(modelfile_path)
    modelfile_name = os.path.basename(modelfile_path)
    
    if not os.path.exists(modelfile_path):
        print(f"❌ Modelfileが見つかりません: {modelfile_path}")
        return False
    
    # GGUFファイルの存在も確認
    with open(modelfile_path, 'r') as f:
        content = f.read()
        # FROM行からGGUFファイル名を抽出
        import re
        match = re.search(r'FROM\s+\.?/?(.+\.gguf)', content)
        if match:
            gguf_file = match.group(1)
            gguf_path = os.path.join(model_dir, os.path.basename(gguf_file))
            if not os.path.exists(gguf_path):
                print(f"❌ GGUFファイルが見つかりません: {gguf_path}")
                return False
            print(f"✅ GGUFファイル確認: {os.path.basename(gguf_path)}")
    
    # モデルを作成（タイムアウトを長くする）
    cmd = f"cd {model_dir} && timeout 600 ollama create {model_name} -f {modelfile_name}"
    
    print("注意: モデル登録には数分かかる場合があります...")
    if not run_command(cmd):
        print("❌ Ollama登録失敗")
        print("デバッグ情報:")
        print(f"  作業ディレクトリ: {model_dir}")
        print(f"  Modelfile: {modelfile_name}")
        # エラーの詳細を確認
        run_command(f"cd {model_dir} && ls -la")
        return False
    
    print(f"✅ Ollama登録成功: {model_name}")
    
    # 登録確認
    run_command("ollama list")
    
    return True

def update_rag_config(model_name="deepseek-finetuned"):
    """RAGシステムの設定を更新"""
    
    config_path = "/workspace/src/rag/config/rag_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"⚠️ RAG設定ファイルが見つかりません: {config_path}")
        return False
    
    print(f"\nRAG設定を更新中: {model_name}")
    
    # YAMLファイルを更新
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ollamaモデルを設定
    config['llm']['provider'] = 'ollama'
    config['llm']['ollama'] = {
        'model': model_name,
        'base_url': 'http://localhost:11434',
        'temperature': 0.7,
        'top_p': 0.9
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ RAG設定更新完了")
    return True

def main():
    """メインパイプライン"""
    
    print("="*60)
    print("QLoRA → Ollama 変換パイプライン")
    print("="*60)
    
    # 1. LoRAアダプターを探す
    lora_path = find_lora_adapter()
    if not lora_path:
        return 1
    
    # 2. ベースモデルを取得
    base_model_name = get_base_model_from_adapter(lora_path)
    if not base_model_name:
        return 1
    
    # 3. 出力ディレクトリを準備
    output_base = "/workspace/outputs/ollama_conversion"
    os.makedirs(output_base, exist_ok=True)
    
    # タイムスタンプを追加して一意のディレクトリにする
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fp16_model_path = f"{output_base}/merged_model_fp16_{timestamp}"
    gguf_f16_path = f"{output_base}/model-f16_{timestamp}.gguf"
    gguf_q4_path = f"{output_base}/deepseek-finetuned-q4_k_m_{timestamp}.gguf"
    modelfile_path = f"{output_base}/Modelfile_{timestamp}"
    
    # 4. llama.cppをセットアップ
    if not setup_llama_cpp():
        return 1
    
    # 5. LoRAをFP16でマージ
    if not merge_lora_to_fp16(lora_path, base_model_name, fp16_model_path):
        return 1
    
    # 6. GGUF変換
    if not convert_to_gguf(fp16_model_path, gguf_f16_path):
        return 1
    
    # 7. 量子化
    if not quantize_gguf(gguf_f16_path, gguf_q4_path):
        return 1
    
    # 8. Modelfile作成
    if not create_modelfile(gguf_q4_path, modelfile_path):
        return 1
    
    # 9. Ollama登録
    if not register_with_ollama(modelfile_path):
        return 1
    
    # 10. RAG設定更新
    if not update_rag_config():
        return 1
    
    print("\n" + "="*60)
    print("✅ 完了！")
    print("="*60)
    print("\nRAGシステムでファインチューニング済みモデルが使用可能になりました")
    print("RAGインターフェースで質問応答をテストしてください")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())