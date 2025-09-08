#!/usr/bin/env python3
"""
シンプル化したLoRA to Ollama変換スクリプト
既にマージ済みのモデルをGGUF化してOllamaに登録する
"""

import os
import subprocess
import sys
import time

def run_command(cmd, description=""):
    """コマンドを実行"""
    if description:
        print(f"\n{description}", flush=True)
    print(f"実行: {cmd}", flush=True)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ エラー: {result.stderr}", flush=True)
        return False
    
    if result.stdout:
        print(result.stdout, flush=True)
    
    return True

def main():
    """メイン処理"""
    print("="*60, flush=True)
    print("シンプル LoRA → Ollama 変換", flush=True)
    print("="*60, flush=True)
    
    # 既存のマージ済みモデルを確認
    merged_path = "/workspace/outputs/merged_model"
    
    if os.path.exists(merged_path):
        print(f"✅ マージ済みモデル発見: {merged_path}", flush=True)
    else:
        print(f"❌ マージ済みモデルが見つかりません", flush=True)
        print("先に local_dual_gpu_merge.py を実行してください", flush=True)
        return False
    
    # Step 1: GGUF変換の準備
    print("\n" + "="*60, flush=True)
    print("Step 1: GGUF変換準備", flush=True)
    print("="*60, flush=True)
    
    gguf_path = "/workspace/outputs/model-f16.gguf"
    
    # llama.cppの準備
    if not os.path.exists("/workspace/llama.cpp"):
        print("llama.cppをクローン中...", flush=True)
        if not run_command("cd /workspace && git clone https://github.com/ggerganov/llama.cpp"):
            return False
        
        print("llama.cppをCMakeでビルド中...", flush=True)
        # CMakeを使用してビルド（CURLを無効化）
        if not run_command("cd /workspace/llama.cpp && cmake -B build -DLLAMA_CURL=OFF"):
            print("CMakeビルド設定失敗", flush=True)
            return False
        if not run_command("cd /workspace/llama.cpp && cmake --build build --config Release -j8"):
            print("CMakeビルド失敗", flush=True)
            return False
    
    # Step 2: GGUF変換
    if not os.path.exists(gguf_path):
        print("\n" + "="*60, flush=True)
        print("Step 2: GGUF変換", flush=True)
        print("="*60, flush=True)
        
        # 必要なパッケージをインストール
        print("必要なパッケージをインストール中...", flush=True)
        run_command("pip install -q gguf sentencepiece protobuf mistral-common")
        
        print("変換中（10-15分かかります）...", flush=True)
        
        cmd = f"""cd /workspace/llama.cpp && python convert_hf_to_gguf.py \
            {merged_path} \
            --outfile {gguf_path} \
            --outtype f16"""
        
        if not run_command(cmd):
            print("❌ GGUF変換に失敗しました", flush=True)
            return False
        
        print("✅ GGUF変換完了", flush=True)
    else:
        print(f"✅ GGUF既に存在: {gguf_path}", flush=True)
    
    # Step 3: 量子化
    gguf_q4_path = "/workspace/outputs/deepseek-finetuned-q4_k_m.gguf"
    
    if not os.path.exists(gguf_q4_path):
        print("\n" + "="*60, flush=True)
        print("Step 3: 量子化（Q4_K_M）", flush=True)
        print("="*60, flush=True)
        
        print("Quantizing", flush=True)  # UI進捗用
        print("量子化中（10-15分かかります）...", flush=True)
        
        # quantizeバイナリの場所を確認（CMakeビルドの場合）
        quantize_path = None
        if os.path.exists("/workspace/llama.cpp/build/bin/llama-quantize"):
            quantize_path = "/workspace/llama.cpp/build/bin/llama-quantize"
        elif os.path.exists("/workspace/llama.cpp/build/llama-quantize"):
            quantize_path = "/workspace/llama.cpp/build/llama-quantize"
        elif os.path.exists("/workspace/llama.cpp/llama-quantize"):
            quantize_path = "/workspace/llama.cpp/llama-quantize"
        else:
            print("量子化ツールが見つかりません。ビルドを試行...", flush=True)
            run_command("cd /workspace/llama.cpp && cmake --build build --target llama-quantize")
            if os.path.exists("/workspace/llama.cpp/build/bin/llama-quantize"):
                quantize_path = "/workspace/llama.cpp/build/bin/llama-quantize"
            else:
                print("❌ 量子化ツールのビルドに失敗しました", flush=True)
                return False
        
        cmd = f"{quantize_path} {gguf_path} {gguf_q4_path} Q4_K_M"
        
        if not run_command(cmd):
            print("❌ 量子化に失敗しました", flush=True)
            return False
        
        print("✅ 量子化完了", flush=True)
        
        # 中間ファイル削除
        if os.path.exists(gguf_path):
            os.remove(gguf_path)
            print("中間ファイル削除済み", flush=True)
    else:
        print(f"✅ 量子化済みファイル既に存在: {gguf_q4_path}", flush=True)
    
    # Step 4: Modelfile作成
    print("\n" + "="*60, flush=True)
    print("Step 4: Modelfile作成", flush=True)
    print("="*60, flush=True)
    
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
    
    print("✅ Modelfile作成完了", flush=True)
    
    # Step 5: Ollama登録
    print("\n" + "="*60, flush=True)
    print("Step 5: Ollama登録", flush=True)
    print("="*60, flush=True)
    
    # Ollama起動確認
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ollamaを起動中...", flush=True)
        subprocess.Popen("ollama serve", shell=True)
        time.sleep(5)
    
    # 既存モデル削除
    run_command("ollama rm deepseek-finetuned 2>/dev/null || true")
    
    # モデル登録
    print("モデルを登録中...", flush=True)
    cmd = "cd /workspace/outputs && ollama create deepseek-finetuned -f Modelfile"
    
    if not run_command(cmd):
        print("❌ Ollama登録に失敗しました", flush=True)
        return False
    
    print("✅ Ollama登録完了", flush=True)
    
    # Step 6: RAG設定更新
    print("\n" + "="*60, flush=True)
    print("Step 6: RAG設定更新", flush=True)
    print("="*60, flush=True)
    
    config_path = "/workspace/src/rag/config/rag_config.yaml"
    
    # 簡単な置換
    cmd = f"sed -i 's/ollama_model:.*/ollama_model: deepseek-finetuned/g' {config_path}"
    run_command(cmd)
    
    print("✅ RAG設定更新完了", flush=True)
    
    # 完了
    print("\n" + "="*60, flush=True)
    print("✅ すべての処理が完了しました！", flush=True)
    print("="*60, flush=True)
    print("\n使用方法:", flush=True)
    print("1. Ollama: ollama run deepseek-finetuned", flush=True)
    print("2. RAG: http://localhost:8050/rag", flush=True)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)