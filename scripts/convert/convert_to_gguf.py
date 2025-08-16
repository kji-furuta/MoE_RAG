#!/usr/bin/env python3
"""
ファインチューニング済みモデルをGGUF形式に変換するスクリプト
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def convert_to_gguf(model_path: str, output_name: str) -> Dict[str, Any]:
    """
    ファインチューニング済みモデルをGGUF形式に変換
    
    Args:
        model_path: ファインチューニング済みモデルのパス
        output_name: 出力モデル名
    
    Returns:
        変換結果
    """
    try:
        logger.info(f"GGUF変換開始: {model_path}")
        
        # 1. llama.cppのインストール確認
        if not check_llama_cpp():
            logger.error("llama.cppがインストールされていません")
            return {"success": False, "error": "llama.cpp not found"}
        
        # 2. 出力ディレクトリの作成
        output_dir = Path("gguf_models")
        output_dir.mkdir(exist_ok=True)
        
        # 3. llama.cppを使用してGGUF変換
        cmd = [
            "python3", "-m", "llama_cpp.convert",
            str(model_path),
            "--outfile", str(output_dir / f"{output_name}.gguf"),
            "--outtype", "q4_k_m"  # 4bit量子化
        ]
        
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1時間のタイムアウト
        )
        
        if result.returncode == 0:
            logger.info("GGUF変換が完了しました")
            return {
                "success": True,
                "output_path": str(output_dir / f"{output_name}.gguf"),
                "message": "GGUF変換が正常に完了しました"
            }
        else:
            logger.error(f"GGUF変換エラー: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        logger.error("GGUF変換がタイムアウトしました")
        return {"success": False, "error": "Conversion timeout"}
    except Exception as e:
        logger.error(f"GGUF変換エラー: {e}")
        return {"success": False, "error": str(e)}

def check_llama_cpp() -> bool:
    """llama.cppがインストールされているかチェック"""
    try:
        result = subprocess.run(
            ["python3", "-c", "import llama_cpp"],
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False

def install_llama_cpp():
    """llama.cppをインストール"""
    try:
        logger.info("llama.cppをインストール中...")
        subprocess.run([
            "pip", "install", "llama-cpp-python"
        ], check=True)
        logger.info("llama.cppのインストールが完了しました")
        return True
    except Exception as e:
        logger.error(f"llama.cppのインストールに失敗: {e}")
        return False

def create_ollama_modelfile(gguf_path: str, model_name: str) -> str:
    """Ollama用のModelfileを作成"""
    return f"""
FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
PARAMETER stop "質問:"
PARAMETER stop "回答:"

# 道路工学専門家モデル
TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """あなたは道路工学の専門家です。質問に対して正確で分かりやすい回答を提供してください。"""
"""

def setup_ollama_model(gguf_path: str, model_name: str) -> Dict[str, Any]:
    """Ollamaモデルをセットアップ"""
    try:
        # 1. Modelfileを作成
        modelfile_content = create_ollama_modelfile(gguf_path, model_name)
        
        # 2. Modelfileを保存
        modelfile_path = Path(f"{model_name}.Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        # 3. Ollamaモデルを作成
        cmd = ["ollama", "create", model_name, str(modelfile_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Ollamaモデル作成成功: {model_name}")
            return {
                "success": True,
                "model_name": model_name,
                "message": "Ollamaモデルが正常に作成されました"
            }
        else:
            return {
                "success": False,
                "error": f"Ollamaモデル作成失敗: {result.stderr}"
            }
            
    except Exception as e:
        logger.error(f"Ollamaモデルセットアップエラー: {e}")
        return {"success": False, "error": str(e)}

def main():
    """メイン処理"""
    # ファインチューニング済みモデルのパス
    model_path = "/workspace/outputs/フルファインチューニング_20250723_041920"
    output_name = "road-engineering-expert"
    
    logger.info("ファインチューニング済みモデルのGGUF変換を開始します")
    
    # 1. llama.cppの確認・インストール
    if not check_llama_cpp():
        logger.info("llama.cppをインストールします")
        if not install_llama_cpp():
            logger.error("llama.cppのインストールに失敗しました")
            return
    
    # 2. GGUF変換
    conversion_result = convert_to_gguf(model_path, output_name)
    
    if not conversion_result["success"]:
        logger.error(f"GGUF変換に失敗: {conversion_result.get('error')}")
        return
    
    # 3. Ollamaモデルのセットアップ
    gguf_path = conversion_result["output_path"]
    ollama_result = setup_ollama_model(gguf_path, output_name)
    
    if ollama_result["success"]:
        logger.info("✅ 変換とセットアップが完了しました")
        logger.info(f"使用方法: ollama run {output_name}")
        logger.info(f"例: ollama run {output_name} '縦断曲線とは何ですか？'")
    else:
        logger.error(f"❌ Ollamaセットアップに失敗: {ollama_result.get('error')}")

if __name__ == "__main__":
    main() 