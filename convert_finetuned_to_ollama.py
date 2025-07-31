#!/usr/bin/env python3
"""
ファインチューニング済みモデルをOllamaで使用できるように変換するスクリプト
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class FinetunedModelConverter:
    """ファインチューニング済みモデルをOllama形式に変換"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.output_dir = Path("ollama_models")
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_to_gguf(self, model_name: str) -> Dict[str, Any]:
        """ファインチューニング済みモデルをGGUF形式に変換"""
        try:
            logger.info(f"ファインチューニング済みモデル変換開始: {self.model_path}")
            
            # 1. モデル情報の確認
            if not self.model_path.exists():
                return {"success": False, "error": "モデルパスが存在しません"}
            
            # 2. 設定ファイルの読み込み
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"モデル設定: {config.get('model_type', 'unknown')}")
            
            # 3. GGUF変換の実行
            return self._run_gguf_conversion(model_name)
            
        except Exception as e:
            logger.error(f"変換エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_gguf_conversion(self, model_name: str) -> Dict[str, Any]:
        """GGUF変換を実行"""
        try:
            # llama.cppを使用したGGUF変換
            output_file = self.output_dir / f"{model_name}.gguf"
            
            # 変換コマンド
            cmd = [
                "python3", "-m", "llama_cpp.convert",
                str(self.model_path),
                "--outfile", str(output_file),
                "--outtype", "q4_k_m"  # 4bit量子化でメモリ効率化
            ]
            
            logger.info(f"GGUF変換コマンド: {' '.join(cmd)}")
            
            # 変換実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2時間のタイムアウト
            )
            
            if result.returncode == 0:
                logger.info(f"GGUF変換完了: {output_file}")
                return {
                    "success": True,
                    "gguf_path": str(output_file),
                    "model_name": model_name
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
    
    def create_ollama_modelfile(self, gguf_path: str, model_name: str) -> str:
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

# ファインチューニング済み道路工学専門家モデル
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
    
    def setup_ollama_model(self, gguf_path: str, model_name: str) -> Dict[str, Any]:
        """Ollamaモデルをセットアップ"""
        try:
            # 1. Modelfileを作成
            modelfile_content = self.create_ollama_modelfile(gguf_path, model_name)
            
            # 2. Modelfileを保存
            modelfile_path = self.output_dir / f"{model_name}.Modelfile"
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
    
    def convert_and_setup(self, model_name: str) -> Dict[str, Any]:
        """ファインチューニング済みモデルを変換してOllamaで使用可能にする"""
        try:
            logger.info("ファインチューニング済みモデルの変換を開始します")
            
            # 1. GGUF変換
            conversion_result = self.convert_to_gguf(model_name)
            
            if not conversion_result["success"]:
                return conversion_result
            
            # 2. Ollamaモデルのセットアップ
            gguf_path = conversion_result["gguf_path"]
            ollama_result = self.setup_ollama_model(gguf_path, model_name)
            
            if ollama_result["success"]:
                logger.info("✅ 変換とセットアップが完了しました")
                return {
                    "success": True,
                    "model_name": model_name,
                    "gguf_path": gguf_path,
                    "message": "ファインチューニング済みモデルがOllamaで使用可能になりました"
                }
            else:
                return ollama_result
                
        except Exception as e:
            logger.error(f"変換・セットアップエラー: {e}")
            return {"success": False, "error": str(e)}

def install_requirements():
    """必要な依存関係をインストール"""
    try:
        logger.info("必要な依存関係をインストール中...")
        
        # llama-cpp-pythonのインストール
        subprocess.run([
            "pip", "install", "llama-cpp-python"
        ], check=True)
        
        # Ollamaのインストール確認
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Ollamaがインストールされていません。手動でインストールしてください:")
            logger.warning("curl -fsSL https://ollama.ai/install.sh | sh")
        
        logger.info("依存関係のインストールが完了しました")
        return True
        
    except Exception as e:
        logger.error(f"依存関係のインストールに失敗: {e}")
        return False

def main():
    """メイン処理"""
    # ファインチューニング済みモデルのパス
    finetuned_model_path = "/workspace/outputs/フルファインチューニング_20250723_041920"
    
    # Ollamaモデル名
    ollama_model_name = "road-engineering-expert"
    
    logger.info("ファインチューニング済みモデルのOllama変換を開始します")
    
    # 1. 依存関係の確認・インストール
    if not install_requirements():
        logger.error("依存関係のインストールに失敗しました")
        return
    
    # 2. 変換・セットアップ
    converter = FinetunedModelConverter(finetuned_model_path)
    result = converter.convert_and_setup(ollama_model_name)
    
    if result["success"]:
        logger.info("✅ 変換が完了しました")
        logger.info(f"使用方法: ollama run {ollama_model_name}")
        logger.info(f"例: ollama run {ollama_model_name} '縦断曲線とは何ですか？'")
        
        # 使用例を表示
        print("\n" + "="*50)
        print("🎉 ファインチューニング済みモデルの変換が完了しました！")
        print("="*50)
        print(f"モデル名: {ollama_model_name}")
        print(f"使用方法: ollama run {ollama_model_name}")
        print("例:")
        print(f"  ollama run {ollama_model_name} '縦断曲線とは何ですか？'")
        print(f"  ollama run {ollama_model_name} '道路の横断勾配の標準値は？'")
        print("="*50)
        
    else:
        logger.error(f"❌ 変換に失敗しました: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 