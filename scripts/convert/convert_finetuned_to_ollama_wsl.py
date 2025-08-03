#!/usr/bin/env python3
"""
WSL環境用：ファインチューニング済みモデルをOllamaで使用できるように変換するスクリプト
"""

import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class WSLFinetunedModelConverter:
    """WSL環境用：ファインチューニング済みモデルをOllama形式に変換"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.output_dir = Path("ollama_models")
        self.output_dir.mkdir(exist_ok=True)
        
    def check_wsl_environment(self) -> bool:
        """WSL環境の確認"""
        try:
            # WSL環境の確認
            with open('/proc/version', 'r') as f:
                version_info = f.read()
                if 'microsoft' in version_info.lower():
                    logger.info("WSL環境を検出しました")
                    return True
                else:
                    logger.warning("WSL環境ではありません")
                    return False
        except Exception as e:
            logger.error(f"環境確認エラー: {e}")
            return False
    
    def install_ollama_wsl(self) -> bool:
        """WSL環境でOllamaをインストール"""
        try:
            logger.info("WSL環境でOllamaをインストール中...")
            
            # インストールスクリプトの実行
            install_script = Path("install_ollama_wsl.sh")
            if install_script.exists():
                subprocess.run(["bash", str(install_script)], check=True)
            else:
                # 手動インストール
                subprocess.run([
                    "curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"
                ], shell=True, check=True)
            
            # 環境変数の設定
            os.environ['PATH'] = f"{os.environ.get('HOME')}/.local/bin:{os.environ.get('PATH')}"
            
            # Ollamaの確認
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Ollamaインストール成功: {result.stdout.strip()}")
                return True
            else:
                logger.error("Ollamaのインストールに失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"Ollamaインストールエラー: {e}")
            return False
    
    def convert_to_gguf(self, model_name: str) -> Dict[str, Any]:
        """ファインチューニング済みモデルをGGUF形式に変換（WSL対応）"""
        try:
            logger.info(f"WSL環境でファインチューニング済みモデル変換開始: {self.model_path}")
            
            # 1. モデル情報の確認
            if not self.model_path.exists():
                return {"success": False, "error": "モデルパスが存在しません"}
            
            # 2. llama-cpp-pythonのインストール確認
            try:
                import llama_cpp
                logger.info("llama-cpp-pythonが利用可能です")
            except ImportError:
                logger.info("llama-cpp-pythonをインストール中...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "llama-cpp-python"
                ], check=True)
            
            # 3. GGUF変換の実行
            return self._run_gguf_conversion_wsl(model_name)
            
        except Exception as e:
            logger.error(f"変換エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_gguf_conversion_wsl(self, model_name: str) -> Dict[str, Any]:
        """WSL環境でGGUF変換を実行"""
        try:
            # llama.cppを使用したGGUF変換
            output_file = self.output_dir / f"{model_name}.gguf"
            
            # WSL環境用の変換コマンド
            cmd = [
                sys.executable, "-m", "llama_cpp.convert",
                str(self.model_path),
                "--outfile", str(output_file),
                "--outtype", "q4_k_m"  # 4bit量子化でメモリ効率化
            ]
            
            logger.info(f"WSL GGUF変換コマンド: {' '.join(cmd)}")
            
            # 変換実行（WSL環境での最適化）
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '4'  # スレッド数を制限
            env['MKL_NUM_THREADS'] = '4'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2時間のタイムアウト
                env=env
            )
            
            if result.returncode == 0:
                logger.info(f"WSL GGUF変換完了: {output_file}")
                return {
                    "success": True,
                    "gguf_path": str(output_file),
                    "model_name": model_name
                }
            else:
                logger.error(f"WSL GGUF変換エラー: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logger.error("WSL GGUF変換がタイムアウトしました")
            return {"success": False, "error": "Conversion timeout"}
        except Exception as e:
            logger.error(f"WSL GGUF変換エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def create_ollama_modelfile_wsl(self, gguf_path: str, model_name: str) -> str:
        """WSL環境用のOllama Modelfileを作成"""
        template_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
PARAMETER stop "質問:"
PARAMETER stop "回答:"

# WSL環境用：ファインチューニング済み道路工学専門家モデル
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
        return template_content
    
    def setup_ollama_model_wsl(self, gguf_path: str, model_name: str) -> Dict[str, Any]:
        """WSL環境でOllamaモデルをセットアップ"""
        try:
            # 1. Modelfileを作成
            modelfile_content = self.create_ollama_modelfile_wsl(gguf_path, model_name)
            
            # 2. Modelfileを保存
            modelfile_path = self.output_dir / f"{model_name}.Modelfile"
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)
            
            # 3. WSL環境でOllamaモデルを作成
            cmd = ["ollama", "create", model_name, str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"WSL Ollamaモデル作成成功: {model_name}")
                return {
                    "success": True,
                    "model_name": model_name,
                    "message": "WSL環境でOllamaモデルが正常に作成されました"
                }
            else:
                return {
                    "success": False,
                    "error": f"WSL Ollamaモデル作成失敗: {result.stderr}"
                }
                
        except Exception as e:
            logger.error(f"WSL Ollamaモデルセットアップエラー: {e}")
            return {"success": False, "error": str(e)}
    
    def convert_and_setup_wsl(self, model_name: str) -> Dict[str, Any]:
        """WSL環境でファインチューニング済みモデルを変換してOllamaで使用可能にする"""
        try:
            logger.info("WSL環境でファインチューニング済みモデルの変換を開始します")
            
            # 1. WSL環境の確認
            if not self.check_wsl_environment():
                logger.warning("WSL環境ではありませんが、処理を続行します")
            
            # 2. Ollamaのインストール確認
            try:
                subprocess.run(["ollama", "--version"], capture_output=True, check=True)
                logger.info("Ollamaが利用可能です")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Ollamaをインストールします")
                if not self.install_ollama_wsl():
                    return {"success": False, "error": "Ollamaのインストールに失敗しました"}
            
            # 3. GGUF変換
            conversion_result = self.convert_to_gguf(model_name)
            
            if not conversion_result["success"]:
                return conversion_result
            
            # 4. Ollamaモデルのセットアップ
            gguf_path = conversion_result["gguf_path"]
            ollama_result = self.setup_ollama_model_wsl(gguf_path, model_name)
            
            if ollama_result["success"]:
                logger.info("✅ WSL環境での変換とセットアップが完了しました")
                return {
                    "success": True,
                    "model_name": model_name,
                    "gguf_path": gguf_path,
                    "message": "WSL環境でファインチューニング済みモデルがOllamaで使用可能になりました"
                }
            else:
                return ollama_result
                
        except Exception as e:
            logger.error(f"WSL変換・セットアップエラー: {e}")
            return {"success": False, "error": str(e)}

def main():
    """WSL環境用メイン処理"""
    # ファインチューニング済みモデルのパス
    finetuned_model_path = "/workspace/outputs/フルファインチューニング_20250723_041920"
    
    # Ollamaモデル名
    ollama_model_name = "road-engineering-expert"
    
    logger.info("WSL環境でのファインチューニング済みモデルのOllama変換を開始します")
    
    # 1. WSL環境での変換・セットアップ
    converter = WSLFinetunedModelConverter(finetuned_model_path)
    result = converter.convert_and_setup_wsl(ollama_model_name)
    
    if result["success"]:
        logger.info("✅ WSL環境での変換が完了しました")
        logger.info(f"使用方法: ollama run {ollama_model_name}")
        logger.info(f"例: ollama run {ollama_model_name} '縦断曲線とは何ですか？'")
        
        # 使用例を表示
        print("\n" + "="*50)
        print("🎉 WSL環境でのファインチューニング済みモデルの変換が完了しました！")
        print("="*50)
        print(f"モデル名: {ollama_model_name}")
        print(f"使用方法: ollama run {ollama_model_name}")
        print("例:")
        print(f"  ollama run {ollama_model_name} '縦断曲線とは何ですか？'")
        print(f"  ollama run {ollama_model_name} '道路の横断勾配の標準値は？'")
        print("="*50)
        
    else:
        logger.error(f"❌ WSL環境での変換に失敗しました: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 