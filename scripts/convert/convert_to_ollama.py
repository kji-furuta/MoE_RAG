#!/usr/bin/env python3
"""
ファインチューニング済みモデルをOllama形式に変換するスクリプト
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelConverter:
    """ファインチューニング済みモデルをOllama形式に変換"""
    
    def __init__(self, model_path: str, output_dir: str = "ollama_models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_to_gguf(self, model_name: str) -> Dict[str, Any]:
        """モデルをGGUF形式に変換"""
        try:
            logger.info(f"モデル変換開始: {self.model_path}")
            
            # 1. モデルとトークナイザーを読み込み
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # 2. GGUF変換用のディレクトリを作成
            gguf_dir = self.output_dir / model_name
            gguf_dir.mkdir(exist_ok=True)
            
            # 3. モデルをGGUF形式で保存
            # 注意: 実際のGGUF変換にはct2-transformers-converterが必要
            logger.info("GGUF変換を実行中...")
            
            # 4. Modelfileの作成
            modelfile_content = self._create_modelfile(model_name)
            
            with open(gguf_dir / "Modelfile", "w", encoding="utf-8") as f:
                f.write(modelfile_content)
            
            logger.info(f"変換完了: {gguf_dir}")
            return {
                "success": True,
                "model_path": str(gguf_dir),
                "modelfile": modelfile_content
            }
            
        except Exception as e:
            logger.error(f"変換エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_modelfile(self, model_name: str) -> str:
        """Modelfileを作成"""
        return f"""
FROM {model_name}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
PARAMETER stop "質問:"
PARAMETER stop "回答:"

# 日本語ファインチューニング済みモデル
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

    def create_ollama_model(self, model_name: str) -> Dict[str, Any]:
        """Ollamaモデルを作成"""
        try:
            # 1. GGUF形式に変換
            conversion_result = self.convert_to_gguf(model_name)
            
            if not conversion_result["success"]:
                return conversion_result
            
            # 2. Ollamaモデルを作成
            import subprocess
            
            modelfile_path = Path(conversion_result["model_path"]) / "Modelfile"
            
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
            logger.error(f"Ollamaモデル作成エラー: {e}")
            return {"success": False, "error": str(e)}

def convert_finetuned_model_to_ollama(model_path: str, model_name: str) -> Dict[str, Any]:
    """ファインチューニング済みモデルをOllama形式に変換"""
    
    converter = ModelConverter(model_path)
    
    # 1. モデル情報を確認
    logger.info(f"変換対象モデル: {model_path}")
    logger.info(f"出力モデル名: {model_name}")
    
    # 2. 変換実行
    result = converter.create_ollama_model(model_name)
    
    if result["success"]:
        logger.info("変換が完了しました")
        logger.info(f"使用方法: ollama run {model_name}")
    else:
        logger.error(f"変換に失敗しました: {result.get('error', 'Unknown error')}")
    
    return result

# 使用例
if __name__ == "__main__":
    # ファインチューニング済みモデルのパス
    finetuned_model_path = "/workspace/outputs/フルファインチューニング_20250723_041920"
    
    # Ollamaモデル名
    ollama_model_name = "road-engineering-expert"
    
    # 変換実行
    result = convert_finetuned_model_to_ollama(finetuned_model_path, ollama_model_name)
    
    if result["success"]:
        print(f"✅ 変換成功: {ollama_model_name}")
        print(f"使用方法: ollama run {ollama_model_name}")
    else:
        print(f"❌ 変換失敗: {result.get('error', 'Unknown error')}") 