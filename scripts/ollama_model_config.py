#!/usr/bin/env python3
"""
Ollama モデル統合設定
メモリ不足時にOllamaモデルを自動的に使用
"""

import os
import json
import requests
from typing import Dict, Optional, List
from loguru import logger

class OllamaModelManager:
    """Ollamaモデル管理クラス"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = {}
        self.model_mapping = {
            # ファインチューニング済みモデル -> Ollamaフォールバックモデル
            "calm3-22b": "llama3.2:3b",
            "qwen2.5-14b": "llama3.2:3b", 
            "qwen2.5-32b": "llama3.2:3b",
            "deepseek-r1": "llama3.2:3b",
            # デフォルトフォールバック
            "default": "llama3.2:3b"
        }
        
    def check_ollama_status(self) -> bool:
        """Ollamaサービスの状態を確認"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.available_models = {
                    model["name"]: model 
                    for model in response.json().get("models", [])
                }
                logger.info(f"✅ Ollama利用可能: {list(self.available_models.keys())}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Ollama接続エラー: {e}")
        return False
    
    def get_fallback_model(self, original_model: str) -> Optional[str]:
        """メモリ不足時のフォールバックモデルを取得"""
        # モデル名を正規化
        model_key = original_model.lower()
        for key in self.model_mapping:
            if key in model_key:
                fallback = self.model_mapping[key]
                if fallback in self.available_models:
                    logger.info(f"🔄 フォールバック: {original_model} -> {fallback}")
                    return fallback
        
        # デフォルトフォールバック
        default = self.model_mapping.get("default")
        if default in self.available_models:
            logger.info(f"🔄 デフォルトフォールバック使用: {default}")
            return default
        
        logger.error("❌ 利用可能なフォールバックモデルがありません")
        return None
    
    def generate_with_ollama(self, 
                           model: str, 
                           prompt: str, 
                           max_tokens: int = 512,
                           temperature: float = 0.7) -> Optional[str]:
        """Ollamaを使用してテキスト生成"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama生成エラー: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama生成例外: {e}")
            return None
    
    def setup_model_files(self):
        """カスタムモデルファイルを設定"""
        modelfiles_dir = "/workspace/ollama_modelfiles"
        os.makedirs(modelfiles_dir, exist_ok=True)
        
        # 日本語対応Modelfile
        japanese_modelfile = '''FROM llama3.2:3b

TEMPLATE """[INST] <<SYS>>
You are a helpful and capable Japanese language assistant.
<</SYS>>

{{ .Prompt }} [/INST]"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 1024
PARAMETER stop "[/INST]"
PARAMETER stop "</s>"

SYSTEM """You are an assistant that answers questions in Japanese politely and explains technical content clearly."""
'''
        
        # Modelfileを保存
        with open(f"{modelfiles_dir}/japanese_assistant.modelfile", "w") as f:
            f.write(japanese_modelfile)
        
        logger.info(f"✅ Modelfileを作成: {modelfiles_dir}")
        
    def register_custom_models(self):
        """カスタムモデルをOllamaに登録"""
        modelfiles = [
            ("japanese-assistant", "/workspace/ollama_modelfiles/japanese_assistant.modelfile")
        ]
        
        for model_name, modelfile_path in modelfiles:
            if os.path.exists(modelfile_path):
                try:
                    # ollama create コマンドを実行
                    import subprocess
                    result = subprocess.run(
                        ["ollama", "create", model_name, "-f", modelfile_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ モデル登録成功: {model_name}")
                    else:
                        logger.warning(f"⚠️ モデル登録失敗: {model_name} - {result.stderr}")
                except Exception as e:
                    logger.error(f"❌ モデル登録エラー: {e}")

def integrate_with_app():
    """アプリケーションとの統合設定"""
    config = {
        "ollama_integration": {
            "enabled": True,
            "base_url": "http://localhost:11434",
            "fallback_on_oom": True,  # OOM時に自動フォールバック
            "models": {
                "primary": "llama3.2:3b",
                "fallback": "llama3.2:3b",
                "custom": ["japanese-assistant"]
            },
            "memory_threshold_gb": 8,  # この閾値以下でOllamaを使用
            "auto_select": True  # 自動モデル選択
        }
    }
    
    # 設定をファイルに保存
    config_path = "/workspace/config/ollama_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Ollama統合設定を保存: {config_path}")
    return config

if __name__ == "__main__":
    # Ollamaマネージャーを初期化
    manager = OllamaModelManager()
    
    # サービス状態を確認
    if manager.check_ollama_status():
        print("✅ Ollamaサービスが正常に動作しています")
        print(f"📦 利用可能なモデル: {list(manager.available_models.keys())}")
        
        # Modelfileを設定
        manager.setup_model_files()
        
        # アプリケーション統合設定
        config = integrate_with_app()
        print(f"⚙️ 統合設定完了: {config}")
        
        # テスト生成
        test_prompt = "日本の首都はどこですか？"
        response = manager.generate_with_ollama("llama3.2:3b", test_prompt, max_tokens=100)
        if response:
            print(f"\n📝 テスト生成結果:\n{response}")
    else:
        print("❌ Ollamaサービスに接続できません")