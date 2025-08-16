"""
Ollama統合モジュール - メモリ効率的な推論のための自動フォールバック
"""

import os
import json
import torch
import psutil
import requests
from typing import Optional, Dict, Any, Union
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

class HybridModelManager:
    """
    HuggingFaceモデルとOllamaモデルのハイブリッド管理
    メモリ不足時に自動的にOllamaにフォールバック
    """
    
    def __init__(self, config_path: str = "/workspace/config/ollama_config.json"):
        self.config = self._load_config(config_path)
        self.ollama_base_url = self.config["ollama_integration"]["base_url"]
        self.memory_threshold_gb = self.config["ollama_integration"]["memory_threshold_gb"]
        self.current_model = None
        self.current_tokenizer = None
        self.use_ollama = False
        
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込み"""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return {
            "ollama_integration": {
                "enabled": True,
                "base_url": "http://localhost:11434",
                "fallback_on_oom": True,
                "models": {
                    "primary": "llama3.2:3b",
                    "fallback": "llama3.2:3b"
                },
                "memory_threshold_gb": 8,
                "auto_select": True
            }
        }
    
    def check_memory_availability(self) -> tuple[float, float]:
        """利用可能なGPUメモリを確認"""
        if torch.cuda.is_available():
            try:
                # GPU メモリ情報
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_allocated = torch.cuda.memory_allocated(0)
                gpu_memory_free = gpu_memory - gpu_memory_allocated
                gpu_memory_free_gb = gpu_memory_free / (1024**3)
                
                # システムRAM情報
                ram = psutil.virtual_memory()
                ram_free_gb = ram.available / (1024**3)
                
                logger.info(f"📊 メモリ状況: GPU={gpu_memory_free_gb:.1f}GB, RAM={ram_free_gb:.1f}GB")
                return gpu_memory_free_gb, ram_free_gb
            except Exception as e:
                logger.warning(f"⚠️ GPUメモリ確認エラー: {e}")
        
        # CPUのみの場合
        ram = psutil.virtual_memory()
        ram_free_gb = ram.available / (1024**3)
        return 0, ram_free_gb
    
    def should_use_ollama(self, model_name: str) -> bool:
        """Ollamaを使用すべきか判定"""
        gpu_free, ram_free = self.check_memory_availability()
        
        # モデルサイズの推定（概算）
        model_size_map = {
            "70b": 140, "32b": 64, "22b": 44, 
            "14b": 28, "13b": 26, "8b": 16, "7b": 14, "3b": 6
        }
        
        estimated_size = 14  # デフォルト
        for size_key, size_gb in model_size_map.items():
            if size_key in model_name.lower():
                estimated_size = size_gb
                break
        
        # 量子化考慮（4bit想定）
        estimated_size_quantized = estimated_size / 4
        
        # メモリ不足判定
        if gpu_free < estimated_size_quantized or gpu_free < self.memory_threshold_gb:
            logger.warning(f"⚠️ メモリ不足: 必要={estimated_size_quantized:.1f}GB, 利用可能={gpu_free:.1f}GB")
            return True
        
        return False
    
    def load_model(self, model_name: str, force_ollama: bool = False) -> bool:
        """モデルをロード（自動選択）"""
        try:
            # Ollamaを強制使用またはメモリ不足の場合
            if force_ollama or self.should_use_ollama(model_name):
                logger.info(f"🔄 Ollamaモデルを使用: {self.config['ollama_integration']['models']['primary']}")
                self.use_ollama = True
                self.current_model = self.config['ollama_integration']['models']['primary']
                return self._test_ollama_connection()
            
            # HuggingFaceモデルをロード
            logger.info(f"📦 HuggingFaceモデルをロード: {model_name}")
            from app.memory_optimized_loader import load_model_with_optimization
            
            self.current_model, self.current_tokenizer = load_model_with_optimization(
                model_name,
                device_map="auto",
                load_in_4bit=True  # メモリ節約のため4bit量子化
            )
            self.use_ollama = False
            return True
            
        except torch.cuda.OutOfMemoryError:
            logger.error("❌ GPU OOM - Ollamaにフォールバック")
            if self.config["ollama_integration"]["fallback_on_oom"]:
                self.use_ollama = True
                self.current_model = self.config['ollama_integration']['models']['fallback']
                return self._test_ollama_connection()
            return False
            
        except Exception as e:
            logger.error(f"❌ モデルロードエラー: {e}")
            return False
    
    def _test_ollama_connection(self) -> bool:
        """Ollama接続テスト"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 512,
                temperature: float = 0.7,
                **kwargs) -> Optional[str]:
        """テキスト生成（自動選択）"""
        
        if self.use_ollama:
            return self._generate_with_ollama(prompt, max_tokens, temperature)
        else:
            return self._generate_with_hf(prompt, max_tokens, temperature, **kwargs)
    
    def _generate_with_ollama(self, 
                             prompt: str, 
                             max_tokens: int,
                             temperature: float) -> Optional[str]:
        """Ollamaでテキスト生成"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.current_model,
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
                generated_text = result.get("response", "")
                logger.info(f"✅ Ollama生成完了: {len(generated_text)}文字")
                return generated_text
            else:
                logger.error(f"Ollama生成エラー: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama生成例外: {e}")
            return None
    
    def _generate_with_hf(self, 
                         prompt: str, 
                         max_tokens: int,
                         temperature: float,
                         **kwargs) -> Optional[str]:
        """HuggingFaceモデルでテキスト生成"""
        try:
            if not self.current_model or not self.current_tokenizer:
                logger.error("HuggingFaceモデルが未ロード")
                return None
            
            inputs = self.current_tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    **kwargs
                )
            
            generated_text = self.current_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            logger.info(f"✅ HF生成完了: {len(generated_text)}文字")
            return generated_text
            
        except Exception as e:
            logger.error(f"HF生成エラー: {e}")
            return None
    
    def cleanup(self):
        """リソースをクリーンアップ"""
        if not self.use_ollama and self.current_model:
            del self.current_model
            del self.current_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("🧹 モデルメモリを解放")

# グローバルインスタンス
_hybrid_manager = None

def get_hybrid_manager() -> HybridModelManager:
    """シングルトンマネージャーを取得"""
    global _hybrid_manager
    if _hybrid_manager is None:
        _hybrid_manager = HybridModelManager()
    return _hybrid_manager

def generate_with_best_model(
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs
) -> Optional[str]:
    """
    最適なモデルで生成（メモリ状況に応じて自動選択）
    
    Args:
        model_name: 希望するモデル名
        prompt: 入力プロンプト
        max_tokens: 最大生成トークン数
        temperature: 温度パラメータ
        
    Returns:
        生成されたテキスト
    """
    manager = get_hybrid_manager()
    
    # モデルをロード
    if not manager.load_model(model_name):
        logger.error("モデルロード失敗")
        return None
    
    # テキスト生成
    result = manager.generate(prompt, max_tokens, temperature, **kwargs)
    
    # クリーンアップ（必要に応じて）
    # manager.cleanup()
    
    return result

if __name__ == "__main__":
    # テスト実行
    test_prompt = "日本の四季について説明してください。"
    
    # メモリ効率的な生成
    result = generate_with_best_model(
        "cyberagent/calm3-22b-chat",  # 大規模モデルを指定
        test_prompt,
        max_tokens=200
    )
    
    if result:
        print(f"生成結果:\n{result}")
    else:
        print("生成失敗")