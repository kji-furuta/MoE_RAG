#!/usr/bin/env python3
"""
モデル探索ユーティリティ
ファインチューニングシステムで作成された全てのモデルを検出
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """ファインチューニング済みモデルの探索と管理"""
    
    def __init__(self, outputs_dir: str = None):
        # Dockerコンテナ環境を考慮したパス設定
        if outputs_dir:
            self.outputs_dir = Path(outputs_dir)
        else:
            # Docker環境のデフォルトパス
            docker_path = Path("/workspace/outputs")
            # ローカル環境のフォールバック
            local_path = Path("outputs")
            
            if docker_path.exists():
                self.outputs_dir = docker_path
            elif local_path.exists():
                self.outputs_dir = local_path
            else:
                # 最終的なフォールバック
                self.outputs_dir = docker_path
        self.model_patterns = {
            "lora": ["lora_*", "*/final_lora_model", "*/best_lora_model"],
            "merged": ["merged_model", "*/merged_model", "merged_model_*"],
            "continual": ["continual_task_*", "ewc_model_*"],
            "gguf": ["*.gguf", "*/*.gguf"],
            "ollama": ["ollama_conversion/*"]
        }
    
    def find_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """全てのファインチューニング済みモデルを検出"""
        all_models = {
            "lora_adapters": [],
            "merged_models": [],
            "continual_models": [],
            "gguf_models": [],
            "ollama_ready": []
        }
        
        if not self.outputs_dir.exists():
            logger.warning(f"Outputs directory not found: {self.outputs_dir}")
            return all_models
        
        # LoRAアダプターを検索
        all_models["lora_adapters"] = self._find_lora_adapters()
        
        # マージ済みモデルを検索
        all_models["merged_models"] = self._find_merged_models()
        
        # 継続学習モデルを検索
        all_models["continual_models"] = self._find_continual_models()
        
        # GGUFファイルを検索
        all_models["gguf_models"] = self._find_gguf_models()
        
        # Ollama登録可能なモデルを検索
        all_models["ollama_ready"] = self._find_ollama_ready_models()
        
        return all_models
    
    def _find_lora_adapters(self) -> List[Dict[str, Any]]:
        """LoRAアダプターを検索"""
        lora_models = []
        
        # 複数のパターンで検索
        patterns = [
            "lora_*",
            "*/final_lora_model",
            "*/best_lora_model",
            "checkpoint-*"
        ]
        
        for pattern in patterns:
            for path in self.outputs_dir.glob(pattern):
                if path.is_dir():
                    adapter_config = path / "adapter_config.json"
                    if adapter_config.exists():
                        model_info = self._extract_lora_info(path)
                        if model_info and model_info not in lora_models:
                            lora_models.append(model_info)
        
        # タイムスタンプでソート（新しいものから）
        lora_models.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
        
        return lora_models
    
    def _extract_lora_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """LoRAアダプターの情報を抽出"""
        try:
            adapter_config = path / "adapter_config.json"
            if not adapter_config.exists():
                return None
            
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            
            model_info = {
                "type": "lora_adapter",
                "path": str(path),
                "name": path.name,
                "base_model": config.get("base_model_name_or_path", "Unknown"),
                "r": config.get("r", None),
                "lora_alpha": config.get("lora_alpha", None),
                "target_modules": config.get("target_modules", []),
                "modified_time": path.stat().st_mtime,
                "modified_date": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
            # アダプターファイルのサイズを確認
            adapter_files = list(path.glob("adapter_model.*"))
            if adapter_files:
                total_size = sum(f.stat().st_size for f in adapter_files)
                model_info["size_mb"] = total_size / (1024 * 1024)
            
            # トレーニング情報があれば追加
            training_args = path / "training_args.json"
            if training_args.exists():
                with open(training_args, 'r') as f:
                    train_config = json.load(f)
                    model_info["training_info"] = {
                        "epochs": train_config.get("num_train_epochs"),
                        "batch_size": train_config.get("per_device_train_batch_size"),
                        "learning_rate": train_config.get("learning_rate")
                    }
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Failed to extract LoRA info from {path}: {e}")
            return None
    
    def _find_merged_models(self) -> List[Dict[str, Any]]:
        """マージ済みモデルを検索"""
        merged_models = []
        
        patterns = [
            "merged_model",
            "*/merged_model",
            "merged_model_*",
            "merged_*"
        ]
        
        for pattern in patterns:
            for path in self.outputs_dir.glob(pattern):
                if path.is_dir():
                    # config.jsonまたはmodel.safetensors.index.jsonを確認
                    config_file = path / "config.json"
                    index_file = path / "model.safetensors.index.json"
                    
                    if config_file.exists() or index_file.exists():
                        model_info = self._extract_merged_model_info(path)
                        if model_info and model_info not in merged_models:
                            merged_models.append(model_info)
        
        merged_models.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
        
        return merged_models
    
    def _extract_merged_model_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """マージ済みモデルの情報を抽出"""
        try:
            model_info = {
                "type": "merged_model",
                "path": str(path),
                "name": path.name,
                "modified_time": path.stat().st_mtime,
                "modified_date": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
            # config.jsonから情報を取得
            config_file = path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    model_info["model_type"] = config.get("model_type", "Unknown")
                    model_info["hidden_size"] = config.get("hidden_size")
                    model_info["num_hidden_layers"] = config.get("num_hidden_layers")
            
            # モデルファイルのサイズを計算
            model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files)
                model_info["size_gb"] = total_size / (1024 ** 3)
            
            # 量子化情報があれば追加
            quantization_config = path / "quantization_config.json"
            if quantization_config.exists():
                with open(quantization_config, 'r') as f:
                    quant_config = json.load(f)
                    model_info["quantization"] = {
                        "load_in_4bit": quant_config.get("load_in_4bit", False),
                        "load_in_8bit": quant_config.get("load_in_8bit", False),
                        "quant_type": quant_config.get("bnb_4bit_quant_type", None)
                    }
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Failed to extract merged model info from {path}: {e}")
            return None
    
    def _find_continual_models(self) -> List[Dict[str, Any]]:
        """継続学習モデルを検索"""
        continual_models = []
        
        patterns = [
            "continual_task_*",
            "ewc_model_*",
            "ewc_data/task_*"
        ]
        
        for pattern in patterns:
            for path in self.outputs_dir.glob(pattern):
                if path.is_dir():
                    # タスク情報ファイルを確認
                    task_info = path / "task_info.json"
                    if task_info.exists():
                        model_info = self._extract_continual_model_info(path)
                        if model_info and model_info not in continual_models:
                            continual_models.append(model_info)
        
        # EWCタスク履歴も確認
        task_history_file = self.outputs_dir / "ewc_data" / "task_history.json"
        if task_history_file.exists():
            try:
                with open(task_history_file, 'r') as f:
                    task_history = json.load(f)
                    for task in task_history.get("tasks", []):
                        task_path = Path(task.get("model_path", ""))
                        if task_path.exists() and task_path.is_dir():
                            model_info = self._extract_continual_model_info(task_path)
                            if model_info and model_info not in continual_models:
                                continual_models.append(model_info)
            except Exception as e:
                logger.warning(f"Failed to read task history: {e}")
        
        continual_models.sort(key=lambda x: x.get("task_id", 0))
        
        return continual_models
    
    def _extract_continual_model_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """継続学習モデルの情報を抽出"""
        try:
            model_info = {
                "type": "continual_model",
                "path": str(path),
                "name": path.name,
                "modified_time": path.stat().st_mtime,
                "modified_date": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
            # タスク情報を読み込み
            task_info_file = path / "task_info.json"
            if task_info_file.exists():
                with open(task_info_file, 'r') as f:
                    task_info = json.load(f)
                    model_info["task_id"] = task_info.get("task_id")
                    model_info["task_name"] = task_info.get("task_name")
                    model_info["base_model"] = task_info.get("base_model")
                    model_info["ewc_lambda"] = task_info.get("ewc_lambda")
                    model_info["training_samples"] = task_info.get("num_samples")
            
            # Fisher行列の存在を確認
            fisher_file = self.outputs_dir / "ewc_data" / f"fisher_task_{model_info.get('task_id', 0)}.pt"
            model_info["has_fisher_matrix"] = fisher_file.exists()
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Failed to extract continual model info from {path}: {e}")
            return None
    
    def _find_gguf_models(self) -> List[Dict[str, Any]]:
        """GGUFファイルを検索"""
        gguf_models = []
        
        for gguf_file in self.outputs_dir.glob("**/*.gguf"):
            model_info = {
                "type": "gguf_model",
                "path": str(gguf_file),
                "name": gguf_file.name,
                "size_gb": gguf_file.stat().st_size / (1024 ** 3),
                "modified_time": gguf_file.stat().st_mtime,
                "modified_date": datetime.fromtimestamp(gguf_file.stat().st_mtime).isoformat(),
                "quantization": self._guess_quantization_from_name(gguf_file.name)
            }
            
            # Modelfileが同じディレクトリにあるか確認
            modelfile = gguf_file.parent / "Modelfile"
            model_info["has_modelfile"] = modelfile.exists()
            model_info["ollama_ready"] = model_info["has_modelfile"]
            
            gguf_models.append(model_info)
        
        gguf_models.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
        
        return gguf_models
    
    def _guess_quantization_from_name(self, filename: str) -> str:
        """ファイル名から量子化形式を推測"""
        filename_lower = filename.lower()
        
        if "q4_k_m" in filename_lower:
            return "Q4_K_M"
        elif "q4_0" in filename_lower:
            return "Q4_0"
        elif "q5_k_m" in filename_lower:
            return "Q5_K_M"
        elif "q8_0" in filename_lower:
            return "Q8_0"
        elif "f16" in filename_lower:
            return "F16"
        elif "f32" in filename_lower:
            return "F32"
        else:
            return "Unknown"
    
    def _find_ollama_ready_models(self) -> List[Dict[str, Any]]:
        """Ollama登録可能なモデルを検索"""
        ollama_models = []
        
        # ollama_conversionディレクトリを確認
        ollama_dir = self.outputs_dir / "ollama_conversion"
        if ollama_dir.exists():
            for modelfile in ollama_dir.glob("*/Modelfile"):
                model_dir = modelfile.parent
                gguf_files = list(model_dir.glob("*.gguf"))
                
                if gguf_files:
                    model_info = {
                        "type": "ollama_ready",
                        "path": str(model_dir),
                        "name": model_dir.name,
                        "modelfile": str(modelfile),
                        "gguf_file": str(gguf_files[0]),
                        "size_gb": gguf_files[0].stat().st_size / (1024 ** 3),
                        "modified_time": modelfile.stat().st_mtime,
                        "modified_date": datetime.fromtimestamp(modelfile.stat().st_mtime).isoformat()
                    }
                    
                    # Modelfileから情報を抽出
                    try:
                        with open(modelfile, 'r') as f:
                            content = f.read()
                            if "FROM " in content:
                                from_line = [l for l in content.split('\n') if l.startswith("FROM ")][0]
                                model_info["base_gguf"] = from_line.replace("FROM ", "").strip()
                    except:
                        pass
                    
                    ollama_models.append(model_info)
        
        return ollama_models
    
    def get_model_summary(self) -> Dict[str, Any]:
        """モデルのサマリー情報を取得"""
        all_models = self.find_all_models()
        
        summary = {
            "total_models": sum(len(models) for models in all_models.values()),
            "by_type": {
                "lora_adapters": len(all_models["lora_adapters"]),
                "merged_models": len(all_models["merged_models"]),
                "continual_models": len(all_models["continual_models"]),
                "gguf_models": len(all_models["gguf_models"]),
                "ollama_ready": len(all_models["ollama_ready"])
            },
            "latest_model": None,
            "recommended_for_rag": []
        }
        
        # 最新のモデルを特定
        all_model_list = []
        for model_list in all_models.values():
            all_model_list.extend(model_list)
        
        if all_model_list:
            all_model_list.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
            summary["latest_model"] = all_model_list[0]
        
        # RAG用の推奨モデルを選定
        # 1. Ollama登録可能なモデル
        summary["recommended_for_rag"].extend(all_models["ollama_ready"])
        
        # 2. GGUFファイル（Modelfileがあるもの）
        for gguf in all_models["gguf_models"]:
            if gguf.get("has_modelfile"):
                summary["recommended_for_rag"].append(gguf)
        
        # 3. マージ済みモデル（量子化可能）
        for merged in all_models["merged_models"]:
            if merged not in summary["recommended_for_rag"]:
                merged["needs_quantization"] = True
                summary["recommended_for_rag"].append(merged)
        
        return summary
    
    def get_model_for_rag(self, model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """RAGシステムで使用するモデルを取得"""
        
        if model_path:
            # 指定されたパスのモデルを返す
            path = Path(model_path)
            if path.exists():
                if path.suffix == ".gguf":
                    return {
                        "type": "gguf",
                        "path": str(path),
                        "ready": True
                    }
                elif path.is_dir():
                    # ディレクトリの種類を判定
                    if (path / "adapter_config.json").exists():
                        return {
                            "type": "lora_adapter",
                            "path": str(path),
                            "ready": False,
                            "needs": "merge_and_quantize"
                        }
                    elif (path / "config.json").exists():
                        return {
                            "type": "merged_model",
                            "path": str(path),
                            "ready": False,
                            "needs": "quantize"
                        }
        
        # 最新の利用可能なモデルを返す
        all_models = self.find_all_models()
        
        # 優先順位: Ollama ready > GGUF > Merged > LoRA
        if all_models["ollama_ready"]:
            return all_models["ollama_ready"][0]
        elif all_models["gguf_models"]:
            return all_models["gguf_models"][0]
        elif all_models["merged_models"]:
            model = all_models["merged_models"][0]
            model["needs_quantization"] = True
            return model
        elif all_models["lora_adapters"]:
            model = all_models["lora_adapters"][0]
            model["needs_merge_and_quantization"] = True
            return model
        
        return None


def test_discovery():
    """モデル探索のテスト"""
    discovery = ModelDiscovery()
    
    print("="*60)
    print("ファインチューニング済みモデル検出")
    print("="*60)
    
    all_models = discovery.find_all_models()
    
    for model_type, models in all_models.items():
        print(f"\n{model_type}: {len(models)} models")
        for model in models[:3]:  # 最初の3つを表示
            print(f"  - {model['name']} ({model['type']})")
            print(f"    Path: {model['path']}")
            print(f"    Modified: {model.get('modified_date', 'Unknown')}")
    
    print("\n" + "="*60)
    print("サマリー")
    print("="*60)
    
    summary = discovery.get_summary()
    print(f"総モデル数: {summary['total_models']}")
    print(f"最新モデル: {summary['latest_model']['name'] if summary['latest_model'] else 'None'}")
    print(f"RAG推奨モデル数: {len(summary['recommended_for_rag'])}")


if __name__ == "__main__":
    test_discovery()