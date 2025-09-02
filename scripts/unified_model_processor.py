#!/usr/bin/env python3
"""
統合モデル処理スクリプト
ファインチューニングシステムで作成された全てのモデルタイプを
RAGシステムで使用可能な形式に変換
"""

import os
import sys
import json
import subprocess
import torch
import gc
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append('/workspace')
from src.utils.model_discovery import ModelDiscovery
from app.memory_optimized_loader import get_optimal_quantization_config


class UnifiedModelProcessor:
    """全モデルタイプ対応の統合処理"""
    
    def __init__(self):
        self.discovery = ModelDiscovery()
        self.output_base = Path("/workspace/outputs/rag_ready_models")
        self.output_base.mkdir(parents=True, exist_ok=True)
    
    def process_model(self, model_path: str, model_type: str = None) -> Dict[str, Any]:
        """モデルを処理してRAG用に準備"""
        
        path = Path(model_path)
        
        # モデルタイプを自動判定
        if model_type is None:
            model_type = self._detect_model_type(path)
        
        print(f"\n処理開始: {path.name}")
        print(f"モデルタイプ: {model_type}")
        
        if model_type == "lora_adapter":
            return self._process_lora_adapter(path)
        elif model_type == "merged_model":
            return self._process_merged_model(path)
        elif model_type == "continual_model":
            return self._process_continual_model(path)
        elif model_type == "gguf_model":
            return self._process_gguf_model(path)
        elif model_type == "ollama_ready":
            return self._process_ollama_ready(path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _detect_model_type(self, path: Path) -> str:
        """モデルタイプを自動検出"""
        
        if path.suffix == ".gguf":
            return "gguf_model"
        
        if path.is_dir():
            # LoRAアダプター
            if (path / "adapter_config.json").exists():
                return "lora_adapter"
            
            # マージ済みモデル
            if (path / "config.json").exists():
                # 継続学習モデルかチェック
                if (path / "task_info.json").exists():
                    return "continual_model"
                return "merged_model"
            
            # Ollama ready
            if (path / "Modelfile").exists():
                return "ollama_ready"
        
        raise ValueError(f"Cannot detect model type for: {path}")
    
    def _process_lora_adapter(self, path: Path) -> Dict[str, Any]:
        """LoRAアダプターを処理（マージ→量子化→Ollama登録）"""
        
        print("\n[LoRAアダプター処理]")
        
        # 1. アダプター情報を読み込み
        with open(path / "adapter_config.json", 'r') as f:
            config = json.load(f)
        
        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Base model not found in adapter config")
        
        # 2. 出力ディレクトリを準備
        output_dir = self.output_base / f"lora_{path.name}_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. マージ処理
        print(f"ベースモデル: {base_model}")
        print("マージ処理中...")
        
        merged_path = output_dir / "merged_model"
        if not self._merge_lora(path, base_model, merged_path):
            return {"status": "failed", "error": "Merge failed"}
        
        # 4. GGUF変換
        print("GGUF変換中...")
        gguf_path = output_dir / "model.gguf"
        if not self._convert_to_gguf(merged_path, gguf_path):
            return {"status": "failed", "error": "GGUF conversion failed"}
        
        # 5. 量子化
        print("量子化中...")
        quantized_path = output_dir / "model-q4_k_m.gguf"
        if not self._quantize_gguf(gguf_path, quantized_path):
            return {"status": "failed", "error": "Quantization failed"}
        
        # 6. Ollama登録
        print("Ollama登録中...")
        model_name = f"finetuned-{path.name}"
        if not self._register_ollama(quantized_path, model_name):
            return {"status": "failed", "error": "Ollama registration failed"}
        
        return {
            "status": "success",
            "model_name": model_name,
            "output_dir": str(output_dir),
            "gguf_path": str(quantized_path)
        }
    
    def _process_merged_model(self, path: Path) -> Dict[str, Any]:
        """マージ済みモデルを処理（量子化→Ollama登録）"""
        
        print("\n[マージ済みモデル処理]")
        
        # 出力ディレクトリを準備
        output_dir = self.output_base / f"merged_{path.name}_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. GGUF変換
        print("GGUF変換中...")
        gguf_path = output_dir / "model.gguf"
        if not self._convert_to_gguf(path, gguf_path):
            return {"status": "failed", "error": "GGUF conversion failed"}
        
        # 2. 量子化
        print("量子化中...")
        quantized_path = output_dir / "model-q4_k_m.gguf"
        if not self._quantize_gguf(gguf_path, quantized_path):
            return {"status": "failed", "error": "Quantization failed"}
        
        # 3. Ollama登録
        print("Ollama登録中...")
        model_name = f"merged-{path.name}"
        if not self._register_ollama(quantized_path, model_name):
            return {"status": "failed", "error": "Ollama registration failed"}
        
        return {
            "status": "success",
            "model_name": model_name,
            "output_dir": str(output_dir),
            "gguf_path": str(quantized_path)
        }
    
    def _process_continual_model(self, path: Path) -> Dict[str, Any]:
        """継続学習モデルを処理"""
        
        print("\n[継続学習モデル処理]")
        
        # タスク情報を読み込み
        task_info_file = path / "task_info.json"
        if task_info_file.exists():
            with open(task_info_file, 'r') as f:
                task_info = json.load(f)
            task_name = task_info.get("task_name", "unknown")
        else:
            task_name = path.name
        
        # マージ済みモデルと同じ処理
        result = self._process_merged_model(path)
        
        if result["status"] == "success":
            result["model_name"] = f"continual-{task_name}"
            result["task_info"] = task_info if 'task_info' in locals() else {}
        
        return result
    
    def _process_gguf_model(self, path: Path) -> Dict[str, Any]:
        """GGUFモデルを処理（Ollama登録）"""
        
        print("\n[GGUFモデル処理]")
        
        # Modelfileを作成
        modelfile_path = path.parent / "Modelfile"
        if not modelfile_path.exists():
            self._create_modelfile(path, modelfile_path)
        
        # Ollama登録
        model_name = f"gguf-{path.stem}"
        if not self._register_ollama(path, model_name):
            return {"status": "failed", "error": "Ollama registration failed"}
        
        return {
            "status": "success",
            "model_name": model_name,
            "gguf_path": str(path)
        }
    
    def _process_ollama_ready(self, path: Path) -> Dict[str, Any]:
        """Ollama登録済みモデルを確認"""
        
        print("\n[Ollama Ready モデル]")
        print("既に使用可能な状態です")
        
        # モデル名を推測
        modelfile = path / "Modelfile"
        if modelfile.exists():
            model_name = path.name
        else:
            model_name = "unknown"
        
        return {
            "status": "success",
            "model_name": model_name,
            "ready": True
        }
    
    def _merge_lora(self, lora_path: Path, base_model: str, output_path: Path) -> bool:
        """LoRAアダプターをマージ"""
        try:
            # メモリ最適化設定を取得
            quantization_config, device_map = get_optimal_quantization_config(base_model)
            
            # ベースモデルをロード
            if "32B" in base_model or "22B" in base_model:
                # 大規模モデルは4bit量子化
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            # LoRAアダプターをロード
            model = PeftModel.from_pretrained(base, lora_path)
            
            # マージ
            model = model.merge_and_unload()
            
            # 保存
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                output_path,
                torch_dtype=torch.float16,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # トークナイザーも保存
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)
            
            # クリーンアップ
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"マージエラー: {e}")
            return False
    
    def _convert_to_gguf(self, model_path: Path, output_file: Path) -> bool:
        """モデルをGGUF形式に変換"""
        try:
            # llama.cppのセットアップ確認
            if not Path("/workspace/llama.cpp").exists():
                self._setup_llama_cpp()
            
            cmd = f"cd /workspace/llama.cpp && python convert_hf_to_gguf.py {model_path} --outfile {output_file} --outtype f16"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"GGUF変換エラー: {e}")
            return False
    
    def _quantize_gguf(self, input_file: Path, output_file: Path, quant_type: str = "Q4_K_M") -> bool:
        """GGUFファイルを量子化"""
        try:
            cmd = f"/workspace/llama.cpp/build/bin/llama-quantize {input_file} {output_file} {quant_type}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"量子化エラー: {e}")
            return False
    
    def _create_modelfile(self, gguf_path: Path, output_path: Path):
        """Modelfileを作成"""
        content = f'''FROM {gguf_path.name}

# ファインチューニング済みモデル
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
            f.write(content)
    
    def _register_ollama(self, model_path: Path, model_name: str) -> bool:
        """Ollamaにモデルを登録"""
        try:
            # Modelfileが必要
            if model_path.suffix == ".gguf":
                modelfile = model_path.parent / "Modelfile"
                if not modelfile.exists():
                    self._create_modelfile(model_path, modelfile)
            else:
                modelfile = model_path / "Modelfile"
            
            # 既存モデルを削除
            subprocess.run(f"ollama rm {model_name}", shell=True, capture_output=True)
            
            # モデルを作成
            cmd = f"cd {modelfile.parent} && ollama create {model_name} -f {modelfile.name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Ollama登録成功: {model_name}")
                return True
            else:
                print(f"❌ Ollama登録失敗: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Ollama登録エラー: {e}")
            return False
    
    def _setup_llama_cpp(self):
        """llama.cppをセットアップ"""
        print("llama.cppをセットアップ中...")
        
        cmd = """
        cd /workspace && \
        git clone https://github.com/ggerganov/llama.cpp && \
        cd llama.cpp && \
        cmake -B build -DLLAMA_CURL=OFF -DLLAMA_CUDA=ON && \
        cmake --build build --config Release -j$(nproc)
        """
        
        subprocess.run(cmd, shell=True)
    
    def process_all_models(self) -> Dict[str, Any]:
        """全てのモデルを処理"""
        
        print("="*60)
        print("全モデル自動処理")
        print("="*60)
        
        all_models = self.discovery.find_all_models()
        results = []
        
        # 処理が必要なモデルを特定
        models_to_process = []
        
        # LoRAアダプター
        for model in all_models["lora_adapters"]:
            models_to_process.append({
                "path": model["path"],
                "type": "lora_adapter",
                "name": model["name"]
            })
        
        # マージ済みモデル（GGUFでないもの）
        for model in all_models["merged_models"]:
            models_to_process.append({
                "path": model["path"],
                "type": "merged_model",
                "name": model["name"]
            })
        
        # 継続学習モデル
        for model in all_models["continual_models"]:
            models_to_process.append({
                "path": model["path"],
                "type": "continual_model",
                "name": model["name"]
            })
        
        print(f"\n処理対象: {len(models_to_process)} モデル")
        
        for i, model in enumerate(models_to_process, 1):
            print(f"\n[{i}/{len(models_to_process)}] {model['name']}")
            
            try:
                result = self.process_model(model["path"], model["type"])
                results.append({
                    "model": model["name"],
                    "result": result
                })
            except Exception as e:
                print(f"❌ エラー: {e}")
                results.append({
                    "model": model["name"],
                    "result": {"status": "failed", "error": str(e)}
                })
        
        # サマリー表示
        print("\n" + "="*60)
        print("処理結果サマリー")
        print("="*60)
        
        success_count = sum(1 for r in results if r["result"]["status"] == "success")
        print(f"成功: {success_count}/{len(results)}")
        
        for result in results:
            status = "✅" if result["result"]["status"] == "success" else "❌"
            print(f"{status} {result['model']}")
            if result["result"]["status"] == "success":
                print(f"   → Ollama: {result['result'].get('model_name')}")
        
        return {
            "total": len(models_to_process),
            "success": success_count,
            "results": results
        }


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="統合モデル処理")
    parser.add_argument("--model", type=str, help="特定のモデルパスを処理")
    parser.add_argument("--all", action="store_true", help="全モデルを自動処理")
    parser.add_argument("--type", type=str, help="モデルタイプを指定")
    
    args = parser.parse_args()
    
    processor = UnifiedModelProcessor()
    
    if args.all:
        # 全モデル処理
        processor.process_all_models()
    elif args.model:
        # 特定モデル処理
        result = processor.process_model(args.model, args.type)
        print(f"\n結果: {result}")
    else:
        # モデル一覧表示
        discovery = ModelDiscovery()
        summary = discovery.get_model_summary()
        
        print("="*60)
        print("利用可能なモデル")
        print("="*60)
        print(f"総数: {summary['total_models']}")
        print(f"タイプ別:")
        for type_name, count in summary["by_type"].items():
            print(f"  - {type_name}: {count}")
        
        if summary["latest_model"]:
            print(f"\n最新モデル: {summary['latest_model']['name']}")
            print(f"  タイプ: {summary['latest_model']['type']}")
            print(f"  パス: {summary['latest_model']['path']}")


if __name__ == "__main__":
    main()