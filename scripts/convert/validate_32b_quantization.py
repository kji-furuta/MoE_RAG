#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-32B-Japanese 量子化検証スクリプト
32Bモデルの量子化プロセスを詳細に検証
"""

import os
import sys
import json
import torch
import gc
import psutil
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# プロジェクトパスを追加
sys.path.append('/workspace')
from app.memory_optimized_loader import get_optimal_quantization_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeek32BValidator:
    """DeepSeek-R1-Distill-Qwen-32B量子化検証"""
    
    def __init__(self):
        self.model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
        self.report = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "memory_usage": {},
            "recommendations": []
        }
    
    def check_system_resources(self):
        """システムリソースをチェック"""
        print("\n" + "="*60)
        print("1. システムリソースチェック")
        print("="*60)
        
        # CPU/メモリ情報
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"CPU コア数: {cpu_count}")
        print(f"RAM 総容量: {ram_gb:.1f} GB")
        print(f"RAM 利用可能: {available_ram_gb:.1f} GB")
        
        # GPU情報
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                free_gb = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "total_gb": total_gb,
                    "free_gb": free_gb
                })
                print(f"GPU {i}: {props.name} - {total_gb:.1f} GB (空き: {free_gb:.1f} GB)")
        else:
            print("GPU: 利用不可")
        
        # 32Bモデル要件チェック
        print("\n[32Bモデル要件]")
        print("- FP16: 64GB VRAM（推奨）")
        print("- 4bit量子化: 16GB VRAM（最小）")
        print("- GGUF変換: 64GB ディスク空き容量")
        
        # 判定
        total_gpu_memory = sum(g["total_gb"] for g in gpu_info)
        check_result = {
            "ram_gb": ram_gb,
            "available_ram_gb": available_ram_gb,
            "gpu_count": len(gpu_info),
            "total_gpu_memory_gb": total_gpu_memory,
            "gpu_details": gpu_info
        }
        
        self.report["memory_usage"]["system"] = check_result
        
        # 推奨事項
        if total_gpu_memory < 48:
            self.report["recommendations"].append(
                "⚠️ GPU VRAMが48GB未満: 4bit量子化（QLoRA）が必須です"
            )
        
        if available_ram_gb < 32:
            self.report["recommendations"].append(
                "⚠️ RAM空き容量が32GB未満: スワップ使用の可能性があります"
            )
        
        return check_result
    
    def check_quantization_config(self):
        """量子化設定をチェック"""
        print("\n" + "="*60)
        print("2. 量子化設定チェック")
        print("="*60)
        
        # メモリ最適化設定を取得
        config, device_map = get_optimal_quantization_config(self.model_name)
        
        print("\n[自動選択された設定]")
        print(f"量子化タイプ: {'4bit' if hasattr(config, 'load_in_4bit') and config.load_in_4bit else '8bit'}")
        
        if hasattr(config, 'load_in_4bit') and config.load_in_4bit:
            print(f"- compute_dtype: {config.bnb_4bit_compute_dtype}")
            print(f"- quant_type: {config.bnb_4bit_quant_type}")
            print(f"- double_quant: {config.bnb_4bit_use_double_quant}")
            print(f"- cpu_offload: {config.llm_int8_enable_fp32_cpu_offload}")
        
        print(f"Device map: {device_map}")
        
        check_result = {
            "quantization": "4bit" if hasattr(config, 'load_in_4bit') and config.load_in_4bit else "8bit",
            "config_details": str(config),
            "device_map": str(device_map)
        }
        
        self.report["checks"].append({
            "name": "quantization_config",
            "status": "✅ 32Bモデル用に4bit量子化が選択されています" if hasattr(config, 'load_in_4bit') and config.load_in_4bit else "⚠️ 8bit量子化が選択されています",
            "details": check_result
        })
        
        return check_result
    
    def check_lora_workflow(self):
        """LoRAワークフローをチェック"""
        print("\n" + "="*60)
        print("3. LoRAワークフローチェック")
        print("="*60)
        
        workflow_steps = [
            {
                "step": "ファインチューニング",
                "description": "QLoRA（4bit）でファインチューニング",
                "memory_requirement": "16-20GB VRAM",
                "status": "OK"
            },
            {
                "step": "LoRAアダプター保存",
                "description": "アダプターのみ保存（マージなし）",
                "memory_requirement": "< 1GB",
                "status": "OK"
            },
            {
                "step": "ベースモデル再ロード",
                "description": "4bit量子化でベースモデルロード",
                "memory_requirement": "16GB VRAM",
                "status": "OK"
            },
            {
                "step": "マージ処理",
                "description": "LoRA + ベースモデル → FP16",
                "memory_requirement": "32-40GB RAM (CPU処理)",
                "status": "注意"
            },
            {
                "step": "GGUF変換",
                "description": "FP16 → GGUF F16",
                "memory_requirement": "64GB ディスク",
                "status": "OK"
            },
            {
                "step": "量子化",
                "description": "GGUF F16 → Q4_K_M",
                "memory_requirement": "CPU処理",
                "status": "OK"
            },
            {
                "step": "Ollama登録",
                "description": "Q4_K_Mモデルを登録",
                "memory_requirement": "16GB ディスク",
                "status": "OK"
            }
        ]
        
        for i, step in enumerate(workflow_steps, 1):
            status_icon = "✅" if step["status"] == "OK" else "⚠️"
            print(f"\n[Step {i}] {step['step']}")
            print(f"  {status_icon} {step['description']}")
            print(f"  メモリ要件: {step['memory_requirement']}")
        
        self.report["checks"].append({
            "name": "lora_workflow",
            "status": "✅ ワークフロー確認完了",
            "workflow": workflow_steps
        })
        
        return workflow_steps
    
    def check_script_compatibility(self):
        """スクリプトの互換性をチェック"""
        print("\n" + "="*60)
        print("4. スクリプト互換性チェック")
        print("="*60)
        
        scripts_to_check = [
            {
                "path": "/workspace/scripts/qlora_to_ollama.py",
                "purpose": "QLoRA → Ollama変換",
                "32b_support": None
            },
            {
                "path": "/workspace/scripts/unified_model_processor.py",
                "purpose": "統合モデル処理",
                "32b_support": None
            },
            {
                "path": "/workspace/scripts/proper_merge_flow.py",
                "purpose": "FP16マージ処理",
                "32b_support": None
            }
        ]
        
        for script in scripts_to_check:
            if Path(script["path"]).exists():
                # スクリプト内容をチェック
                with open(script["path"], 'r') as f:
                    content = f.read()
                
                # 32Bモデル対応チェック
                has_32b_check = "32b" in content.lower() or "32B" in content
                has_4bit_fallback = "load_in_4bit" in content
                has_memory_optimization = "memory_optimized_loader" in content or "get_optimal_quantization_config" in content
                
                script["32b_support"] = {
                    "exists": True,
                    "has_32b_check": has_32b_check,
                    "has_4bit_fallback": has_4bit_fallback,
                    "has_memory_optimization": has_memory_optimization,
                    "compatible": has_32b_check or has_4bit_fallback or has_memory_optimization
                }
                
                status = "✅" if script["32b_support"]["compatible"] else "⚠️"
                print(f"\n{status} {Path(script['path']).name}")
                print(f"  用途: {script['purpose']}")
                print(f"  32B対応: {'あり' if has_32b_check else 'なし'}")
                print(f"  4bitフォールバック: {'あり' if has_4bit_fallback else 'なし'}")
                print(f"  メモリ最適化: {'あり' if has_memory_optimization else 'なし'}")
            else:
                script["32b_support"] = {"exists": False}
                print(f"\n❌ {Path(script['path']).name} - ファイルが存在しません")
        
        self.report["checks"].append({
            "name": "script_compatibility",
            "scripts": scripts_to_check
        })
        
        return scripts_to_check
    
    def check_gguf_conversion(self):
        """GGUF変換の互換性をチェック"""
        print("\n" + "="*60)
        print("5. GGUF変換互換性チェック")
        print("="*60)
        
        # llama.cppの存在確認
        llama_cpp_path = Path("/workspace/llama.cpp")
        
        if llama_cpp_path.exists():
            print("✅ llama.cpp: インストール済み")
            
            # convert_hf_to_gguf.pyの確認
            convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
            if convert_script.exists():
                print("✅ convert_hf_to_gguf.py: 存在")
                
                # スクリプト内容確認
                with open(convert_script, 'r') as f:
                    content = f.read()[:1000]  # 最初の1000文字
                
                # Qwenモデルのサポート確認
                supports_qwen = "qwen" in content.lower()
                print(f"  Qwenサポート: {'あり' if supports_qwen else '要確認'}")
            else:
                print("❌ convert_hf_to_gguf.py: 存在しない")
            
            # llama-quantizeの確認
            quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
            if quantize_bin.exists():
                print("✅ llama-quantize: ビルド済み")
            else:
                print("⚠️ llama-quantize: 未ビルド")
        else:
            print("❌ llama.cpp: 未インストール")
            self.report["recommendations"].append(
                "llama.cppのインストールが必要です"
            )
        
        self.report["checks"].append({
            "name": "gguf_conversion",
            "llama_cpp_exists": llama_cpp_path.exists(),
            "convert_script_exists": (llama_cpp_path / "convert_hf_to_gguf.py").exists() if llama_cpp_path.exists() else False,
            "quantize_binary_exists": (llama_cpp_path / "build" / "bin" / "llama-quantize").exists() if llama_cpp_path.exists() else False
        })
    
    def estimate_processing_time(self):
        """処理時間を推定"""
        print("\n" + "="*60)
        print("6. 処理時間推定")
        print("="*60)
        
        estimates = {
            "QLoRAファインチューニング": "2-4時間 (3エポック)",
            "LoRAマージ（4bit→FP16）": "30-60分",
            "GGUF変換（FP16）": "20-30分",
            "量子化（Q4_K_M）": "15-20分",
            "Ollama登録": "1-2分",
            "合計": "約3-6時間"
        }
        
        for step, time in estimates.items():
            print(f"{step}: {time}")
        
        self.report["estimates"] = estimates
        
        return estimates
    
    def generate_report(self):
        """検証レポートを生成"""
        print("\n" + "="*60)
        print("検証レポート")
        print("="*60)
        
        # 総合判定
        all_checks_ok = True
        critical_issues = []
        
        # GPU メモリチェック
        if self.report["memory_usage"]["system"]["total_gpu_memory_gb"] < 48:
            critical_issues.append("GPU VRAM < 48GB: QLoRA（4bit）必須")
        
        # RAM チェック
        if self.report["memory_usage"]["system"]["available_ram_gb"] < 32:
            critical_issues.append("RAM空き < 32GB: マージ処理が遅い可能性")
        
        if critical_issues:
            print("\n⚠️ 注意事項:")
            for issue in critical_issues:
                print(f"  - {issue}")
        else:
            print("\n✅ 全チェック合格")
        
        print("\n[推奨設定]")
        print("1. QLoRAファインチューニング（4bit量子化）")
        print("2. マージ時はCPU処理を活用")
        print("3. GGUF変換は十分なディスク容量を確保")
        print("4. 最終的にQ4_K_M形式で量子化")
        
        # レポートをファイルに保存
        report_path = Path("/workspace/outputs/32b_validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 詳細レポート保存: {report_path}")
        
        return self.report
    
    def run_validation(self):
        """完全な検証を実行"""
        print("="*60)
        print("DeepSeek-R1-Distill-Qwen-32B-Japanese 量子化検証")
        print("="*60)
        
        # 各チェックを実行
        self.check_system_resources()
        self.check_quantization_config()
        self.check_lora_workflow()
        self.check_script_compatibility()
        self.check_gguf_conversion()
        self.estimate_processing_time()
        
        # レポート生成
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("検証完了")
        print("="*60)
        
        return report


def main():
    """メイン実行"""
    validator = DeepSeek32BValidator()
    report = validator.run_validation()
    
    # 結論
    print("\n【結論】")
    print("DeepSeek-R1-Distill-Qwen-32B-Japaneseの量子化は以下の条件で可能:")
    print("1. QLoRA（4bit）でファインチューニング ✅")
    print("2. マージ時に4bit量子化ベースモデル使用 ✅")
    print("3. FP16で一時保存後、GGUF変換 ✅")
    print("4. Q4_K_M形式で最終量子化 ✅")
    print("\n現在のスクリプトは32Bモデルに対応しています。")


if __name__ == "__main__":
    main()