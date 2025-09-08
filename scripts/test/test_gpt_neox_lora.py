#!/usr/bin/env python3
"""
GPT-NeoX-20B LoRA統合テストスクリプト
ワークフローB（動的適用）のテスト
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import time
import tempfile

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_gpt_neox_lora_dynamic_application():
    """GPT-NeoX-20B LoRA動的適用をテスト"""
    
    logger.info("=" * 60)
    logger.info("GPT-NeoX-20B LoRA動的適用テスト（ワークフローB）")
    logger.info("=" * 60)
    
    # パスの設定
    workspace_dir = Path("/workspace")
    models_dir = workspace_dir / "models"
    outputs_dir = workspace_dir / "outputs"
    
    # GPT-NeoX-20B関連ファイルを探す
    base_model_path = models_dir / "gpt-neox-20b.Q4_K_M.gguf"
    lora_adapter_path = outputs_dir / "lora_20250906_104924"  # GPT-NeoX用のLoRAアダプタ
    output_model_path = models_dir / "gpt-neox-20b-lora-fixed.gguf"
    
    # ファイルの存在確認
    logger.info("\n1. ファイルの存在確認")
    
    if not base_model_path.exists():
        logger.error(f"ベースモデルが見つかりません: {base_model_path}")
        return False
    logger.info(f"✓ ベースモデル: {base_model_path}")
    
    if not lora_adapter_path.exists():
        logger.error(f"LoRAアダプタが見つかりません: {lora_adapter_path}")
        logger.info("利用可能なLoRAアダプタを検索中...")
        
        # 代替のLoRAアダプタを探す
        available_loras = list(outputs_dir.glob("lora_*"))
        for lora_dir in available_loras:
            config_path = lora_dir / "adapter_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "gpt-neox" in config.get("base_model_name_or_path", "").lower():
                        lora_adapter_path = lora_dir
                        logger.info(f"GPT-NeoX用LoRAアダプタを発見: {lora_adapter_path}")
                        break
        
        if not lora_adapter_path.exists():
            logger.error("GPT-NeoX用のLoRAアダプタが見つかりません")
            return False
    
    logger.info(f"✓ LoRAアダプタ: {lora_adapter_path}")
    
    # adapter_config.jsonの内容を確認
    adapter_config_path = lora_adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"  - ベースモデル: {config.get('base_model_name_or_path', 'N/A')}")
            logger.info(f"  - target_modules: {config.get('target_modules', [])}")
            logger.info(f"  - LoRA rank: {config.get('r', 'N/A')}")
    
    # 変換スクリプトを実行
    logger.info("\n2. GPT-NeoX専用変換スクリプトを実行")
    
    script_path = Path(__file__).parent / "apply_lora_gpt_neox.py"
    
    cmd = [
        "python", str(script_path),
        "--lora-path", str(lora_adapter_path),
        "--base-model", str(base_model_path),
        "--output", str(output_model_path),
        "--verbose"
    ]
    
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ 変換成功")
            logger.info(f"出力: {result.stdout}")
        else:
            logger.error(f"❌ 変換失敗")
            logger.error(f"エラー: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("タイムアウト: 変換処理が5分以内に完了しませんでした")
        return False
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        return False
    
    # 出力ファイルの確認
    logger.info("\n3. 出力ファイルの確認")
    
    if output_model_path.exists():
        file_size = output_model_path.stat().st_size / (1024 ** 3)  # GB
        logger.info(f"✓ 出力モデル生成成功: {output_model_path}")
        logger.info(f"  ファイルサイズ: {file_size:.2f} GB")
        
        # Ollamaでテスト（オプション）
        logger.info("\n4. Ollamaへの登録（オプション）")
        
        # Modelfileを作成
        modelfile_content = f"""FROM {output_model_path}

# GPT-NeoX-20B with LoRA adapter (Fixed)
SYSTEM "You are GPT-NeoX-20B with Japanese civil engineering fine-tuning. Provide accurate and detailed technical responses."

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

TEMPLATE \"\"\"
{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}User: {{{{ .Prompt }}}}
Assistant: \"\"\"
"""
        
        modelfile_path = models_dir / "Modelfile_gpt_neox_20b_lora_fixed"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile作成: {modelfile_path}")
        
        # Ollamaに登録
        try:
            cmd = ["ollama", "create", "gpt-neox-20b-lora-fixed", "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Ollamaへの登録成功")
                
                # テスト実行
                logger.info("\n5. 動作テスト")
                test_prompt = "道路設計における最小曲線半径について説明してください。"
                cmd = ["ollama", "run", "gpt-neox-20b-lora-fixed", test_prompt]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("テスト応答:")
                    logger.info(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                else:
                    logger.warning(f"テスト実行失敗: {result.stderr}")
                    
        except Exception as e:
            logger.warning(f"Ollama登録/テストはスキップ: {e}")
        
        return True
    else:
        logger.error(f"❌ 出力モデルが生成されませんでした: {output_model_path}")
        return False


def check_dependencies():
    """依存関係をチェック"""
    logger.info("依存関係のチェック...")
    
    required_packages = ["torch", "safetensors", "numpy"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package}")
    
    if missing:
        logger.error(f"不足パッケージをインストールしてください: pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """メイン処理"""
    logger.info("GPT-NeoX-20B LoRA動的適用テストを開始\n")
    
    # 依存関係チェック
    if not check_dependencies():
        sys.exit(1)
    
    # テスト実行
    success = test_gpt_neox_lora_dynamic_application()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 テスト完了: GPT-NeoX-20B LoRA動的適用が正常に動作しました")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ テスト失敗: 問題を確認してください")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()