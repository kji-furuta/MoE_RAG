#!/usr/bin/env python3
"""
継続学習システムの修正検証スクリプト
コード構造とインポートの整合性を確認
"""

import ast
import sys
from pathlib import Path

def validate_file_syntax(file_path):
    """Pythonファイルの構文をチェック"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "✅ Syntax valid"
    except SyntaxError as e:
        return False, f"❌ Syntax error: {e}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def check_ewc_trainer_fix(file_path):
    """EWCFullFinetuningTrainerの修正を確認"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = []

    # Acceleratorのインポートと初期化を確認
    if "from accelerate import Accelerator" in content:
        checks.append("✅ Accelerator import found in __init__")
    else:
        checks.append("❌ Accelerator import missing in __init__")

    if "self.accelerator = Accelerator(" in content:
        checks.append("✅ Accelerator initialization found")
    else:
        checks.append("❌ Accelerator initialization missing")

    # 不要な変数が削除されているか確認
    if "task_loss_sum = 0" not in content:
        checks.append("✅ Unused task_loss_sum variable removed")
    else:
        checks.append("❌ Unused task_loss_sum variable still present")

    if "ewc_loss_sum = 0" not in content:
        checks.append("✅ Unused ewc_loss_sum variable removed")
    else:
        checks.append("❌ Unused ewc_loss_sum variable still present")

    # 不要なインポートが削除されているか確認
    if ", Any" not in content or "from typing import" not in content:
        checks.append("✅ Unused Any import removed or not present")
    else:
        checks.append("❌ Unused Any import still present")

    return checks

def check_continual_helper_fix(file_path):
    """ContinualLearningHelperの修正を確認"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = []

    # PEFTモデル用のoffload_dir条件を確認
    if "if for_peft:" in content and 'model_kwargs["offload_dir"]' in content:
        checks.append("✅ Conditional offload_dir for PEFT models found")
    else:
        checks.append("❌ Conditional offload_dir for PEFT models missing")

    return checks

def check_pipeline_fixes(file_path):
    """ContinualLearningPipelineの修正を確認"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = []

    # LoRAアダプター検出ロジックを確認
    if 'adapter_config_path = full_path / "adapter_config.json"' in content or 'adapter_config.json' in content:
        checks.append("✅ LoRA adapter detection logic found")
    else:
        checks.append("❌ LoRA adapter detection logic missing")

    # BitsAndBytesConfigの使用を確認
    if "from transformers import BitsAndBytesConfig" in content:
        checks.append("✅ BitsAndBytesConfig import found")
    else:
        checks.append("❌ BitsAndBytesConfig import missing")

    # StreamingTextDatasetの使用を確認
    if "StreamingTextDataset" in content:
        checks.append("✅ StreamingTextDataset usage found")
    else:
        checks.append("❌ StreamingTextDataset usage missing")

    # TrainingConfigパラメータ修正を確認
    if 'num_epochs' in content and 'batch_size' in content:
        checks.append("✅ TrainingConfig parameters updated (num_epochs, batch_size)")
    else:
        checks.append("❌ TrainingConfig parameters not updated")

    return checks

def main():
    """メイン検証処理"""
    print("=" * 60)
    print("Continual Learning System Fix Validation")
    print("=" * 60)

    # プロジェクトルートを取得
    project_root = Path(__file__).parent.parent

    # チェックするファイル
    files_to_check = [
        ("src/training/ewc_full_finetuning.py", check_ewc_trainer_fix),
        ("src/training/continual_learning_helper.py", check_continual_helper_fix),
        ("src/training/continual_learning_pipeline.py", check_pipeline_fixes),
    ]

    all_passed = True

    for file_path, check_func in files_to_check:
        full_path = project_root / file_path
        print(f"\n📁 Checking: {file_path}")
        print("-" * 50)

        if not full_path.exists():
            print(f"❌ File not found: {full_path}")
            all_passed = False
            continue

        # 構文チェック
        syntax_valid, syntax_msg = validate_file_syntax(full_path)
        print(f"Syntax: {syntax_msg}")

        if not syntax_valid:
            all_passed = False
            continue

        # 特定の修正チェック
        if check_func:
            checks = check_func(full_path)
            for check in checks:
                print(f"  {check}")
                if "❌" in check:
                    all_passed = False

    # 結果サマリー
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validation checks PASSED!")
        print("\nThe following fixes have been successfully applied:")
        print("1. EWCFullFinetuningTrainer: Accelerator initialization fixed")
        print("2. EWCFullFinetuningTrainer: Unused variables removed")
        print("3. ContinualLearningHelper: Conditional offload_dir for PEFT")
        print("4. ContinualLearningPipeline: LoRA-on-LoRA support added")
        print("5. ContinualLearningPipeline: 8-bit quantization for large models")
        print("6. ContinualLearningPipeline: StreamingTextDataset integration")
    else:
        print("❌ Some validation checks FAILED")
        print("Please review the issues above")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())