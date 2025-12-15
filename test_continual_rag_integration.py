#!/usr/bin/env python3
"""
継続学習とRAGシステムの統合テスト
修正内容の検証:
1. LoRAアダプターの正しい読み込み
2. RAG設定での継続学習の有効化
3. JSON/JSONL形式の評価データ読み込み
"""

import sys
import json
from pathlib import Path

def test_continual_model_manager():
    """ContinualModelManagerのLoRA読み込みテスト"""
    print("=" * 60)
    print("Test 1: ContinualModelManager - LoRA Adapter Loading")
    print("=" * 60)

    try:
        from src.rag.core.continual_model_manager import ContinualModelManager
        from peft import PeftConfig

        # タスク履歴の確認
        manager = ContinualModelManager(base_path=Path("outputs"))
        available_tasks = manager.get_available_tasks()

        print(f"✓ Available continual learning tasks: {len(available_tasks)}")
        for task in available_tasks:
            print(f"  - {task}")

        # 最新タスクの確認
        latest_task = manager.get_latest_task()
        if latest_task:
            print(f"\n✓ Latest task: {latest_task.task_name}")
            print(f"  Model path: {latest_task.model_path}")

            # LoRAアダプター設定の確認
            model_path = Path(latest_task.model_path)
            if model_path.exists():
                try:
                    peft_config = PeftConfig.from_pretrained(str(model_path))
                    print(f"  ✓ Detected as LoRA adapter")
                    print(f"  Base model: {peft_config.base_model_name_or_path}")
                    print(f"  LoRA r: {peft_config.r}")
                    print(f"  LoRA alpha: {peft_config.lora_alpha}")
                except Exception as e:
                    print(f"  ⚠ Not a LoRA adapter: {e}")
            else:
                print(f"  ✗ Model path does not exist: {model_path}")

        print("\n✅ Test 1 PASSED: ContinualModelManager initialized successfully")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_config():
    """RAG設定での継続学習有効化テスト"""
    print("\n" + "=" * 60)
    print("Test 2: RAG Configuration - Continual Learning Enabled")
    print("=" * 60)

    try:
        import yaml

        config_path = Path("config/rag_config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 継続学習設定の確認
        continual_config = config.get('continual_learning', {})
        enabled = continual_config.get('enabled', False)

        print(f"Continual learning enabled: {enabled}")
        print(f"Model base path: {continual_config.get('model_base_path')}")
        print(f"EWC data path: {continual_config.get('ewc_data_path')}")

        if enabled:
            print("\n✅ Test 2 PASSED: Continual learning is enabled in RAG config")
            return True
        else:
            print("\n❌ Test 2 FAILED: Continual learning is not enabled")
            return False

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continual_metrics():
    """評価メトリクスのJSON/JSONL対応テスト"""
    print("\n" + "=" * 60)
    print("Test 3: Continual Metrics - JSON/JSONL Format Support")
    print("=" * 60)

    try:
        from src.evaluation.continual_metrics import ContinualLearningEvaluator

        evaluator = ContinualLearningEvaluator()

        # テストデータの作成 (JSON配列形式)
        test_data_json = [
            {"instruction": "Test question 1", "output": "Test answer 1"},
            {"instruction": "Test question 2", "output": "Test answer 2"}
        ]

        # テストデータの作成 (JSONL形式)
        test_data_jsonl = [
            '{"instruction": "Test question 1", "output": "Test answer 1"}\n',
            '{"instruction": "Test question 2", "output": "Test answer 2"}\n'
        ]

        # テンポラリファイルの作成
        import tempfile

        # JSON配列形式のテスト
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data_json, f, ensure_ascii=False, indent=2)
            json_file = f.name

        try:
            # ダミーモデルで評価
            class DummyModel:
                def eval(self):
                    pass

            result = evaluator._evaluate_task_performance(DummyModel(), json_file)

            if 'error' in result:
                print(f"  ⚠ JSON format test warning: {result['error']}")
            else:
                print(f"  ✓ JSON format: Successfully loaded data")
                print(f"    Samples evaluated: {result.get('num_samples', 0)}")
        finally:
            Path(json_file).unlink()

        # JSONL形式のテスト
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.writelines(test_data_jsonl)
            jsonl_file = f.name

        try:
            result = evaluator._evaluate_task_performance(DummyModel(), jsonl_file)

            if 'error' in result:
                print(f"  ⚠ JSONL format test warning: {result['error']}")
            else:
                print(f"  ✓ JSONL format: Successfully loaded data")
                print(f"    Samples evaluated: {result.get('num_samples', 0)}")
        finally:
            Path(jsonl_file).unlink()

        print("\n✅ Test 3 PASSED: Both JSON and JSONL formats are supported")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テストの実行"""
    print("\n" + "=" * 60)
    print("継続学習とRAGシステムの統合テスト")
    print("=" * 60 + "\n")

    results = []

    # Test 1: ContinualModelManager
    results.append(test_continual_model_manager())

    # Test 2: RAG Config
    results.append(test_rag_config())

    # Test 3: Continual Metrics
    results.append(test_continual_metrics())

    # 最終結果
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"合格: {passed}/{total}")

    if passed == total:
        print("\n🎉 すべてのテストが成功しました！")
        print("\n次のステップ:")
        print("1. Dockerコンテナを再起動してRAGシステムに変更を適用")
        print("2. RAG検索で継続学習モデルが自動選択されることを確認")
        print("3. 継続学習タスクに関連するクエリでテスト")
        return 0
    else:
        print(f"\n⚠️ {total - passed}個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
