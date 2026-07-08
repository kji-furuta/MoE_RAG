"""
MoE Refactoring Test Suite
リファクタリング後のMoEシステムのテスト
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.moe.constants import (
    ExpertType,
    DEFAULT_VOCAB_SIZE,
    SAFE_VOCAB_LIMIT,
    EXPERT_DISPLAY_NAMES,
    DEFAULT_EXPERT_SPECIALIZATION
)
from src.moe.base_config import MoEConfig, MoETrainingConfig
from src.moe.utils import (
    validate_input_ids,
    compute_load_balance_loss,
    get_device,
    count_parameters,
    format_size
)
from src.moe.exceptions import (
    MoEModelError,
    MoEConfigError,
    MoEDataError
)


class TestConstants(unittest.TestCase):
    """定数のテスト"""
    
    def test_expert_types(self):
        """エキスパートタイプの定義を確認"""
        self.assertEqual(len(list(ExpertType)), 8)
        self.assertEqual(ExpertType.STRUCTURAL_DESIGN.value, "structural_design")
        self.assertEqual(ExpertType.ROAD_DESIGN.value, "road_design")
    
    def test_expert_display_names(self):
        """エキスパート表示名の確認"""
        self.assertEqual(len(EXPERT_DISPLAY_NAMES), 8)
        self.assertEqual(EXPERT_DISPLAY_NAMES[0], "構造設計")
        self.assertEqual(EXPERT_DISPLAY_NAMES[1], "道路設計")
    
    def test_vocab_constants(self):
        """語彙サイズ定数の確認"""
        self.assertEqual(DEFAULT_VOCAB_SIZE, 32000)
        self.assertEqual(SAFE_VOCAB_LIMIT, 30000)
        self.assertLess(SAFE_VOCAB_LIMIT, DEFAULT_VOCAB_SIZE)


class TestConfig(unittest.TestCase):
    """設定クラスのテスト"""
    
    def test_moe_config_defaults(self):
        """MoEConfigのデフォルト値を確認"""
        config = MoEConfig()
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertTrue(config.domain_specific_routing)
    
    def test_moe_config_validation(self):
        """MoEConfigの検証機能をテスト"""
        # 正常な設定
        config = MoEConfig()
        config.validate()  # Should not raise
        
        # 異常な設定
        config = MoEConfig(hidden_size=-1)
        with self.assertRaises(AssertionError):
            config.validate()
        
        config = MoEConfig(num_experts_per_tok=10, num_experts=5)
        with self.assertRaises(AssertionError):
            config.validate()
    
    def test_training_config_defaults(self):
        """MoETrainingConfigのデフォルト値を確認"""
        config = MoETrainingConfig()
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.mixed_precision, "bf16")
    
    def test_training_config_validation(self):
        """MoETrainingConfigの検証機能をテスト"""
        # 正常な設定
        config = MoETrainingConfig()
        config.validate()  # Should not raise
        
        # 異常な設定
        config = MoETrainingConfig(learning_rate=-0.1)
        with self.assertRaises(AssertionError):
            config.validate()
        
        config = MoETrainingConfig(mixed_precision="invalid")
        with self.assertRaises(AssertionError):
            config.validate()


class TestUtils(unittest.TestCase):
    """ユーティリティ関数のテスト"""
    
    def test_validate_input_ids(self):
        """入力ID検証のテスト"""
        # 正常な入力
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = validate_input_ids(input_ids, vocab_size=10)
        self.assertTrue(torch.equal(result, input_ids))
        
        # 範囲外の入力
        input_ids = torch.tensor([[1, 2, 35000]])
        result = validate_input_ids(input_ids, vocab_size=32000)
        self.assertTrue(result.max() < 32000)
        
        # 無効な入力
        with self.assertRaises(MoEModelError):
            validate_input_ids(None)
        
        with self.assertRaises(MoEModelError):
            validate_input_ids("invalid")
    
    def test_compute_load_balance_loss(self):
        """ロードバランス損失計算のテスト"""
        # 均等な分布
        router_logits = torch.randn(4, 8)
        expert_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        loss = compute_load_balance_loss(router_logits, expert_indices, 8)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)
        
        # 不均等な分布
        expert_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = compute_load_balance_loss(router_logits, expert_indices, 8)
        self.assertGreater(loss.item(), 0)
    
    def test_format_size(self):
        """数値フォーマットのテスト"""
        self.assertEqual(format_size(100), "100.0")
        self.assertEqual(format_size(1500), "1.5K")
        self.assertEqual(format_size(2500000), "2.5M")
        self.assertEqual(format_size(3500000000), "3.5B")


class TestExceptions(unittest.TestCase):
    """例外クラスのテスト"""
    
    def test_exception_hierarchy(self):
        """例外階層の確認"""
        # 基底クラスの確認
        self.assertTrue(issubclass(MoEConfigError, Exception))
        self.assertTrue(issubclass(MoEModelError, Exception))
        self.assertTrue(issubclass(MoEDataError, Exception))
    
    def test_exception_raising(self):
        """例外の発生テスト"""
        with self.assertRaises(MoEConfigError):
            raise MoEConfigError("Config error")
        
        with self.assertRaises(MoEModelError):
            raise MoEModelError("Model error")
        
        with self.assertRaises(MoEDataError):
            raise MoEDataError("Data error")


class TestIntegration(unittest.TestCase):
    """統合テスト"""
    
    def test_config_with_constants(self):
        """設定と定数の統合テスト"""
        config = MoEConfig()
        
        # エキスパート特化度が定数から設定されることを確認
        self.assertIsNotNone(config.expert_specialization)
        self.assertEqual(
            config.expert_specialization[ExpertType.STRUCTURAL_DESIGN.value],
            DEFAULT_EXPERT_SPECIALIZATION[ExpertType.STRUCTURAL_DESIGN.value]
        )
    
    def test_safe_vocab_handling(self):
        """安全な語彙サイズ処理のテスト"""
        # 大きな値を入力
        input_ids = torch.tensor([[50000, 60000, 70000]])
        
        # サニタイズ後は安全な範囲内
        result = validate_input_ids(input_ids)
        self.assertTrue(result.max() <= SAFE_VOCAB_LIMIT)
        self.assertTrue(result.min() >= 0)


def run_tests():
    """テストスイートの実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 各テストクラスを追加
    suite.addTests(loader.loadTestsFromTestCase(TestConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)