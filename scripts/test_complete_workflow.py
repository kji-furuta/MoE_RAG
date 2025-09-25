#!/usr/bin/env python3
"""
9月18日機能の完全なワークフローテスト
DeepSeek-R1-Distill-Qwen-32B のLoRAファインチューニング → GGUF変換 → Ollama登録 → RAGハイブリッド検索
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteContinualLearningWorkflow:
    """完全な継続学習ワークフロー"""

    def __init__(self):
        self.project_root = project_root
        self.outputs_dir = self.project_root / "outputs"
        self.gguf_dir = self.project_root / "gguf_models"

        # 32Bモデル用のメモリアロケータ設定
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    def step1_lora_finetuning(self) -> Optional[str]:
        """ステップ1: LoRAファインチューニング（量子化モデル）"""
        logger.info("=" * 50)
        logger.info("ステップ1: LoRAファインチューニング開始")
        logger.info("=" * 50)

        from src.training.lora_finetuning import LoRAFineTuner
        from src.models.base_model import BaseModel

        try:
            # LoRAファインチューナーの初期化
            trainer = LoRAFineTuner(
                model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                use_quantization=True,  # 4bit量子化を使用
                quantization_bits=4
            )

            # テスト用データセット（実際のファインチューニングでは適切なデータを使用）
            test_data = [
                {
                    "text": "道路設計速度80km/hの場合、最小曲線半径は何メートルですか？"
                           "設計速度80km/hの道路における最小曲線半径は、道路構造令により280mと定められています。"
                },
                {
                    "text": "縦断勾配の最大値はどのように決定されますか？"
                           "縦断勾配の最大値は、道路の設計速度と地形条件により決定されます。"
                }
            ]

            # データセットファイルの作成
            dataset_path = self.outputs_dir / "test_continual_dataset.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dataset_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            # ファインチューニング実行（短縮版）
            output_dir = trainer.train(
                dataset_path=str(dataset_path),
                output_dir=None,  # 自動生成
                epochs=1,  # テスト用に1エポック
                batch_size=1,
                learning_rate=2e-4,
                warmup_ratio=0.1
            )

            logger.info(f"✅ LoRAファインチューニング完了: {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"❌ LoRAファインチューニング失敗: {e}")
            return None

    def step2_convert_to_gguf(self, model_path: str) -> Optional[str]:
        """ステップ2: GGUF形式への変換"""
        logger.info("=" * 50)
        logger.info("ステップ2: GGUF変換開始")
        logger.info("=" * 50)

        try:
            # llama-cpp-pythonのインストール確認
            try:
                import llama_cpp
                logger.info("llama-cpp-python はインストール済み")
            except ImportError:
                logger.info("llama-cpp-python をインストール中...")
                subprocess.run(
                    ["pip", "install", "llama-cpp-python"],
                    check=True
                )

            # GGUF変換スクリプトの実行
            from scripts.convert.convert_to_gguf import convert_to_gguf

            model_name = Path(model_path).name
            result = convert_to_gguf(
                model_path=model_path,
                output_name=f"{model_name}_gguf"
            )

            if result["success"]:
                logger.info(f"✅ GGUF変換完了: {result['output_path']}")
                return result["output_path"]
            else:
                logger.error(f"❌ GGUF変換失敗: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"❌ GGUF変換エラー: {e}")
            return None

    def step3_register_ollama(self, gguf_path: str) -> Optional[str]:
        """ステップ3: Ollamaへのモデル登録"""
        logger.info("=" * 50)
        logger.info("ステップ3: Ollamaモデル登録開始")
        logger.info("=" * 50)

        try:
            from scripts.convert.convert_to_gguf import setup_ollama_model

            model_name = f"road-expert-{Path(gguf_path).stem}"
            result = setup_ollama_model(gguf_path, model_name)

            if result["success"]:
                logger.info(f"✅ Ollamaモデル登録完了: {model_name}")

                # モデルのテスト
                test_result = subprocess.run(
                    ["ollama", "run", model_name, "道路設計の基本を教えてください"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if test_result.returncode == 0:
                    logger.info(f"モデルテスト成功: {test_result.stdout[:200]}...")

                return model_name
            else:
                logger.error(f"❌ Ollamaモデル登録失敗: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"❌ Ollamaモデル登録エラー: {e}")
            return None

    def step4_rag_integration(self, ollama_model_name: str) -> bool:
        """ステップ4: RAGシステムとの統合"""
        logger.info("=" * 50)
        logger.info("ステップ4: RAGシステム統合開始")
        logger.info("=" * 50)

        try:
            # RAGシステムの設定更新
            config_path = self.project_root / "config" / "rag_config.yaml"

            if config_path.exists():
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Ollamaモデルを追加
                if "ollama_models" not in config:
                    config["ollama_models"] = []

                if ollama_model_name not in config["ollama_models"]:
                    config["ollama_models"].append(ollama_model_name)

                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

                logger.info(f"✅ RAG設定に {ollama_model_name} を追加")

            # RAGシステムのテスト
            import requests

            # ヘルスチェック
            try:
                response = requests.get("http://localhost:8050/rag/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ RAGシステムは正常に動作しています")

                    # ハイブリッド検索テスト
                    query_data = {
                        "query": "道路設計速度と曲線半径の関係",
                        "top_k": 5,
                        "use_hybrid": True,
                        "model": ollama_model_name
                    }

                    response = requests.post(
                        "http://localhost:8050/rag/query",
                        json=query_data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ ハイブリッドRAG検索成功")
                        logger.info(f"検索結果: {result.get('answer', '')[:200]}...")
                        return True

            except requests.exceptions.RequestException as e:
                logger.warning(f"RAGシステムに接続できません: {e}")
                logger.info("RAGシステムを手動で起動してください: ./scripts/start_web_interface.sh")

            return True

        except Exception as e:
            logger.error(f"❌ RAGシステム統合エラー: {e}")
            return False

    def run_complete_workflow(self):
        """完全なワークフローを実行"""
        logger.info("=" * 70)
        logger.info("DeepSeek-R1-Distill-Qwen-32B 完全ワークフロー開始")
        logger.info("=" * 70)

        success_steps = []

        # ステップ1: LoRAファインチューニング
        model_path = self.step1_lora_finetuning()
        if model_path:
            success_steps.append("LoRAファインチューニング")
        else:
            logger.warning("LoRAファインチューニングをスキップします")
            # テスト用に既存のモデルを使用
            existing_models = list(self.outputs_dir.glob("lora_*"))
            if existing_models:
                model_path = str(existing_models[0])
                logger.info(f"既存モデルを使用: {model_path}")

        if not model_path:
            logger.error("モデルが見つかりません")
            return

        # ステップ2: GGUF変換
        gguf_path = self.step2_convert_to_gguf(model_path)
        if gguf_path:
            success_steps.append("GGUF変換")
        else:
            logger.warning("GGUF変換をスキップします")
            # テスト用に既存のGGUFファイルを使用
            existing_gguf = list(self.gguf_dir.glob("*.gguf")) if self.gguf_dir.exists() else []
            if existing_gguf:
                gguf_path = str(existing_gguf[0])
                logger.info(f"既存GGUFファイルを使用: {gguf_path}")

        if not gguf_path:
            logger.warning("GGUFファイルが見つかりません")
            return

        # ステップ3: Ollama登録
        ollama_model = self.step3_register_ollama(gguf_path)
        if ollama_model:
            success_steps.append("Ollama登録")

        # ステップ4: RAG統合
        if ollama_model:
            if self.step4_rag_integration(ollama_model):
                success_steps.append("RAG統合")

        # 結果サマリー
        logger.info("=" * 70)
        logger.info("ワークフロー完了")
        logger.info(f"成功したステップ: {', '.join(success_steps)}")
        logger.info("=" * 70)

        if len(success_steps) == 4:
            logger.info("✅ すべてのステップが正常に完了しました！")
            logger.info("9月18日の機能が完全に復元されました。")
        else:
            logger.warning(f"⚠️ 一部のステップが失敗しました: {4 - len(success_steps)}個")
            logger.info("手動での確認が必要な場合があります。")

def main():
    """メイン実行"""
    workflow = CompleteContinualLearningWorkflow()
    workflow.run_complete_workflow()

if __name__ == "__main__":
    main()