"""
GGUF変換とOllama統合モジュール
継続学習後のモデルを自動的にGGUF形式に変換し、Ollamaに登録
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class GGUFIntegration:
    """GGUF変換とOllama統合"""

    def __init__(self):
        self.project_root = Path(os.getcwd())
        self.gguf_dir = self.project_root / "gguf_models"
        self.gguf_dir.mkdir(parents=True, exist_ok=True)

    def check_llama_cpp(self) -> bool:
        """llama-cpp-pythonがインストールされているか確認"""
        try:
            result = subprocess.run(
                ["python3", "-c", "import llama_cpp"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def install_llama_cpp(self) -> bool:
        """llama-cpp-pythonをインストール"""
        try:
            logger.info("llama-cpp-pythonをインストール中...")
            result = subprocess.run(
                ["pip", "install", "--no-cache-dir", "llama-cpp-python"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logger.info("llama-cpp-pythonのインストールが完了")
                return True
            else:
                logger.error(f"インストール失敗: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"インストールエラー: {e}")
            return False

    def convert_to_gguf(
        self,
        model_path: str,
        output_name: Optional[str] = None,
        quantization: str = "q4_k_m"
    ) -> Dict[str, Any]:
        """
        モデルをGGUF形式に変換

        Args:
            model_path: 変換するモデルのパス
            output_name: 出力ファイル名（省略時は自動生成）
            quantization: 量子化タイプ (q4_k_m, q8_0, f16など)

        Returns:
            変換結果の辞書
        """
        try:
            # llama-cpp-pythonの確認とインストール
            if not self.check_llama_cpp():
                if not self.install_llama_cpp():
                    return {
                        "success": False,
                        "error": "llama-cpp-pythonのインストールに失敗しました"
                    }

            # 出力ファイル名の生成
            if not output_name:
                model_name = Path(model_path).name
                output_name = f"{model_name}_{quantization}"

            output_path = self.gguf_dir / f"{output_name}.gguf"

            logger.info(f"GGUF変換開始: {model_path} → {output_path}")

            # 変換コマンドの実行
            cmd = [
                "python3", "-m", "llama_cpp.convert",
                str(model_path),
                "--outfile", str(output_path),
                "--outtype", quantization
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1時間のタイムアウト
            )

            if result.returncode == 0:
                logger.info(f"✅ GGUF変換完了: {output_path}")

                # ファイルサイズの確認
                file_size = output_path.stat().st_size / (1024 ** 3)  # GB
                logger.info(f"GGUFファイルサイズ: {file_size:.2f} GB")

                return {
                    "success": True,
                    "output_path": str(output_path),
                    "file_size_gb": file_size,
                    "quantization": quantization
                }
            else:
                logger.error(f"変換失敗: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr[:500]  # エラーメッセージの最初の500文字
                }

        except subprocess.TimeoutExpired:
            logger.error("GGUF変換がタイムアウトしました")
            return {"success": False, "error": "変換タイムアウト"}
        except Exception as e:
            logger.error(f"GGUF変換エラー: {e}")
            return {"success": False, "error": str(e)}

    def create_ollama_modelfile(
        self,
        gguf_path: str,
        model_info: Dict[str, Any]
    ) -> str:
        """Ollama用のModelfileを作成"""

        # モデル情報からシステムプロンプトを生成
        system_prompt = model_info.get(
            "system_prompt",
            "あなたは道路工学の専門家です。質問に対して正確で分かりやすい回答を提供してください。"
        )

        modelfile = f"""FROM {gguf_path}

# パラメータ設定
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

# 停止トークン
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
PARAMETER stop "質問:"
PARAMETER stop "回答:"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

# テンプレート
TEMPLATE """
{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

# システムプロンプト
SYSTEM """{system_prompt}"""
"""
        return modelfile

    def register_ollama_model(
        self,
        gguf_path: str,
        model_name: str,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        OllamaにGGUFモデルを登録

        Args:
            gguf_path: GGUFファイルのパス
            model_name: Ollamaでのモデル名
            model_info: モデル情報（メタデータ）

        Returns:
            登録結果の辞書
        """
        try:
            # Ollamaが実行中か確認
            check_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if check_result.returncode != 0:
                logger.warning("Ollamaサービスが起動していません")
                # Ollamaの起動を試みる
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                import time
                time.sleep(3)  # 起動を待つ

            # Modelfileの作成
            if model_info is None:
                model_info = {}

            modelfile_content = self.create_ollama_modelfile(gguf_path, model_info)
            modelfile_path = self.gguf_dir / f"{model_name}.Modelfile"

            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)

            logger.info(f"Ollamaモデル登録開始: {model_name}")

            # モデルの作成
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分のタイムアウト
            )

            if result.returncode == 0:
                logger.info(f"✅ Ollamaモデル登録成功: {model_name}")

                # モデルのテスト
                test_result = subprocess.run(
                    ["ollama", "run", model_name, "こんにちは"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if test_result.returncode == 0:
                    logger.info(f"モデルテスト成功: {model_name}")

                return {
                    "success": True,
                    "model_name": model_name,
                    "message": f"Ollamaモデル '{model_name}' が正常に登録されました"
                }
            else:
                logger.error(f"登録失敗: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr[:500]
                }

        except subprocess.TimeoutExpired:
            logger.error("Ollamaモデル登録がタイムアウトしました")
            return {"success": False, "error": "登録タイムアウト"}
        except Exception as e:
            logger.error(f"Ollamaモデル登録エラー: {e}")
            return {"success": False, "error": str(e)}

    def update_rag_config(self, ollama_model_name: str) -> bool:
        """RAG設定にOllamaモデルを追加"""
        try:
            config_path = self.project_root / "config" / "rag_config.yaml"

            if config_path.exists():
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # Ollamaモデルセクションの追加
                if "ollama" not in config:
                    config["ollama"] = {
                        "enabled": True,
                        "models": []
                    }

                if ollama_model_name not in config["ollama"]["models"]:
                    config["ollama"]["models"].append(ollama_model_name)

                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

                logger.info(f"✅ RAG設定を更新: {ollama_model_name}を追加")
                return True

        except Exception as e:
            logger.error(f"RAG設定更新エラー: {e}")

        return False

    def process_continual_learning_model(
        self,
        model_path: str,
        task_name: str,
        quantization: str = "q4_k_m"
    ) -> Dict[str, Any]:
        """
        継続学習モデルの完全な処理パイプライン

        Args:
            model_path: 継続学習で生成されたモデルのパス
            task_name: タスク名
            quantization: 量子化タイプ

        Returns:
            処理結果の辞書
        """
        results = {
            "model_path": model_path,
            "task_name": task_name,
            "steps": {}
        }

        # ステップ1: GGUF変換
        logger.info(f"継続学習モデルの処理開始: {task_name}")

        gguf_result = self.convert_to_gguf(
            model_path=model_path,
            output_name=f"continual_{task_name}",
            quantization=quantization
        )

        results["steps"]["gguf_conversion"] = gguf_result

        if not gguf_result["success"]:
            logger.error("GGUF変換に失敗しました")
            return results

        # ステップ2: Ollama登録
        ollama_model_name = f"continual-{task_name}-{quantization}"
        ollama_result = self.register_ollama_model(
            gguf_path=gguf_result["output_path"],
            model_name=ollama_model_name,
            model_info={
                "task_name": task_name,
                "base_model": model_path,
                "quantization": quantization,
                "system_prompt": f"継続学習タスク '{task_name}' で訓練されたモデルです。"
            }
        )

        results["steps"]["ollama_registration"] = ollama_result

        if ollama_result["success"]:
            # ステップ3: RAG設定更新
            rag_updated = self.update_rag_config(ollama_model_name)
            results["steps"]["rag_config_update"] = {
                "success": rag_updated,
                "model_name": ollama_model_name
            }

            # 処理結果の保存
            result_file = self.gguf_dir / f"continual_{task_name}_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 継続学習モデルの処理完了: {result_file}")

        return results


# グローバルインスタンス
gguf_integration = GGUFIntegration()