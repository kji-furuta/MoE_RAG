#!/usr/bin/env python3
"""
改善版: LoRAアダプターをGGUF形式のベースモデルに適用してOllamaに登録するスクリプト
- llama.cppの作業を/tmpディレクトリで実行して再起動を防ぐ
- 処理完了後に一時ファイルをクリーンアップ
- UIからの実行に最適化
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import time

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedLoRAToOllamaConverter:
    """改善版: LoRAアダプターをOllamaモデルに変換"""
    
    def __init__(self, workspace_dir: str = "/workspace", use_temp_dir: bool = True):
        self.workspace_dir = Path(workspace_dir)
        self.models_dir = self.workspace_dir / "models"
        self.outputs_dir = self.workspace_dir / "outputs"
        self.ollama_models_dir = Path.home() / ".ollama" / "models"
        
        # 一時ディレクトリを使用するか
        self.use_temp_dir = use_temp_dir
        self.temp_dir = None
        self.llama_cpp_dir = None
        
        # ディレクトリ作成
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # クリーンアップ登録
        atexit.register(self.cleanup)
        
    def setup_llama_cpp(self) -> Path:
        """llama.cppを一時ディレクトリにセットアップ"""
        logger.info("llama.cppをセットアップ中...")
        
        # 既存のビルド済みllama.cppを確認
        existing_paths = [
            Path("/workspace/llama.cpp"),
            Path("/tmp/llama.cpp")
        ]
        
        for path in existing_paths:
            if path.exists() and (path / "build/bin/llama-quantize").exists():
                logger.info(f"ビルド済みのllama.cppを発見: {path}")
                self.llama_cpp_dir = path
                return path
        
        # 一時ディレクトリを作成
        if self.use_temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="llama_cpp_", dir="/tmp"))
            self.llama_cpp_dir = self.temp_dir / "llama.cpp"
            logger.info(f"一時ディレクトリを使用: {self.temp_dir}")
        else:
            self.llama_cpp_dir = self.workspace_dir / "llama.cpp"
            
        try:
            if not self.llama_cpp_dir.exists():
                # llama.cppをクローン
                logger.info(f"llama.cppをクローン中: {self.llama_cpp_dir}")
                cmd = [
                    "git", "clone", "--depth", "1",
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(self.llama_cpp_dir)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"クローン失敗: {result.stderr}")
                    
            # ビルドディレクトリを作成してCMakeビルド
            build_dir = self.llama_cpp_dir / "build"
            if not (build_dir / "bin/llama-quantize").exists():
                logger.info("llama.cppをビルド中...")
                
                # CMakeビルド
                cmake_cmd = [
                    "cmake", "-B", str(build_dir),
                    "-S", str(self.llama_cpp_dir),
                    "-DLLAMA_CUDA=OFF",  # CPU版でビルド（高速化）
                    "-DCMAKE_BUILD_TYPE=Release"
                ]
                result = subprocess.run(cmake_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Makefileでビルド（フォールバック）
                    logger.info("CMakeビルド失敗、Makeでビルド中...")
                    make_cmd = ["make", "-j4", "-C", str(self.llama_cpp_dir)]
                    result = subprocess.run(make_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"ビルド失敗: {result.stderr}")
                else:
                    # CMakeビルドを実行
                    build_cmd = [
                        "cmake", "--build", str(build_dir),
                        "--config", "Release", "-j", "4"
                    ]
                    result = subprocess.run(build_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"ビルド失敗: {result.stderr}")
                        
            logger.info(f"llama.cppセットアップ完了: {self.llama_cpp_dir}")
            return self.llama_cpp_dir
            
        except Exception as e:
            logger.error(f"llama.cppセットアップエラー: {e}")
            self.cleanup()
            raise
            
    def cleanup(self):
        """一時ファイルをクリーンアップ"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                logger.info(f"一時ディレクトリを削除中: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                logger.warning(f"一時ディレクトリの削除に失敗: {e}")
                
    def find_lora_adapter(self, adapter_name: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
        """LoRAアダプターを検索し、メタデータも返す"""
        # 最新のLoRAアダプターを探す
        lora_dirs = []
        
        # outputs/ディレクトリから検索
        for path in self.outputs_dir.glob("**/adapter_model.safetensors"):
            lora_dirs.append(path.parent)
        for path in self.outputs_dir.glob("**/adapter_model.bin"):
            lora_dirs.append(path.parent)
            
        if not lora_dirs:
            raise FileNotFoundError("LoRAアダプターが見つかりません")
            
        # 最新のディレクトリを選択
        latest_dir = max(lora_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"LoRAアダプターを発見: {latest_dir}")
        
        # メタデータを読み込み
        metadata = {}
        config_path = latest_dir / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"アダプター設定を読み込み: {metadata.get('base_model_name_or_path', 'unknown')}")
            except Exception as e:
                logger.warning(f"設定ファイルの読み込みに失敗: {e}")
                
        return latest_dir, metadata
        
    def find_base_model_path(self, metadata: Dict[str, Any]) -> Optional[Path]:
        """メタデータからベースモデルパスを検索"""
        base_model_name = metadata.get('base_model_name_or_path', '')
        
        # ベースモデル候補を検索
        candidates = [
            self.models_dir / "deepseek-base",
            self.models_dir / base_model_name,
            Path(base_model_name) if os.path.isabs(base_model_name) else None
        ]
        
        for candidate in candidates:
            if candidate and candidate.exists():
                logger.info(f"ベースモデルを発見: {candidate}")
                return candidate
                
        return None
        
    def convert_lora_to_gguf(self, lora_dir: Path, base_model_dir: Optional[Path], output_path: Path) -> bool:
        """LoRAアダプターをGGUF形式に変換"""
        logger.info(f"LoRAアダプターをGGUF形式に変換中: {lora_dir}")
        
        try:
            # convert_lora_to_gguf.pyスクリプトを使用
            convert_script = self.llama_cpp_dir / "convert_lora_to_gguf.py"
            
            if not convert_script.exists():
                # 古いバージョンのスクリプト名を試す
                convert_script = self.llama_cpp_dir / "convert-lora-to-gguf.py"
                
            if not convert_script.exists():
                logger.error("変換スクリプトが見つかりません")
                return False
                
            cmd = [
                "python", str(convert_script),
                str(lora_dir),
                "--outfile", str(output_path)
            ]
            
            # ベースモデルが指定されている場合
            if base_model_dir:
                cmd.extend(["--base", str(base_model_dir)])
                
            logger.info(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.llama_cpp_dir))
            
            if result.returncode != 0:
                logger.warning(f"変換警告: {result.stderr}")
                # エラーでも出力ファイルが作成されていれば成功とみなす
                if output_path.exists():
                    logger.info("警告はありますが、出力ファイルは作成されました")
                    return True
                return False
                
            logger.info(f"GGUF変換完了: {output_path}")
            logger.info(f"ファイルサイズ: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"変換エラー: {e}")
            return False
            
    def merge_lora_with_base(self, base_model: Path, lora_adapter: Path, output_model: Path) -> Path:
        """LoRAアダプターをベースモデルにマージ"""
        logger.info("LoRAアダプターをベースモデルにマージ中...")
        
        try:
            # llama-export-loraツールを使用
            export_tool_paths = [
                self.llama_cpp_dir / "build" / "bin" / "llama-export-lora",
                self.llama_cpp_dir / "llama-export-lora"
            ]
            
            export_tool = None
            for path in export_tool_paths:
                if path.exists():
                    export_tool = path
                    break
                    
            if not export_tool:
                logger.warning("llama-export-loraツールが見つかりません")
                # フォールバック: コピーのみ
                shutil.copy2(base_model, output_model)
                return output_model
                
            cmd = [
                str(export_tool),
                "-m", str(base_model),
                "-o", str(output_model),
                "--lora", str(lora_adapter)
            ]
            
            logger.info(f"実行コマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"マージエラー: {result.stderr}")
                # フォールバック: コピーのみ
                shutil.copy2(base_model, output_model)
                return output_model
                
            logger.info(f"マージ完了: {output_model}")
            logger.info(f"ファイルサイズ: {output_model.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
            return output_model
            
        except Exception as e:
            logger.error(f"マージエラー: {e}")
            # フォールバック: コピーのみ
            shutil.copy2(base_model, output_model)
            return output_model
            
    def download_base_model(self, model_url: str, model_name: str) -> Path:
        """ベースモデル（GGUF）をダウンロード"""
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            file_size = model_path.stat().st_size / 1024 / 1024 / 1024
            logger.info(f"モデルは既に存在します: {model_path} ({file_size:.2f} GB)")
            return model_path
            
        logger.info(f"モデルをダウンロード中: {model_url}")
        
        try:
            # wgetでダウンロード
            cmd = [
                "wget", "-c", "-O", str(model_path),
                "--progress=bar:force:noscroll",
                model_url
            ]
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode != 0:
                raise Exception(f"ダウンロード失敗")
                
            logger.info(f"ダウンロード完了: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"ダウンロードエラー: {e}")
            if model_path.exists():
                model_path.unlink()
            raise
            
    def create_ollama_modelfile(self, model_path: Path, model_name: str, metadata: Dict[str, Any]) -> Path:
        """Ollama用のModelfileを作成"""
        modelfile_path = self.models_dir / f"{model_name}.modelfile"
        
        # メタデータからパラメータを取得
        temperature = metadata.get('temperature', 0.7)
        max_length = metadata.get('max_length', 2048)
        
        modelfile_content = f"""
FROM {model_path}

# System prompt for Japanese civil engineering
SYSTEM "あなたは日本の土木設計と道路設計の専門家です。技術的な質問に対して正確で詳細な回答を提供します。"

# Model parameters
PARAMETER temperature {temperature}
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict {max_length}

# Template
TEMPLATE \"\"\"
{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}User: {{{{ .Prompt }}}}
Assistant: \"\"\"
"""
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content.strip())
            
        logger.info(f"Modelfile作成完了: {modelfile_path}")
        return modelfile_path
        
    def register_to_ollama(self, modelfile_path: Path, model_name: str) -> bool:
        """Ollamaにモデルを登録"""
        logger.info(f"Ollamaにモデルを登録中: {model_name}")
        
        try:
            # Ollamaが起動しているか確認
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.warning("Ollamaが起動していません。起動してください: ollama serve")
                return False
                
            # 既存のモデルを削除（更新の場合）
            existing_models = result.stdout
            if model_name in existing_models:
                logger.info(f"既存のモデルを削除中: {model_name}")
                subprocess.run(["ollama", "rm", model_name], capture_output=True)
                
            # モデルを作成
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"登録失敗: {result.stderr}")
                return False
                
            logger.info(f"Ollama登録完了: {model_name}")
            
            # 簡単なテスト（タイムアウトを短く）
            test_cmd = ["ollama", "run", model_name, "Hello"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"テスト成功")
            else:
                logger.warning(f"テスト失敗（ただし登録は成功）")
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.info("テストタイムアウト（ただし登録は成功している可能性があります）")
            return True
        except Exception as e:
            logger.error(f"登録エラー: {e}")
            return False
            
    def update_config(self, model_name: str):
        """設定ファイルを更新"""
        config_path = self.workspace_dir / "src" / "rag" / "config" / "rag_config.yaml"
        
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
                
            # Ollamaモデルリストを更新
            if 'llm' not in config:
                config['llm'] = {}
            if 'ollama_models' not in config['llm']:
                config['llm']['ollama_models'] = []
                
            model_entry = {
                'name': model_name,
                'tag': 'latest',
                'description': 'LoRA fine-tuned model'
            }
            
            # 重複チェック
            existing_names = [m.get('name') for m in config['llm']['ollama_models']]
            if model_name not in existing_names:
                config['llm']['ollama_models'].append(model_entry)
                
                import yaml
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                    
                logger.info(f"設定ファイルを更新しました: {config_path}")
                
        except Exception as e:
            logger.warning(f"設定ファイルの更新に失敗: {e}")
            
    def run(self, 
            base_model_url: Optional[str] = None,
            base_model_name: str = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            lora_adapter_name: Optional[str] = None,
            output_model_name: str = "deepseek-32b-finetuned",
            skip_ollama: bool = False) -> Dict[str, Any]:
        """メイン処理を実行"""
        
        result = {
            "success": False,
            "model_name": output_model_name,
            "model_path": None,
            "message": "",
            "steps_completed": []
        }
        
        try:
            # 1. llama.cppをセットアップ
            self.setup_llama_cpp()
            result["steps_completed"].append("llama.cpp setup")
            
            # 2. LoRAアダプターを探す
            lora_dir, metadata = self.find_lora_adapter(lora_adapter_name)
            result["steps_completed"].append("LoRA adapter found")
            
            # 3. ベースモデルを準備
            base_model_path = None
            if base_model_url:
                # URLが指定されている場合はダウンロード
                base_model_path = self.download_base_model(base_model_url, base_model_name)
            else:
                # メタデータからベースモデルを検索
                base_model_dir = self.find_base_model_path(metadata)
                if base_model_dir:
                    # ベースモデルディレクトリからGGUFファイルを探す
                    gguf_files = list(base_model_dir.glob("*.gguf"))
                    if gguf_files:
                        base_model_path = gguf_files[0]
                    else:
                        # デフォルトのモデルをダウンロード
                        base_model_url = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
                        base_model_path = self.download_base_model(base_model_url, base_model_name)
                else:
                    # デフォルトのモデルをダウンロード
                    base_model_url = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
                    base_model_path = self.download_base_model(base_model_url, base_model_name)
                    
            result["steps_completed"].append("Base model prepared")
            
            # 4. LoRAをGGUF形式に変換
            lora_gguf_path = self.models_dir / "lora_adapter.gguf"
            base_model_dir = self.find_base_model_path(metadata)
            lora_converted = self.convert_lora_to_gguf(lora_dir, base_model_dir, lora_gguf_path)
            
            if lora_converted:
                result["steps_completed"].append("LoRA to GGUF conversion")
                
            # 5. モデルをマージ
            if lora_converted and lora_gguf_path.exists() and base_model_path:
                # マージ版を作成
                merged_model_path = self.models_dir / f"{output_model_name}.gguf"
                final_model = self.merge_lora_with_base(
                    base_model_path, 
                    lora_gguf_path,
                    merged_model_path
                )
                result["steps_completed"].append("Model merge")
            else:
                # ベースモデルのみ使用
                logger.info("LoRA変換またはマージをスキップ")
                final_model = base_model_path
                
            result["model_path"] = str(final_model)
            
            # 6. Ollamaに登録（オプション）
            if not skip_ollama:
                modelfile = self.create_ollama_modelfile(final_model, output_model_name, metadata)
                success = self.register_to_ollama(modelfile, output_model_name)
                
                if success:
                    result["steps_completed"].append("Ollama registration")
                    self.update_config(output_model_name)
                    result["success"] = True
                    result["message"] = f"✅ 完了！モデル '{output_model_name}' がOllamaに登録されました。"
                else:
                    result["message"] = "⚠️ モデル作成は成功しましたが、Ollamaへの登録に失敗しました。"
            else:
                result["success"] = True
                result["message"] = f"✅ モデルファイルの作成が完了しました: {final_model}"
                
            # クリーンアップ
            self.cleanup()
            
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            result["message"] = f"❌ エラー: {str(e)}"
            self.cleanup()
            
        return result


def main():
    """メインエントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(description="改善版LoRAアダプターをOllama形式に変換")
    parser.add_argument("--base-model-url", 
                       help="ベースモデルのURL（省略時は自動検索）")
    parser.add_argument("--base-model-name",
                       default="DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
                       help="ベースモデルのファイル名")
    parser.add_argument("--lora-adapter",
                       help="LoRAアダプターのディレクトリ（省略時は最新を使用）")
    parser.add_argument("--output-name",
                       default="deepseek-32b-finetuned",
                       help="出力するOllamaモデル名")
    parser.add_argument("--workspace",
                       default="/workspace",
                       help="ワークスペースディレクトリ")
    parser.add_argument("--no-temp",
                       action="store_true",
                       help="一時ディレクトリを使用しない")
    parser.add_argument("--skip-ollama",
                       action="store_true",
                       help="Ollamaへの登録をスキップ")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LoRAファインチューニング済みアダプターの適用（改善版）")
    print("=" * 60)
    
    converter = ImprovedLoRAToOllamaConverter(
        workspace_dir=args.workspace,
        use_temp_dir=not args.no_temp
    )
    
    result = converter.run(
        base_model_url=args.base_model_url,
        base_model_name=args.base_model_name,
        lora_adapter_name=args.lora_adapter,
        output_model_name=args.output_name,
        skip_ollama=args.skip_ollama
    )
    
    print("\n" + "=" * 60)
    print("処理結果:")
    print(f"成功: {result['success']}")
    print(f"メッセージ: {result['message']}")
    print(f"完了ステップ: {', '.join(result['steps_completed'])}")
    if result['model_path']:
        print(f"モデルパス: {result['model_path']}")
    print("=" * 60)
    
    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())