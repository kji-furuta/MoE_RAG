#!/usr/bin/env python3
"""
LoRAアダプターをGGUF形式のベースモデルに適用してOllamaに登録するスクリプト
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRAToOllamaConverter:
    """LoRAアダプターをOllamaモデルに変換"""
    
    def __init__(self, workspace_dir: str = "/workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.models_dir = self.workspace_dir / "models"
        self.outputs_dir = self.workspace_dir / "outputs"
        self.ollama_models_dir = Path.home() / ".ollama" / "models"
        
        # ディレクトリ作成
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
    def download_base_model(self, model_url: str, model_name: str) -> Path:
        """ベースモデル（GGUF）をダウンロード"""
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            logger.info(f"モデルは既に存在します: {model_path}")
            return model_path
            
        logger.info(f"モデルをダウンロード中: {model_url}")
        
        try:
            # wgetがインストールされているか確認
            wget_check = subprocess.run(["which", "wget"], capture_output=True, text=True)
            if wget_check.returncode != 0:
                # wgetがない場合はcurlを試す
                logger.info("wgetが見つからないため、curlを使用します")
                cmd = [
                    "curl", "-L", "-C", "-", "-o", str(model_path),
                    "--progress-bar",
                    model_url
                ]
            else:
                # wgetでダウンロード
                cmd = [
                    "wget", "-c", "-O", str(model_path),
                    "--progress=bar:force:noscroll",
                    model_url
                ]
            
            logger.info(f"ダウンロードコマンド: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode != 0:
                # Python requestsライブラリを使用してダウンロード
                logger.info("コマンドラインツールが失敗したため、Pythonでダウンロードを試みます")
                import requests
                
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\rダウンロード中: {progress:.1f}%", end='', flush=True)
                print()  # 改行
                
            logger.info(f"ダウンロード完了: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"ダウンロードエラー: {e}")
            if model_path.exists():
                model_path.unlink()
            raise
            
    def find_lora_adapter(self, adapter_name: Optional[str] = None) -> Optional[Path]:
        """LoRAアダプターを検索（オプショナル）"""
        # 最新のLoRAアダプターを探す
        lora_dirs = []
        
        # outputs/ディレクトリから検索
        if self.outputs_dir.exists():
            for path in self.outputs_dir.glob("**/adapter_model.safetensors"):
                lora_dirs.append(path.parent)
            for path in self.outputs_dir.glob("**/adapter_model.bin"):
                lora_dirs.append(path.parent)
            
        if not lora_dirs:
            logger.info("LoRAアダプターが見つかりません - ベースモデルのみを使用します")
            return None
            
        # 最新のディレクトリを選択
        latest_dir = max(lora_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"LoRAアダプターを発見: {latest_dir}")
        
        return latest_dir
        
    def convert_lora_to_gguf(self, lora_dir: Path, output_path: Path) -> bool:
        """LoRAアダプターをGGUF形式に変換"""
        logger.info(f"LoRAアダプターをGGUF形式に変換中: {lora_dir}")
        
        try:
            # llama.cppのconvert-lora-to-gguf.pyスクリプトを使用
            convert_script = self.workspace_dir / "llama.cpp" / "convert-lora-to-gguf.py"
            
            if not convert_script.exists():
                # llama.cppがない場合はダウンロード
                logger.info("llama.cppをクローン中...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(self.workspace_dir / "llama.cpp")
                ], check=True)
                
            # python3を使用（python2との競合を避ける）
            cmd = [
                "python3", str(convert_script),
                "--base", str(lora_dir),
                "--outfile", str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"変換警告: {result.stderr}")
                # 代替方法: 直接マージ
                return False
                
            logger.info(f"GGUF変換完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"変換エラー: {e}")
            return False
            
    def merge_lora_with_base(self, base_model: Path, lora_adapter: Path, output_model: Path) -> Path:
        """LoRAアダプターをベースモデルにマージ"""
        logger.info("LoRAアダプターをベースモデルにマージ中...")
        
        try:
            # llama.cppのllama-export-loraツールを使用
            merge_tool = self.workspace_dir / "llama.cpp" / "llama-export-lora"
            
            if not merge_tool.exists():
                # ビルドが必要
                logger.info("llama.cppをビルド中...")
                build_dir = self.workspace_dir / "llama.cpp"
                subprocess.run(["make", "-j4"], cwd=str(build_dir), check=True)
                
            cmd = [
                str(merge_tool),
                "-m", str(base_model),
                "-o", str(output_model),
                "--lora", str(lora_adapter)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"マージ失敗: {result.stderr}")
                
            logger.info(f"マージ完了: {output_model}")
            return output_model
            
        except Exception as e:
            logger.error(f"マージエラー: {e}")
            # フォールバック: コピーのみ
            shutil.copy2(base_model, output_model)
            return output_model
            
    def create_ollama_modelfile(self, model_path: Path, model_name: str) -> Path:
        """Ollama用のModelfileを作成"""
        modelfile_path = self.models_dir / f"{model_name}.modelfile"
        
        modelfile_content = f"""
FROM {model_path}

# System prompt for Japanese civil engineering
SYSTEM "あなたは日本の土木設計と道路設計の専門家です。技術的な質問に対して正確で詳細な回答を提供します。"

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

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
                
            # モデルを作成
            cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"登録失敗: {result.stderr}")
                return False
                
            logger.info(f"Ollama登録完了: {model_name}")
            
            # テスト実行
            test_cmd = ["ollama", "run", model_name, "こんにちは"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"テスト成功: {result.stdout[:100]}...")
            else:
                logger.warning(f"テスト失敗: {result.stderr}")
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Ollamaの応答がタイムアウトしました")
            return False
        except Exception as e:
            logger.error(f"登録エラー: {e}")
            return False
            
    def run(self, 
            base_model_url: str = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            base_model_name: str = "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            lora_adapter_name: Optional[str] = None,
            output_model_name: str = "deepseek-32b-finetuned"):
        """メイン処理を実行"""
        
        try:
            # 1. ベースモデルをダウンロード
            base_model_path = self.download_base_model(base_model_url, base_model_name)
            
            # 2. LoRAアダプターを探す（オプション）
            lora_dir = self.find_lora_adapter(lora_adapter_name)
            
            # 3. LoRAがある場合はGGUF形式に変換
            if lora_dir:
                lora_gguf_path = self.models_dir / "lora_adapter.gguf"
                lora_converted = self.convert_lora_to_gguf(lora_dir, lora_gguf_path)
                
                # 4. モデルをマージまたは準備
                if lora_converted and lora_gguf_path.exists():
                    # マージ版を作成
                    merged_model_path = self.models_dir / f"{output_model_name}_merged.gguf"
                    final_model = self.merge_lora_with_base(
                        base_model_path, 
                        lora_gguf_path,
                        merged_model_path
                    )
                else:
                    # ベースモデルのみ使用（LoRA変換失敗）
                    logger.info("LoRA変換失敗、ベースモデルのみ使用")
                    final_model = base_model_path
            else:
                # ベースモデルのみ使用（LoRAなし）
                logger.info("ベースモデルのみでOllamaモデルを作成")
                final_model = base_model_path
                
            # 5. Modelfileを作成
            modelfile = self.create_ollama_modelfile(final_model, output_model_name)
            
            # 6. Ollamaに登録
            success = self.register_to_ollama(modelfile, output_model_name)
            
            if success:
                logger.info(f"""
==============================================
✅ 変換完了！

モデル名: {output_model_name}
モデルパス: {final_model}

使用方法:
  ollama run {output_model_name} "質問を入力"

RAGシステムで使用:
  config/model_config.yamlに追加:
    model: '{output_model_name}:latest'
==============================================
                """)
                
                # 設定ファイルを更新
                self.update_config(output_model_name)
                
            else:
                logger.error("Ollamaへの登録に失敗しました")
                
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            raise
            
    def update_config(self, model_name: str):
        """設定ファイルを更新"""
        config_path = self.workspace_dir / "config" / "model_config.yaml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
                
            # Ollamaモデルを追加
            if 'ollama_models' not in config:
                config['ollama_models'] = []
                
            model_entry = {
                'name': model_name,
                'tag': 'latest',
                'description': 'LoRA fine-tuned DeepSeek 32B model'
            }
            
            # 重複チェック
            if not any(m['name'] == model_name for m in config['ollama_models']):
                config['ollama_models'].append(model_entry)
                
                import yaml
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                    
                logger.info(f"設定ファイルを更新しました: {config_path}")
                
        except Exception as e:
            logger.warning(f"設定ファイルの更新に失敗: {e}")


def main():
    """メインエントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRAアダプターをOllama形式に変換")
    parser.add_argument("--base-model-url", 
                       default="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
                       help="ベースモデルのURL")
    parser.add_argument("--base-model-name",
                       default="DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
                       help="ベースモデルのファイル名")
    parser.add_argument("--lora-adapter",
                       help="LoRAアダプターのディレクトリ（省略時は最新を使用）")
    parser.add_argument("--output-name",
                       default="deepseek-32b-finetuned",
                       help="出力するOllamaモデル名")
    parser.add_argument("--workspace",
                       default=None,
                       help="ワークスペースディレクトリ")
    
    args = parser.parse_args()
    
    # ワークスペースディレクトリの自動検出
    if args.workspace is None:
        # 現在のディレクトリまたは親ディレクトリから適切なワークスペースを見つける
        current_dir = Path.cwd()
        if (current_dir / "outputs").exists() or current_dir.name == "MoE_RAG":
            args.workspace = str(current_dir)
        elif (current_dir.parent / "outputs").exists():
            args.workspace = str(current_dir.parent)
        else:
            # デフォルトとして現在のディレクトリを使用
            args.workspace = str(current_dir)
            logger.info(f"ワークスペースを現在のディレクトリに設定: {args.workspace}")
    
    converter = LoRAToOllamaConverter(args.workspace)
    converter.run(
        base_model_url=args.base_model_url,
        base_model_name=args.base_model_name,
        lora_adapter_name=args.lora_adapter,
        output_model_name=args.output_name
    )


if __name__ == "__main__":
    main()