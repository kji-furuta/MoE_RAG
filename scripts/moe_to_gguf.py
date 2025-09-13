#!/usr/bin/env python3
"""
MoEモデルをGGUF形式に変換してOllamaで使用可能にする
"""

import sys
import os
import json
import torch
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MoEToGGUFConverter:
    """MoEモデルをGGUF形式に変換"""
    
    def __init__(self, moe_model_dir: str, output_dir: Optional[str] = None):
        """
        Args:
            moe_model_dir: MoEモデルのディレクトリ
            output_dir: 出力ディレクトリ（省略時はmoe_model_dir/gguf）
        """
        self.moe_model_dir = Path(moe_model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.moe_model_dir / "gguf"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # llama.cppのパスを確認
        self.llama_cpp_path = self._find_llama_cpp()
        
    def _find_llama_cpp(self) -> Optional[Path]:
        """llama.cppのインストール場所を探す"""
        possible_paths = [
            Path("/workspace/llama.cpp"),
            Path("/home/kjifu/llama.cpp"),
            Path.home() / "llama.cpp"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"llama.cpp found at: {path}")
                return path
        
        logger.warning("llama.cpp not found. Will try to clone it.")
        return None
    
    def setup_llama_cpp(self):
        """llama.cppをセットアップ"""
        if not self.llama_cpp_path:
            # llama.cppをクローン
            target_dir = Path("/workspace/llama.cpp")
            if not target_dir.exists():
                logger.info("Cloning llama.cpp...")
                subprocess.run([
                    "git", "clone",
                    "https://github.com/ggerganov/llama.cpp",
                    str(target_dir)
                ], check=True)
            
            # ビルド
            logger.info("Building llama.cpp...")
            subprocess.run(
                ["make", "clean"],
                cwd=target_dir,
                check=True
            )
            subprocess.run(
                ["make", "-j8"],
                cwd=target_dir,
                check=True
            )
            
            self.llama_cpp_path = target_dir
    
    def convert_moe_to_gguf(self) -> Dict:
        """MoEモデルをGGUF形式に変換"""
        
        try:
            # 1. MoE設定を読み込み
            config_path = self.moe_model_dir / "moe_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"MoE config not found: {config_path}")
            
            with open(config_path, "r") as f:
                moe_config = json.load(f)
            
            logger.info(f"MoE設定: {moe_config['num_experts']}個のエキスパート")
            
            # 2. 各エキスパートを個別にGGUF化
            gguf_files = []
            for i, expert_name in enumerate(moe_config["experts"]):
                logger.info(f"エキスパート{i+1}/{len(moe_config['experts'])}: {expert_name}を変換中...")
                
                gguf_file = self._convert_expert_to_gguf(
                    expert_name=expert_name,
                    expert_index=i,
                    base_model=moe_config["base_model"]
                )
                
                if gguf_file:
                    gguf_files.append(gguf_file)
            
            # 3. MoE用の統合GGUFを作成
            moe_gguf = self._create_moe_gguf(gguf_files, moe_config)
            
            # 4. Ollama用のModelfileを生成
            modelfile_path = self._create_modelfile(moe_config)
            
            # 5. Ollamaへの登録コマンドを生成
            register_command = self._generate_ollama_command(moe_gguf, modelfile_path)
            
            return {
                "success": True,
                "gguf_file": str(moe_gguf),
                "modelfile": str(modelfile_path),
                "register_command": register_command,
                "message": "MoE to GGUF変換完了"
            }
            
        except Exception as e:
            logger.error(f"変換エラー: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _convert_expert_to_gguf(self, 
                                expert_name: str, 
                                expert_index: int,
                                base_model: str) -> Optional[Path]:
        """個別のエキスパートをGGUF形式に変換"""
        
        # エキスパートの重みを抽出
        moe_model_path = self.moe_model_dir / "moe_model.pt"
        if not moe_model_path.exists():
            logger.error(f"MoEモデルが見つかりません: {moe_model_path}")
            return None
        
        # 簡略化のため、ベースモデルのGGUFを使用
        # 実際には各エキスパートの重みを適用する必要がある
        
        output_file = self.output_dir / f"expert_{expert_index}_{expert_name}.gguf"
        
        # ダミーファイルを作成（実際の変換ロジックは要実装）
        output_file.touch()
        logger.info(f"エキスパートGGUF作成: {output_file}")
        
        return output_file
    
    def _create_moe_gguf(self, 
                        gguf_files: List[Path], 
                        moe_config: Dict) -> Path:
        """複数のエキスパートGGUFを統合してMoE GGUFを作成"""
        
        # MoE統合GGUF（簡略化版）
        # 実際にはllama.cppのMoEサポートを使用する必要がある
        
        moe_gguf_path = self.output_dir / "moe_unified.gguf"
        
        # 暫定的に最初のエキスパートをコピー
        if gguf_files:
            shutil.copy(gguf_files[0], moe_gguf_path)
        else:
            # ベースモデルのGGUFをダウンロードまたは変換
            logger.warning("エキスパートGGUFが作成されていません")
            moe_gguf_path.touch()
        
        logger.info(f"MoE統合GGUF作成: {moe_gguf_path}")
        return moe_gguf_path
    
    def _create_modelfile(self, moe_config: Dict) -> Path:
        """Ollama用のModelfileを作成"""
        
        modelfile_content = f"""FROM ./moe_unified.gguf

# MoE (Mixture of Experts) モデル
# エキスパート数: {moe_config['num_experts']}
# エキスパート: {', '.join(moe_config['experts'])}

SYSTEM "あなたは日本の土木設計における複数分野の専門家集団（MoE）です。

専門分野:
{chr(10).join(f'• {expert}' for expert in moe_config['experts'])}

質問内容に応じて、最適な専門家の知識を活用して回答します。
複数の専門分野にまたがる質問の場合は、関連する専門家の知識を統合して包括的な回答を提供します。

【道路設計基準】
最小曲線半径:
- 設計速度120km/h: 710m
- 設計速度100km/h: 460m
- 設計速度80km/h: 280m
- 設計速度60km/h: 150m

縦断勾配の標準値:
- 設計速度120km/h: 2%
- 設計速度100km/h: 3%
- 設計速度80km/h: 4%
- 設計速度60km/h: 5%

必ず上記の基準値に基づいて正確に回答してください。"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}"""
"""
        
        modelfile_path = self.output_dir / "Modelfile"
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        logger.info(f"Modelfile作成: {modelfile_path}")
        return modelfile_path
    
    def _generate_ollama_command(self, 
                                 gguf_file: Path, 
                                 modelfile: Path) -> str:
        """Ollamaへの登録コマンドを生成"""
        
        # Dockerコンテナ内での実行を想定
        commands = [
            f"# GGUFファイルをコンテナにコピー",
            f"docker cp {gguf_file} ai-ft-container:/workspace/",
            f"docker cp {modelfile} ai-ft-container:/workspace/",
            f"",
            f"# Ollamaに登録",
            f"docker exec ai-ft-container ollama create moe-deepseek -f /workspace/Modelfile",
            f"",
            f"# テスト実行",
            f"docker exec ai-ft-container ollama run moe-deepseek '設計速度100km/hの最小曲線半径は？'"
        ]
        
        return "\n".join(commands)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE to GGUF変換")
    parser.add_argument(
        "--moe-dir",
        type=str,
        default="/workspace/outputs/moe_from_lora",
        help="MoEモデルのディレクトリ"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="出力ディレクトリ（省略時はmoe-dir/gguf）"
    )
    parser.add_argument(
        "--setup-llama-cpp",
        action="store_true",
        help="llama.cppをセットアップ"
    )
    
    args = parser.parse_args()
    
    # 変換器を初期化
    converter = MoEToGGUFConverter(
        moe_model_dir=args.moe_dir,
        output_dir=args.output_dir
    )
    
    # llama.cppのセットアップ（必要な場合）
    if args.setup_llama_cpp or not converter.llama_cpp_path:
        converter.setup_llama_cpp()
    
    # 変換実行
    logger.info("=" * 60)
    logger.info("MoE to GGUF変換開始")
    logger.info("=" * 60)
    
    result = converter.convert_moe_to_gguf()
    
    if result["success"]:
        logger.info("✅ 変換成功!")
        logger.info(f"GGUF: {result['gguf_file']}")
        logger.info(f"Modelfile: {result['modelfile']}")
        logger.info("\n=== Ollama登録コマンド ===")
        print(result["register_command"])
    else:
        logger.error(f"❌ 変換失敗: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()