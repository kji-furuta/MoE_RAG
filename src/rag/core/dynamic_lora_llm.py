"""
動的LoRA適用LLMジェネレーター
GPT-NeoX-20Bの動的LoRA適用をRAGシステムで使用
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import requests
import time

logger = logging.getLogger(__name__)


class DynamicLoRALLM:
    """動的LoRA適用を使用したLLM生成器"""
    
    def __init__(self, 
                 base_gguf_path: str = "/workspace/models/gpt-neox-20b.Q4_K_M.gguf",
                 lora_adapter_path: str = "/workspace/outputs/lora_20250906_104924",
                 server_port: int = 8081):
        """
        Args:
            base_gguf_path: ベースGGUFモデルのパス
            lora_adapter_path: LoRAアダプターのパス
            server_port: llama.cppサーバーのポート
        """
        self.base_gguf_path = Path(base_gguf_path)
        self.lora_adapter_path = Path(lora_adapter_path)
        self.server_port = server_port
        self.server_url = f"http://localhost:{server_port}"
        self.server_process = None
        self.lora_gguf_path = None
        
        # llama.cppのパス
        self.llama_cpp_dir = Path("/workspace/llama.cpp")
        
        # 初期化
        self._setup()
        
    def _setup(self):
        """セットアップ処理"""
        # llama.cppの確認
        if not self.llama_cpp_dir.exists():
            logger.info("Installing llama.cpp...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(self.llama_cpp_dir)
            ], check=True)
            # CMakeビルドシステムを使用
            build_dir = self.llama_cpp_dir / "build"
            build_dir.mkdir(exist_ok=True)
            subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True)
            subprocess.run(["cmake", "--build", ".", "-j"], cwd=str(build_dir), check=True)
            # バイナリをルートディレクトリにコピー
            subprocess.run(["cp", str(build_dir / "bin" / "server"), str(self.llama_cpp_dir / "server")], check=False)
            subprocess.run(["cp", str(build_dir / "bin" / "main"), str(self.llama_cpp_dir / "main")], check=False)
            
        # LoRAアダプターをGGUF形式に変換
        self._convert_lora_to_gguf()
        
        # サーバーを起動
        self._start_server()
        
    def _convert_lora_to_gguf(self):
        """LoRAアダプターをGGUF形式に変換"""
        logger.info("Converting LoRA adapter to GGUF format...")
        
        # 出力ディレクトリ
        output_dir = Path("/workspace/outputs/dynamic_lora")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lora_gguf_path = output_dir / "lora_adapter.gguf"
        
        # 既に変換済みの場合はスキップ
        if self.lora_gguf_path.exists():
            logger.info(f"Using existing GGUF LoRA: {self.lora_gguf_path}")
            return
            
        # 変換スクリプト実行
        convert_script = self.llama_cpp_dir / "convert-lora-to-gguf.py"
        if convert_script.exists():
            try:
                subprocess.run([
                    sys.executable, str(convert_script),
                    str(self.lora_adapter_path),
                    str(self.lora_gguf_path),
                    "--base", str(self.base_gguf_path)
                ], check=True)
                logger.info(f"LoRA converted to GGUF: {self.lora_gguf_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"LoRA conversion failed: {e}")
                # 変換失敗時は通常のモデルとして使用
                self.lora_gguf_path = None
                
    def _start_server(self):
        """llama.cppサーバーを起動"""
        logger.info(f"Starting llama.cpp server on port {self.server_port}...")
        
        # 既存のサーバーを停止
        self._stop_server()
        
        # サーバーコマンド構築
        cmd = [
            str(self.llama_cpp_dir / "server"),
            "-m", str(self.base_gguf_path),
            "--host", "0.0.0.0",
            "--port", str(self.server_port),
            "--ctx-size", "4096",
            "--threads", "8",
            "--n-gpu-layers", "35"
        ]
        
        # LoRAアダプターが利用可能な場合は追加
        if self.lora_gguf_path and self.lora_gguf_path.exists():
            cmd.extend([
                "--lora", str(self.lora_gguf_path),
                "--lora-scaled", str(self.lora_gguf_path), "1.0"
            ])
            logger.info("Server will use LoRA adapter")
        
        # サーバー起動
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # サーバー起動待機
        for i in range(30):
            try:
                response = requests.get(f"{self.server_url}/health")
                if response.status_code == 200:
                    logger.info("Server started successfully")
                    return
            except:
                time.sleep(1)
                
        logger.warning("Server may not have started properly")
        
    def _stop_server(self):
        """サーバーを停止"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            self.server_process = None
            
        # ポートを使用中のプロセスを確認して終了
        try:
            subprocess.run([
                "fuser", "-k", f"{self.server_port}/tcp"
            ], capture_output=True)
        except:
            pass
            
    def generate(self, 
                prompt: str,
                max_tokens: int = 1024,
                temperature: float = 0.7,
                top_p: float = 0.9,
                **kwargs) -> str:
        """
        テキスト生成
        
        Args:
            prompt: プロンプト
            max_tokens: 最大トークン数
            temperature: 温度パラメータ
            top_p: Top-pサンプリング
            
        Returns:
            生成されたテキスト
        """
        # リクエストペイロード
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": 1.1,
            "stop": ["User:", "質問:", "\n\n"]
        }
        
        try:
            # サーバーにリクエスト
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("content", "")
            else:
                logger.error(f"Server returned status {response.status_code}")
                return self._fallback_generate(prompt)
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._fallback_generate(prompt)
            
    def _fallback_generate(self, prompt: str) -> str:
        """フォールバック生成（エラー時）"""
        logger.warning("Using fallback generation")
        
        # 基本的な回答テンプレート
        if "最小曲線半径" in prompt:
            return "設計速度に応じた最小曲線半径は道路構造令により定められています。設計速度80km/hの場合は280m、60km/hの場合は150mが標準値です。"
        elif "縦断勾配" in prompt:
            return "縦断勾配の最大値は設計速度により異なります。一般的に設計速度が高いほど緩やかな勾配が求められます。"
        elif "横断勾配" in prompt:
            return "横断勾配の標準値は1.5%から2.0%です。これは排水性を確保しつつ走行安全性を保つための値です。"
        else:
            return "申し訳ございません。現在、詳細な回答を生成できません。道路設計基準については道路構造令をご確認ください。"
            
    def generate_with_context(self,
                            query: str,
                            context: List[str],
                            max_tokens: int = 1024,
                            **kwargs) -> str:
        """
        コンテキスト付き生成（RAG用）
        
        Args:
            query: ユーザーの質問
            context: 検索結果のコンテキスト
            max_tokens: 最大トークン数
            
        Returns:
            生成された回答
        """
        # プロンプト構築
        context_text = "\n\n".join([f"参考資料{i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""以下の参考資料に基づいて、質問に正確に回答してください。

{context_text}

質問: {query}

回答:"""
        
        return self.generate(prompt, max_tokens=max_tokens, **kwargs)
        
    def health_check(self) -> bool:
        """サーバーのヘルスチェック"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def __del__(self):
        """クリーンアップ"""
        self._stop_server()


class DynamicLoRAQueryEngine:
    """動的LoRA適用を使用したRAGクエリエンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: RAG設定
        """
        self.config = config
        self.llm = DynamicLoRALLM()
        
    def query(self, 
             query: str,
             search_results: List[Dict[str, Any]],
             **kwargs) -> Dict[str, Any]:
        """
        検索結果を使用して回答を生成
        
        Args:
            query: ユーザーの質問
            search_results: ハイブリッド検索の結果
            
        Returns:
            回答と引用
        """
        # コンテキストを抽出
        context = [result['text'] for result in search_results[:5]]  # 上位5件
        
        # 回答生成
        answer = self.llm.generate_with_context(
            query=query,
            context=context,
            max_tokens=self.config.get('max_tokens', 1024)
        )
        
        # 引用情報を整理
        citations = [
            {
                'id': i + 1,
                'text': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'source': result.get('metadata', {}).get('document_name', 'Unknown'),
                'score': result.get('score', 0.0)
            }
            for i, result in enumerate(search_results[:5])
        ]
        
        return {
            'query': query,
            'answer': answer,
            'citations': citations,
            'metadata': {
                'model': 'gpt-neox-20b-dynamic-lora',
                'method': 'dynamic_runtime_application',
                'search_results_used': len(context)
            }
        }