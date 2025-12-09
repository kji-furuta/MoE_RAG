import requests
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class OllamaIntegration:
    """Ollama APIを使用したモデル統合"""
    
    def __init__(self, base_url: str = None):
        # コンテナ内からWSLのOllamaに接続するためのURL設定
        if base_url is None:
            # 複数の接続先を試行
            possible_urls = [
                "http://host.docker.internal:11434",
                "http://localhost:11434",
                "http://172.17.0.1:11434",
                "http://host.gateway.internal:11434"
            ]
            
            # 接続可能なURLを探す
            import requests
            for url in possible_urls:
                try:
                    response = requests.get(f"{url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        base_url = url
                        logger.info(f"Ollama接続成功: {url}")
                        break
                except:
                    continue
            
            if base_url is None:
                # デフォルトURLを設定（エラーは後で処理）
                base_url = "http://localhost:11434"
                logger.warning("Ollama接続先が見つからないため、localhostを使用します")
        
        self.base_url = base_url
        self.available_models = []
        self._load_available_models()
    
    def _clean_generated_text(self, text: str, original_prompt: str) -> str:
        """生成されたテキストから重複やプロンプトタグをクリーンアップ"""
        import re

        if not text:
            return text

        # 1. プロンプトタグの除去（[INST], [/INST], <s>, </s>など）
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'<s>|</s>', '', text)
        text = re.sub(r'<\|.*?\|>', '', text)  # 特殊トークン

        # 2. 元のプロンプトテキストが含まれている場合は除去
        # プロンプトの最初の100文字を使って検索
        if original_prompt and len(original_prompt) > 50:
            prompt_snippet = original_prompt[:100]
            if prompt_snippet in text:
                # プロンプト部分以降のテキストのみを抽出
                text = text.split(prompt_snippet)[-1]

        # 3. 重複した大きなテキストブロックの検出と除去
        # テキストを2つに分割して、前半と後半が同じか確認
        text_length = len(text)
        if text_length > 200:  # 十分長いテキストの場合のみチェック
            # 文章の境界で分割するため、中央付近の改行を探す
            mid_point = text_length // 2
            search_start = max(0, mid_point - 100)
            search_end = min(text_length, mid_point + 100)

            # 中央付近で改行を探す
            middle_section = text[search_start:search_end]
            newline_positions = [i for i, c in enumerate(middle_section) if c == '\n']

            if newline_positions:
                # 中央に最も近い改行位置を使用
                closest_newline = min(newline_positions, key=lambda x: abs(x - (search_end - search_start) // 2))
                split_point = search_start + closest_newline + 1

                first_half = text[:split_point].strip()
                second_half = text[split_point:].strip()

                # 前半と後半の類似度をチェック（正規化して比較）
                # 完全一致または90%以上の一致の場合は重複とみなす
                if first_half == second_half:
                    logger.info("完全重複を検出、前半のみを使用")
                    text = first_half
                elif len(first_half) > 50 and len(second_half) > 50:
                    # Levenshtein距離の簡易版：文字単位の一致率
                    # 長いテキストの場合は最初の500文字で比較
                    comparison_length = 500
                    first_sample = first_half[:comparison_length]
                    second_sample = second_half[:comparison_length]

                    # 単純な一致率計算
                    matches = sum(1 for a, b in zip(first_sample, second_sample) if a == b)
                    similarity = matches / max(len(first_sample), len(second_sample))

                    if similarity > 0.9:
                        logger.info(f"高類似度重複を検出 (類似度: {similarity:.2%})、前半のみを使用")
                        text = first_half

        # 4. 前後の空白をトリム
        text = text.strip()

        return text

    def _load_available_models(self):
        """利用可能なOllamaモデルを取得"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Ollamaレスポンス: {data}")
                
                # Ollamaのレスポンス形式を確認して適切に処理
                if isinstance(data, dict) and "models" in data:
                    models_list = data["models"]
                    logger.debug(f"モデルリスト: {models_list}")
                    
                    # 各モデルの構造を確認
                    for i, model in enumerate(models_list):
                        logger.debug(f"モデル {i}: {model}")
                        if not isinstance(model, dict):
                            logger.warning(f"モデル {i} が辞書ではありません: {type(model)}")
                            continue
                        if "name" not in model:
                            logger.warning(f"モデル {i} にnameキーがありません: {model}")
                            continue
                    
                    self.available_models = [model["name"] for model in models_list if isinstance(model, dict) and "name" in model]
                elif isinstance(data, list):
                    # 直接リストが返される場合
                    logger.debug(f"直接リスト形式: {data}")
                    self.available_models = [model["name"] for model in data if isinstance(model, dict) and "name" in model]
                else:
                    logger.warning(f"予期しないOllamaレスポンス形式: {type(data)}")
                    logger.debug(f"レスポンス内容: {data}")
                    self.available_models = []
                
                logger.info(f"利用可能なOllamaモデル: {self.available_models}")
            else:
                logger.warning(f"Ollama APIに接続できません: {response.status_code}")
                self.available_models = []
        except Exception as e:
            logger.error(f"Ollamaモデル取得エラー: {e}")
            import traceback
            logger.error(f"詳細エラー: {traceback.format_exc()}")
            self.available_models = []
    
    def generate_text(self, model_name: str = None, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Ollamaを使用してテキスト生成"""
        try:
            # パラメータの検証
            if not prompt:
                return {
                    "error": "プロンプトが指定されていません",
                    "success": False
                }
            
            # モデル名が指定されていない場合はデフォルトモデルを使用
            if not model_name:
                if self.available_models:
                    model_name = self.available_models[0]
                    logger.info(f"デフォルトモデルを使用: {model_name}")
                else:
                    return {
                        "error": "利用可能なモデルがありません",
                        "success": False
                    }
            
            # Ollamaのパラメータに変換
            ollama_params = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # 追加パラメータの処理
            ollama_params["options"] = {}

            if "temperature" in kwargs:
                ollama_params["options"]["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                ollama_params["options"]["top_p"] = kwargs["top_p"]
            if "max_tokens" in kwargs:
                ollama_params["options"]["num_predict"] = kwargs["max_tokens"]

            # 繰り返し防止パラメータ（強化版）
            if "repetition_penalty" in kwargs:
                ollama_params["options"]["repeat_penalty"] = kwargs["repetition_penalty"]
            else:
                # 繰り返しペナルティを1.3に強化
                ollama_params["options"]["repeat_penalty"] = 1.3

            # 頻度ペナルティ（同じトークンの出現頻度を抑制）
            if "frequency_penalty" in kwargs:
                ollama_params["options"]["frequency_penalty"] = kwargs["frequency_penalty"]
            else:
                ollama_params["options"]["frequency_penalty"] = 0.7

            # 存在ペナルティ（既出トークンを抑制）
            if "presence_penalty" in kwargs:
                ollama_params["options"]["presence_penalty"] = kwargs["presence_penalty"]
            else:
                ollama_params["options"]["presence_penalty"] = 0.6

            # コンテキストウィンドウを拡大（文脈維持）
            ollama_params["options"]["num_ctx"] = 4096

            # 最小生成トークン数を設定（途中切断防止）
            ollama_params["options"]["min_p"] = 0.05
            
            logger.info(f"Ollama生成リクエスト: {ollama_params}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=ollama_params,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")

                # 重複テキストとプロンプトタグのクリーンアップ
                generated_text = self._clean_generated_text(generated_text, prompt)

                logger.info(f"Ollama生成成功: {len(generated_text)}文字")
                return {
                    "generated_text": generated_text,
                    "model": model_name,
                    "usage": result.get("usage", {}),
                    "success": True
                }
            else:
                error_msg = f"Ollama API エラー: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', 'Unknown error')}"
                except:
                    pass
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Ollama生成エラー: {e}")
            import traceback
            logger.error(f"詳細エラー: {traceback.format_exc()}")
            return {
                "error": str(e),
                "success": False
            }
    
    def create_model_from_hf(self, hf_model_name: str, ollama_model_name: str) -> Dict[str, Any]:
        """HuggingFaceモデルからOllamaモデルを作成"""
        try:
            # Modelfileの作成
            modelfile_content = f"""
FROM {hf_model_name}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
"""
            
            # Modelfileを保存
            modelfile_path = Path(f"modelfiles/{ollama_model_name}.Modelfile")
            modelfile_path.parent.mkdir(exist_ok=True)
            
            with open(modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)
            
            # Ollamaモデルの作成
            payload = {
                "name": ollama_model_name,
                "modelfile": modelfile_content
            }
            
            response = requests.post(
                f"{self.base_url}/api/create",
                json=payload,
                timeout=600
            )
            
            if response.status_code == 200:
                logger.info(f"Ollamaモデル作成成功: {ollama_model_name}")
                return {"success": True, "model_name": ollama_model_name}
            else:
                return {"success": False, "error": f"モデル作成失敗: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Ollamaモデル作成エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """モデル情報を取得"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"モデル情報取得失敗: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """利用可能なモデル一覧を取得"""
        try:
            # モデル一覧を再取得
            self._load_available_models()
            
            # 各モデルの詳細情報を取得
            models_info = []
            for model_name in self.available_models:
                try:
                    model_info = self.get_model_info(model_name)
                    if "error" not in model_info:
                        models_info.append({
                            "name": model_name,
                            "size": model_info.get("size", "Unknown"),
                            "modified": model_info.get("modified_at", "Unknown"),
                            "details": model_info.get("details", {})
                        })
                    else:
                        # 詳細情報が取得できない場合は基本情報のみ
                        models_info.append({
                            "name": model_name,
                            "size": "Unknown",
                            "modified": "Unknown"
                        })
                except Exception as e:
                    logger.warning(f"モデル {model_name} の詳細情報取得エラー: {e}")
                    models_info.append({
                        "name": model_name,
                        "size": "Unknown",
                        "modified": "Unknown"
                    })
            
            return {
                "success": True,
                "models": models_info
            }
        except Exception as e:
            logger.error(f"モデル一覧取得エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "models": []
            }

# 使用例
def setup_ollama_for_large_models():
    """大きなモデル用のOllama設定"""
    ollama = OllamaIntegration()
    
    # 利用可能なモデルを確認
    models = ollama.list_models()
    logger.info(f"利用可能なOllamaモデル: {models}")
    
    # 大きなモデル用の推奨設定
    large_model_configs = {
        "llama2:70b-q4_K_M": {
            "description": "Llama2 70B (4bit量子化)",
            "memory_requirement": "約20GB",
            "recommended": True
        },
        "qwen2.5:32b-q4_K_M": {
            "description": "Qwen2.5 32B (4bit量子化)",
            "memory_requirement": "約16GB",
            "recommended": True
        },
        "qwen2.5:14b-q4_K_M": {
            "description": "Qwen2.5 14B (4bit量子化)",
            "memory_requirement": "約8GB",
            "recommended": True
        }
    }
    
    return large_model_configs

if __name__ == "__main__":
    # テスト実行
    ollama = OllamaIntegration()
    configs = setup_ollama_for_large_models()
    print("Ollama設定完了:", configs) 