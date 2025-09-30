#!/usr/bin/env python3
"""
Ollamaモデルの設定を動的に更新するスクリプト
Dockerコンテナ内のOllamaモデルを取得してRAG設定ファイルを更新
"""

import subprocess
import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ollama_models() -> List[str]:
    """Ollamaで利用可能なモデル一覧を取得"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # 出力を解析してモデル名を抽出
        models = []
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # ヘッダー行をスキップ
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)
                    logger.info(f"Found Ollama model: {model_name}")

        return models
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get Ollama models: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        return []

def update_rag_config(models: List[str]):
    """RAG設定ファイルのモデルリストを更新"""
    config_path = Path("/workspace/src/rag/config/rag_config.yaml")

    if not config_path.exists():
        logger.error(f"RAG config file not found: {config_path}")
        return False

    try:
        # 既存の設定を読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 既存の設定モデルを保持しつつ、実際のOllamaモデルを追加
        existing_models = config.get('llm', {}).get('available_models', [])

        # Ollamaモデルと既存モデルをマージ（重複を除去）
        all_models = list(set(models + existing_models))
        all_models.sort()  # アルファベット順にソート

        # 設定を更新
        config['llm']['available_models'] = all_models

        # デフォルトモデルの設定（存在する場合のみ更新）
        if models and 'base_model' in config.get('llm', {}):
            # llama3.2:3bを優先、なければ最初のモデルを使用
            if 'llama3.2:3b' in models:
                config['llm']['base_model'] = 'llama3.2:3b'
            elif 'deepseek-32b-base:latest' in models:
                config['llm']['base_model'] = 'deepseek-32b-base:latest'
            elif models:
                config['llm']['base_model'] = models[0]

        # ファイルに書き戻し
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"Updated RAG config with {len(all_models)} models")
        logger.info(f"Available models: {all_models}")
        return True

    except Exception as e:
        logger.error(f"Error updating RAG config: {e}")
        return False

def create_model_registry():
    """Ollamaモデルレジストリファイルを作成"""
    registry_path = Path("/workspace/models/ollama_models.json")

    try:
        models = get_ollama_models()

        # モデル情報を収集
        model_info = []
        for model_name in models:
            try:
                # ollamaでモデル情報を取得
                result = subprocess.run(
                    ["ollama", "show", model_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                # 基本情報を作成
                info = {
                    "name": model_name,
                    "type": "ollama",
                    "available": True
                }

                # サイズ情報を取得（可能な場合）
                list_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True
                )

                for line in list_result.stdout.split('\n'):
                    if model_name in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            # サイズ情報を抽出
                            size_str = parts[2]
                            info["size"] = size_str
                        break

                model_info.append(info)

            except Exception as e:
                logger.warning(f"Could not get info for model {model_name}: {e}")
                model_info.append({
                    "name": model_name,
                    "type": "ollama",
                    "available": True,
                    "size": "Unknown"
                })

        # レジストリを保存
        registry = {
            "models": model_info,
            "last_updated": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip()
        }

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

        logger.info(f"Created Ollama model registry: {registry_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating model registry: {e}")
        return False

def wait_for_ollama_ready(max_wait=30):
    """Ollamaサービスが完全に起動するまで待機"""
    import time
    for i in range(max_wait):
        try:
            # pgrep でプロセスを確認
            subprocess.run(["pgrep", "-x", "ollama"], check=True, capture_output=True)
            # ollama list で実際に応答するか確認
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"Ollama service is ready (waited {i} seconds)")
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        if i % 5 == 0 and i > 0:
            logger.info(f"Waiting for Ollama service... ({i}/{max_wait} seconds)")
        time.sleep(1)

    logger.error(f"Ollama service did not become ready within {max_wait} seconds")
    return False

def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("Updating Ollama models configuration")
    logger.info("=" * 60)

    # Ollamaサービスが起動しているか確認
    try:
        subprocess.run(["pgrep", "-x", "ollama"], check=True, capture_output=True)
        logger.info("Ollama process found")
    except subprocess.CalledProcessError:
        logger.warning("Ollama process not found. Starting...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Ollamaが完全に起動するまで待機
    if not wait_for_ollama_ready():
        logger.error("Failed to start Ollama service")
        return

    # モデル一覧を取得
    models = get_ollama_models()

    if not models:
        logger.warning("No Ollama models found")
        # デフォルトモデルをプル
        logger.info("Pulling default model: llama3.2:3b")
        subprocess.run(["ollama", "pull", "llama3.2:3b"], check=False)
        models = get_ollama_models()

    # RAG設定を更新
    if models:
        update_rag_config(models)
        create_model_registry()
        logger.info("Configuration update complete")
    else:
        logger.error("No models available to update configuration")

    logger.info("=" * 60)

if __name__ == "__main__":
    main()