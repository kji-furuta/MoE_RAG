#!/usr/bin/env python3
"""
GGUFモデルの初期化と登録を行うスタートアップスクリプト
Dockerコンテナ起動時に実行され、利用可能なGGUFモデルをシステムに登録する
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GGUFModelManager:
    """GGUFモデルの管理と登録"""

    def __init__(self):
        self.workspace_dir = Path("/workspace")
        self.models_dir = self.workspace_dir / "models"
        self.registry_file = self.models_dir / "gguf_registry.json"
        self.ollama_models_cache = Path.home() / ".ollama" / "models"

    def scan_gguf_models(self):
        """modelsディレクトリのGGUFファイルをスキャン（サブディレクトリも含む）"""
        gguf_files = []
        if self.models_dir.exists():
            # サブディレクトリも含めて再帰的にスキャン
            gguf_files = list(self.models_dir.rglob("*.gguf"))
            logger.info(f"Found {len(gguf_files)} GGUF files in {self.models_dir} (including subdirectories)")
        else:
            logger.warning(f"Models directory not found: {self.models_dir}")
        return gguf_files

    def load_or_create_registry(self):
        """レジストリファイルの読み込みまたは作成"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                    logger.info(f"Loaded existing registry with {len(registry.get('models', []))} models")
                    return registry
            except Exception as e:
                logger.error(f"Error loading registry: {e}")

        # 新規作成
        registry = {
            "models": [],
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
        logger.info("Created new model registry")
        return registry

    def determine_model_type(self, filename):
        """ファイル名からモデルタイプを判定"""
        name = filename.lower()

        if "lora" in name or "adapter" in name:
            return "lora_adapter"
        elif any(x in name for x in ["q4", "q5", "q8", "quantized"]):
            return "quantized_base"
        elif "finetuned" in name:
            if any(c.isdigit() for c in filename[:3]):  # 番号付きは適用済み
                return "lora_applied"
            return "finetuned"
        elif "base" in name or "original" in name:
            return "base"
        else:
            return "custom"

    def get_model_metadata(self, file_path):
        """モデルファイルのメタデータを取得"""
        stat = file_path.stat()
        return {
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:]
        }

    def update_registry(self):
        """レジストリを更新"""
        registry = self.load_or_create_registry()
        gguf_files = self.scan_gguf_models()

        # 既存のモデルマップ
        existing_models = {m["name"]: m for m in registry.get("models", [])}

        # 更新されたモデルリスト
        updated_models = []
        new_count = 0
        updated_count = 0

        for gguf_file in gguf_files:
            model_name = gguf_file.stem
            metadata = self.get_model_metadata(gguf_file)

            model_entry = {
                "name": model_name,
                "filename": gguf_file.name,
                "path": str(gguf_file),
                "type": self.determine_model_type(gguf_file.name),
                "format": "gguf",
                **metadata,
                "registered_at": datetime.now().isoformat()
            }

            if model_name in existing_models:
                # 既存エントリを更新
                existing = existing_models[model_name]
                if existing.get("size_mb") != metadata["size_mb"] or \
                   existing.get("modified") != metadata["modified"]:
                    model_entry["registered_at"] = existing.get("registered_at", datetime.now().isoformat())
                    model_entry["updated_at"] = datetime.now().isoformat()
                    updated_count += 1
                else:
                    model_entry = existing  # 変更なし
            else:
                # 新規エントリ
                new_count += 1

            updated_models.append(model_entry)

        # ファイルが削除されたモデルを検出
        removed_models = []
        for name, model in existing_models.items():
            if not any(m["name"] == name for m in updated_models):
                removed_models.append(name)

        # レジストリを更新
        registry["models"] = updated_models
        registry["last_updated"] = datetime.now().isoformat()
        registry["statistics"] = {
            "total": len(updated_models),
            "new": new_count,
            "updated": updated_count,
            "removed": len(removed_models)
        }

        # ファイルに保存
        self.save_registry(registry)

        # サマリーを出力
        logger.info(f"Registry updated: {len(updated_models)} total models")
        if new_count:
            logger.info(f"  - {new_count} new models added")
        if updated_count:
            logger.info(f"  - {updated_count} models updated")
        if removed_models:
            logger.info(f"  - {len(removed_models)} models removed: {removed_models}")

        return registry

    def save_registry(self, registry):
        """レジストリをファイルに保存"""
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            logger.info(f"Registry saved to {self.registry_file}")

            # バックアップも作成
            backup_file = self.registry_file.with_suffix('.json.bak')
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def register_with_ollama(self, model_name, gguf_path):
        """OllamaにGGUFモデルを登録（オプション）"""
        try:
            # Modelfileを作成
            modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.95"""

            modelfile_path = self.models_dir / f"{model_name}.Modelfile"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)

            # Ollamaに登録
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info(f"Successfully registered {model_name} with Ollama")
                return True
            else:
                logger.warning(f"Failed to register {model_name} with Ollama: {result.stderr}")
                return False

        except Exception as e:
            logger.warning(f"Could not register with Ollama: {e}")
            return False

    def initialize(self):
        """初期化処理のメインエントリポイント"""
        logger.info("=" * 60)
        logger.info("Starting GGUF Model Initialization")
        logger.info("=" * 60)

        try:
            # レジストリを更新
            registry = self.update_registry()

            # モデルのサマリーを表示
            logger.info("\nAvailable GGUF Models:")
            for model in registry["models"]:
                logger.info(f"  - {model['name']}")
                logger.info(f"    Type: {model['type']}")
                logger.info(f"    Size: {model['size_mb']} MB")
                logger.info(f"    Path: {model['path']}")

            logger.info("\n" + "=" * 60)
            logger.info("GGUF Model Initialization Complete")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """メイン処理"""
    manager = GGUFModelManager()

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        if sys.argv[1] == "--register-ollama":
            # Ollamaへの登録も実行
            registry = manager.load_or_create_registry()
            for model in registry.get("models", []):
                if model["type"] in ["base", "finetuned", "lora_applied", "quantized_base"]:
                    manager.register_with_ollama(model["name"], model["path"])

    # 初期化を実行
    success = manager.initialize()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()