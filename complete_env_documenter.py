import os
import json
import subprocess
import datetime
from pathlib import Path
import yaml

class EnvironmentDocumenter:
    def __init__(self, project_root="/home/kjifuruta/AI_FT/AI_FT_3"):
        self.project_root = Path(project_root)
        self.container_name = "ai-ft-container"
        self.documentation = {
            "timestamp": datetime.datetime.now().isoformat(),
            "host_structure": {},
            "container_structure": {},
            "configuration": {},
            "dependencies": {},
            "setup_steps": []
        }
    
    def execute_in_container(self, command):
        """コンテナ内でコマンドを実行"""
        try:
            result = subprocess.check_output(
                ["docker", "exec", self.container_name] + command.split(),
                text=True,
                stderr=subprocess.STDOUT
            )
            return result.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output}"
    
    def document_container_structure(self):
        """コンテナ内部の構造を詳細に文書化"""
        print("コンテナ内部構造の調査中...")
        
        # ディレクトリ構造の取得
        structure_cmd = "find /workspace -type d -not -path '*/\\.*' -not -path '*/__pycache__*' | sort"
        directories = self.execute_in_container(structure_cmd).split("\n")
        
        # 各ディレクトリのファイル数とサイズを取得
        dir_info = {}
        for directory in directories[:50]:  # 上位50ディレクトリのみ
            if directory:
                count_cmd = f"find {directory} -maxdepth 1 -type f | wc -l"
                file_count = self.execute_in_container(count_cmd)
                dir_info[directory] = {"file_count": file_count}
        
        self.documentation["container_structure"] = {
            "directories": directories,
            "directory_info": dir_info
        }
    
    def document_critical_files(self):
        """重要ファイルの存在と内容を確認"""
        print("重要ファイルの確認中...")
        
        critical_files = {
            "configs": [
                "/workspace/config/rag_config.yaml",
                "/workspace/configs/available_models.json",
                "/workspace/docker/.env.example"
            ],
            "scripts": [
                "/workspace/scripts/rag/index_documents.py",
                "/workspace/app/main_unified.py",
                "/workspace/start_server.sh"
            ],
            "data_dirs": [
                "/workspace/data/rag_documents",
                "/workspace/temp_uploads",
                "/workspace/qdrant_data"
            ]
        }
        
        file_status = {}
        for category, files in critical_files.items():
            file_status[category] = {}
            for file_path in files:
                # ファイルまたはディレクトリの存在確認
                check_cmd = f"test -e {file_path} && echo 'exists' || echo 'missing'"
                status = self.execute_in_container(check_cmd)
                
                # ファイルの場合は最初の数行を取得
                if status == "exists" and not file_path.endswith('/'):
                    head_cmd = f"head -n 5 {file_path}"
                    preview = self.execute_in_container(head_cmd)
                    file_status[category][file_path] = {
                        "status": "exists",
                        "preview": preview[:200]  # 最初の200文字のみ
                    }
                else:
                    file_status[category][file_path] = {"status": status}
        
        self.documentation["configuration"]["critical_files"] = file_status
    
    def document_python_environment(self):
        """Python環境の詳細を文書化"""
        print("Python環境の調査中...")
        
        # インストール済みパッケージ
        packages_cmd = "pip list --format=json"
        packages_json = self.execute_in_container(packages_cmd)
        
        try:
            packages = json.loads(packages_json)
            # 主要パッケージのみ抽出
            key_packages = ["torch", "transformers", "accelerate", "peft", 
                          "qdrant-client", "fastapi", "langchain", "sentence-transformers"]
            filtered_packages = {
                pkg["name"]: pkg["version"] 
                for pkg in packages 
                if any(key in pkg["name"] for key in key_packages)
            }
            self.documentation["dependencies"]["python_packages"] = filtered_packages
        except:
            self.documentation["dependencies"]["python_packages"] = "Failed to parse"
        
        # Python実行環境
        python_info_cmd = "python -c \"import sys; print(f'Python: {sys.version}')\""
        python_version = self.execute_in_container(python_info_cmd)
        self.documentation["dependencies"]["python_version"] = python_version
    
    def generate_setup_documentation(self):
        """セットアップ手順を生成"""
        print("セットアップ手順の生成中...")
        
        setup_steps = [
            {
                "step": 1,
                "name": "リポジトリのクローン",
                "commands": [
                    "git clone <repository_url>",
                    "cd AI_FT_3"
                ]
            },
            {
                "step": 2,
                "name": "必要なディレクトリの作成",
                "commands": [
                    "mkdir -p data/{raw,processed,uploaded,rag_documents}",
                    "mkdir -p outputs/rag_index/processed_documents",
                    "mkdir -p temp_uploads qdrant_data logs docker/logs",
                    "mkdir -p models/checkpoints"
                ]
            },
            {
                "step": 3,
                "name": "環境変数の設定",
                "commands": [
                    "cp docker/.env.example docker/.env",
                    "# docker/.env を編集して HF_TOKEN を設定"
                ]
            },
            {
                "step": 4,
                "name": "Dockerイメージのビルド",
                "commands": [
                    "cd docker",
                    "docker-compose build"
                ]
            },
            {
                "step": 5,
                "name": "コンテナの起動",
                "commands": [
                    "docker-compose up -d"
                ]
            },
            {
                "step": 6,
                "name": "初期設定の実行",
                "commands": [
                    "# RAGインデックスの作成",
                    "docker exec -it ai-ft-container python scripts/rag/index_documents.py"
                ]
            }
        ]
        
        self.documentation["setup_steps"] = setup_steps
    
    def generate_complete_manual(self):
        """完全なマニュアルを生成"""
        output_dir = self.project_root / "docs" / "complete_setup"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ収集
        self.document_container_structure()
        self.document_critical_files()
        self.document_python_environment()
        self.generate_setup_documentation()
        # Markdownマニュアルの生成
        manual_content = f"""# AI_FT_3 完全環境構築マニュアル

生成日時: {self.documentation['timestamp']}

## 1. システム要件

### ハードウェア要件
- GPU: NVIDIA GPU (CUDA 12.6対応)
- メモリ: 32GB以上推奨
- ディスク: 100GB以上の空き容量

### ソフトウェア要件
- Docker Desktop (WSL2バックエンド有効)
- WSL2 (Ubuntu 20.04以降)
- NVIDIA Docker Runtime

## 2. プロジェクト構造

### コンテナ内部構造 (/workspace)
```
{chr(10).join(self.documentation['container_structure']['directories'][:30])}
...
```

### 重要ファイルの確認結果
"""
        
        # 重要ファイルの状態を追加
        for category, files in self.documentation['configuration']['critical_files'].items():
            manual_content += f"\
#### {category}\
"
            for file_path, info in files.items():
                status_icon = "✓" if info['status'] == 'exists' else "✗"
                manual_content += f"- {status_icon} `{file_path}`\
"
        
        # Python環境情報
        manual_content += "\
## 3. Python環境\
\
"
        manual_content += f"{self.documentation['dependencies']['python_version']}\
\
"
        manual_content += "### 主要パッケージ\
"
        
        if isinstance(self.documentation['dependencies']['python_packages'], dict):
            for pkg, version in self.documentation['dependencies']['python_packages'].items():
                manual_content += f"- {pkg}: {version}\
"
        
        # セットアップ手順
        manual_content += "\
## 4. セットアップ手順\
\
"
        for step in self.documentation['setup_steps']:
            manual_content += f"### {step['step']}. {step['name']}\
\
```bash\
"
            manual_content += "\n".join(step['commands'])
            manual_content += "\
```\
\
"
        
        # トラブルシューティング
        manual_content += """## 5. トラブルシューティング

### コンテナが起動しない場合
1. Docker Desktop が起動していることを確認
2. WSL2 が有効になっていることを確認
3. GPU ドライバーが最新であることを確認

### ファイルが見つからないエラー
1. 必要なディレクトリがすべて作成されているか確認
2. ファイルの権限を確認: `chmod -R 755 .`

### GPU が認識されない場合
```bash
docker exec ai-ft-container python -c \"import torch; print(torch.cuda.is_available())\"
```

## 6. 動作確認

### Web UI へのアクセス
- http://localhost:8050 - メインインターフェース
- http://localhost:8888 - Jupyter Lab

### コンテナ内での作業
```bash
docker exec -it ai-ft-container bash
```

### ログの確認
```bash
docker logs ai-ft-container
tail -f docker/logs/indexing.log
```
"""
        
        # マニュアルを保存
        manual_path = output_dir / "complete_setup_manual.md"
        with open(manual_path, 'w', encoding='utf-8') as f:
            f.write(manual_content)
        
        # JSON形式でも保存
        json_path = output_dir / "environment_details.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.documentation, f, indent=2, ensure_ascii=False)
        
        print(f"\
✓ マニュアルを生成しました:")
        print(f"  - {manual_path}")
        print(f"  - {json_path}")
        
        return self.documentation

if __name__ == "__main__":
    documenter = EnvironmentDocumenter()
    
    # コンテナが起動しているか確認
    try:
        result = subprocess.check_output(["docker", "ps", "-q", "-f", f"name={documenter.container_name}"], text=True)
        if result.strip():
            print("コンテナが検出されました。文書化を開始します...")
            documenter.generate_complete_manual()
        else:
            print("エラー: コンテナが起動していません。")
            print("以下のコマンドでコンテナを起動してください:")
            print("cd ~/AI_FT/AI_FT_3/docker && docker-compose up -d")
    except subprocess.CalledProcessError:
        print("エラー: Dockerコマンドの実行に失敗しました。")
        print("Docker Desktopが起動していることを確認してください。")
