#!/usr/bin/env python3
"""
AI_FT_3 コンテナ環境調査スクリプト（修正版）
"""

import os
import json
import subprocess
import datetime
from pathlib import Path

class ContainerInspector:
    def __init__(self):
        self.container_name = "ai-ft-container"
        self.report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "container_info": {},
            "file_structure": {},
            "packages": {},
            "volumes": {}
        }
    
    def run_docker_command(self, command):
        """Dockerコマンドを安全に実行"""
        try:
            cmd = ["docker", "exec", self.container_name] + command
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr.strip()}"
        except Exception as e:
            return f"Exception: {str(e)}"
    
    def inspect_workspace(self):
        """ワークスペースの構造を調査"""
        print("ワークスペース構造を調査中...")
        
        # ルートディレクトリの内容
        workspace_contents = self.run_docker_command(["ls", "-la", "/workspace/"])
        self.report["file_structure"]["workspace_root"] = workspace_contents
        
        # 主要ディレクトリの存在確認
        dirs_to_check = ["src", "app", "configs", "scripts", "data", "outputs", "docker"]
        existing_dirs = {}
        
        for dir_name in dirs_to_check:
            check_result = self.run_docker_command(["test", "-d", f"/workspace/{dir_name}"])
            exists = "Error" not in check_result
            existing_dirs[dir_name] = exists
            
            if exists:
                # ディレクトリの内容を取得
                contents = self.run_docker_command(["ls", "-la", f"/workspace/{dir_name}/"])
                existing_dirs[f"{dir_name}_contents"] = contents[:500]  # 最初の500文字
        
        self.report["file_structure"]["directories"] = existing_dirs
    
    def check_critical_files(self):
        """重要ファイルの存在確認"""
        print("重要ファイルを確認中...")
        
        files_to_check = [
            "requirements.txt",
            "requirements_rag.txt",
            "docker-compose.yml",
            "app/main_unified.py",
            "scripts/rag/index_documents.py"
        ]
        
        file_status = {}
        for file_path in files_to_check:
            full_path = f"/workspace/{file_path}"
            # ファイル存在確認
            result = self.run_docker_command(["test", "-f", full_path])
            exists = "Error" not in result
            
            file_status[file_path] = {
                "exists": exists,
                "path": full_path
            }
            
            if exists:
                # ファイルの最初の5行を取得
                head_content = self.run_docker_command(["head", "-n", "5", full_path])
                file_status[file_path]["preview"] = head_content[:200]
        
        self.report["file_structure"]["critical_files"] = file_status
    
    def check_python_packages(self):
        """Pythonパッケージを確認"""
        print("Pythonパッケージを確認中...")
        
        # pip listをJSON形式で取得
        pip_list = self.run_docker_command(["pip", "list", "--format", "json"])
        
        try:
            all_packages = json.loads(pip_list)
            # 主要パッケージをフィルタ
            important_packages = [
                "torch", "transformers", "accelerate", "peft",
                "qdrant-client", "fastapi", "langchain", "sentence-transformers"
            ]
            
            filtered = {}
            for pkg in all_packages:
                if any(imp in pkg["name"] for imp in important_packages):
                    filtered[pkg["name"]] = pkg["version"]
            
            self.report["packages"] = filtered
        except:
            self.report["packages"] = {"error": "Failed to parse pip list"}
    
    def check_volumes(self):
        """ボリュームマウントを確認"""
        print("ボリュームマウントを確認中...")
        
        # docker inspectでマウント情報を取得
        inspect_result = subprocess.run(
            ["docker", "inspect", self.container_name],
            capture_output=True, text=True
        )
        
        if inspect_result.returncode == 0:
            try:
                container_info = json.loads(inspect_result.stdout)[0]
                mounts = container_info.get("Mounts", [])
                
                mount_info = []
                for mount in mounts:
                    mount_info.append({
                        "source": mount.get("Source", ""),
                        "destination": mount.get("Destination", ""),
                        "type": mount.get("Type", "")
                    })
                
                self.report["volumes"] = mount_info
            except:
                self.report["volumes"] = {"error": "Failed to parse mount info"}
    
    def generate_setup_guide(self):
        """セットアップガイドを生成"""
        output_dir = Path("docs/container_setup")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ収集
        self.inspect_workspace()
        self.check_critical_files()
        self.check_python_packages()
        self.check_volumes()
        
        # Markdownレポート生成
        report_content = f"""# AI_FT_3 コンテナ環境セットアップガイド

生成日時: {self.report['timestamp']}

## 1. コンテナ内ファイル構造

### ワークスペースルート (/workspace)
```
{self.report['file_structure'].get('workspace_root', 'N/A')}
```

### 存在するディレクトリ
"""
        
        for dir_name, exists in self.report['file_structure']['directories'].items():
            if not dir_name.endswith('_contents') and exists:
                report_content += f"- ✓ /workspace/{dir_name}\n"
        
        report_content += "\n### 重要ファイルの状態\n"
        for file_path, info in self.report['file_structure'].get('critical_files', {}).items():
            if info['exists']:
                report_content += f"- ✓ {file_path}\n"
            else:
                report_content += f"- ✗ {file_path} (要作成)\n"
        
        report_content += "\n## 2. インストール済みパッケージ\n"
        for pkg, version in self.report.get('packages', {}).items():
            report_content += f"- {pkg}: {version}\n"
        
        report_content += "\n## 3. ボリュームマウント設定\n"
        report_content += "| ホスト | コンテナ | タイプ |\n"
        report_content += "|--------|----------|--------|\n"
        
        for mount in self.report.get('volumes', []):
            if isinstance(mount, dict):
                source = mount.get('source', '').replace('/home/kjifuruta/AI_FT/AI_FT_3/', './')
                dest = mount.get('destination', '')
                mount_type = mount.get('type', '')
                report_content += f"| {source} | {dest} | {mount_type} |\n"
        
        report_content += """
## 4. 環境構築手順

### 必要なファイルとディレクトリ

1. **プロジェクト構造の作成**
```bash
mkdir -p src app configs scripts data outputs docker
mkdir -p data/{raw,processed,uploaded,rag_documents}
mkdir -p outputs/rag_index
```

2. **必要なファイルの配置**
- `requirements.txt` - Python依存関係
- `docker/docker-compose.yml` - Docker構成
- `docker/Dockerfile` - コンテナイメージ定義
- `app/main_unified.py` - メインアプリケーション

3. **環境変数の設定**
```bash
# docker/.env ファイルを作成
HF_TOKEN=your_token_here
CUDA_VISIBLE_DEVICES=0,1
```

4. **コンテナの起動**
```bash
cd docker
docker-compose up -d
```

## 5. 動作確認

```bash
# コンテナにアクセス
docker exec -it ai-ft-container bash

# GPU確認
python -c "import torch; print(torch.cuda.is_available())"

# Webアプリケーション起動
python app/main_unified.py
```
"""
        
        # ファイル保存
        report_path = output_dir / "container_setup_guide.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # JSONデータも保存
        json_path = output_dir / "container_inspection_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ セットアップガイドを生成しました:")
        print(f"  - {report_path}")
        print(f"  - {json_path}")

if __name__ == "__main__":
    inspector = ContainerInspector()
    
    # コンテナの存在確認
    check_result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={inspector.container_name}"],
        capture_output=True, text=True
    )
    
    if check_result.stdout.strip():
        print("コンテナを検出しました。調査を開始します...")
        inspector.generate_setup_guide()
    else:
        print("エラー: コンテナが起動していません。")
        print("docker-compose up -d でコンテナを起動してください。")
