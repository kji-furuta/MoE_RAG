#!/usr/bin/env python3
"""
WSL Ubuntu と Docker Desktop のシステム構造情報を収集するスクリプト
"""

import os
import json
import subprocess
import datetime
from pathlib import Path

class SystemInfoCollector:
    def __init__(self, project_root="/home/kjifuruta/AI_FT/AI_FT_3"):
        self.project_root = Path(project_root)
        self.report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "wsl_info": {},
            "docker_info": {},
            "project_structure": {},
            "network_info": {},
            "volume_info": {}
        }
    
    def collect_wsl_info(self):
        """WSL環境情報を収集"""
        try:
            # WSL バージョン
            wsl_version = subprocess.check_output(
                ["wsl", "--version"], 
                text=True, 
                stderr=subprocess.DEVNULL
            ).strip()
            
            # ディストリビューション情報
            distro_info = subprocess.check_output(
                ["lsb_release", "-a"], 
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            
            # ディスク使用状況
            disk_usage = subprocess.check_output(
                ["df", "-h", str(self.project_root)], 
                text=True
            ).strip()
            
            self.report["wsl_info"] = {
                "wsl_version": wsl_version,
                "distribution": distro_info,
                "disk_usage": disk_usage,
                "project_path": str(self.project_root)
            }
        except Exception as e:
            self.report["wsl_info"]["error"] = str(e)
    
    def collect_docker_info(self):
        """Docker環境情報を収集"""
        try:
            # Docker バージョン
            docker_version = subprocess.check_output(
                ["docker", "version", "--format", "json"], 
                text=True
            )
            
            # 実行中のコンテナ
            containers = subprocess.check_output(
                ["docker", "ps", "--format", "json"], 
                text=True
            ).strip().split('\n')
            
            # Docker Compose サービス状態
            compose_services = subprocess.check_output(
                ["docker-compose", "-f", str(self.project_root / "docker/docker-compose.yml"), "ps", "--format", "json"],
                text=True,
                cwd=str(self.project_root / "docker")
            )
            
            self.report["docker_info"] = {
                "version": json.loads(docker_version) if docker_version else {},
                "running_containers": [json.loads(c) for c in containers if c],
                "compose_services": json.loads(compose_services) if compose_services else []
            }
        except Exception as e:
            self.report["docker_info"]["error"] = str(e)
    
    def collect_project_structure(self):
        """プロジェクト構造を収集"""
        structure = {}
        
        def scan_directory(path, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return {"type": "directory", "truncated": True}
            
            result = {"type": "directory", "children": {}}
            try:
                for item in sorted(path.iterdir()):
                    if item.name.startswith('.') and item.name not in ['.env', '.gitignore']:
                        continue
                    
                    if item.is_dir():
                        result["children"][item.name] = scan_directory(item, max_depth, current_depth + 1)
                    else:
                        result["children"][item.name] = {
                            "type": "file",
                            "size": item.stat().st_size
                        }
            except PermissionError:
                result["permission_denied"] = True
            
            return result
        
        self.report["project_structure"] = scan_directory(self.project_root)
    
    def collect_network_info(self):
        """ネットワーク情報を収集"""
        try:
            # Docker ネットワーク一覧
            networks = subprocess.check_output(
                ["docker", "network", "ls", "--format", "json"],
                text=True
            ).strip().split('\n')
            
            # ポートマッピング情報
            port_mappings = []
            containers = subprocess.check_output(
                ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
                text=True
            ).strip().split('\n')
            
            for container in containers:
                if container:
                    parts = container.split('\t')
                    if len(parts) == 2:
                        port_mappings.append({
                            "container": parts[0],
                            "ports": parts[1]
                        })
            
            self.report["network_info"] = {
                "networks": [json.loads(n) for n in networks if n],
                "port_mappings": port_mappings
            }
        except Exception as e:
            self.report["network_info"]["error"] = str(e)
    
    def collect_volume_info(self):
        """ボリューム情報を収集"""
        try:
            # Docker ボリューム一覧
            volumes = subprocess.check_output(
                ["docker", "volume", "ls", "--format", "json"],
                text=True
            ).strip().split('\n')
            
            # マウント情報
            mount_info = subprocess.check_output(
                ["docker", "inspect", "ai-ft-container", "--format", "{{json .Mounts}}"],
                text=True
            ).strip()
            
            self.report["volume_info"] = {
                "volumes": [json.loads(v) for v in volumes if v],
                "container_mounts": json.loads(mount_info) if mount_info else []
            }
        except Exception as e:
            self.report["volume_info"]["error"] = str(e)
    
    def generate_report(self, output_path="system_structure_report.json"):
        """レポートを生成"""
        print("システム情報を収集中...")
        
        self.collect_wsl_info()
        print("✓ WSL情報を収集しました")
        
        self.collect_docker_info()
        print("✓ Docker情報を収集しました")
        
        self.collect_project_structure()
        print("✓ プロジェクト構造を収集しました")
        
        self.collect_network_info()
        print("✓ ネットワーク情報を収集しました")
        
        self.collect_volume_info()
        print("✓ ボリューム情報を収集しました")
        
        # レポート保存
        output_file = self.project_root / output_path
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print(f"\nレポートを保存しました: {output_file}")
        return self.report
    
    def generate_markdown_report(self, output_path="system_structure_manual.md"):
        """Markdownフォーマットのマニュアルを生成"""
        if not self.report.get("timestamp"):
            self.generate_report()
        
        md_content = f"""# WSL Ubuntu と Docker Desktop システム構造マニュアル

生成日時: {self.report['timestamp']}

## WSL環境情報

{self.report.get('wsl_info', {}).get('distribution', 'N/A')}

プロジェクトパス: `{self.report.get('wsl_info', {}).get('project_path', 'N/A')}`

## Docker環境情報

### 実行中のコンテナ
"""
        
        # コンテナ情報を追加
        for container in self.report.get('docker_info', {}).get('running_containers', []):
            md_content += f"\n- **{container.get('Names', 'N/A')}**: {container.get('Status', 'N/A')}"
        
        # ネットワーク情報を追加
        md_content += "\n\n## ネットワーク構成\n\n### ポートマッピング\n"
        for mapping in self.report.get('network_info', {}).get('port_mappings', []):
            md_content += f"\n- **{mapping['container']}**: {mapping['ports']}"
        
        # ボリューム情報を追加
        md_content += "\n\n## ボリューム構成\n"
        for volume in self.report.get('volume_info', {}).get('volumes', []):
            md_content += f"\n- {volume.get('Name', 'N/A')}"
        
        # ファイルに保存
        output_file = self.project_root / output_path
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdownマニュアルを保存しました: {output_file}")

if __name__ == "__main__":
    # プロジェクトルートパスを適切に設定
    collector = SystemInfoCollector()
    
    # JSON形式のレポート生成
    collector.generate_report()
    
    # Markdown形式のマニュアル生成
    collector.generate_markdown_report()
    
    print("\n収集完了！生成されたファイルを確認してください。")

Successfully wrote to \\wsl$\Ubuntu\home\kjifuruta\AI_FT\AI_FT_3\complete_env_documenter.py
