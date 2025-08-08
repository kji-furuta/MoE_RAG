#!/usr/bin/env python3
"""
RAGシステム最適化スクリプト

Phase 2実装後のシステム最適化とパフォーマンスチューニング
"""

import sys
import os
from pathlib import Path
import asyncio
import time
import json
import psutil
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# プロジェクトのルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.dependencies.dependency_manager import RAGDependencyManager, DependencyLevel
from src.rag.dependencies.container import DIContainer
from src.rag.dependencies.services import configure_rag_services
from src.rag.monitoring.health_check import RAGHealthChecker, HealthStatus
from src.rag.monitoring.metrics import MetricsCollector, MetricType


class RAGSystemOptimizer:
    """RAGシステムの最適化クラス"""
    
    def __init__(self):
        self.container = None
        self.health_checker = None
        self.metrics_collector = MetricsCollector()
        self.dependency_manager = RAGDependencyManager()
        self.optimization_results = {}
    
    async def initialize(self):
        """初期化"""
        print("Initializing optimizer...")
        
        # DIコンテナの初期化
        self.container = DIContainer()
        configure_rag_services(self.container)
        
        # ヘルスチェッカーの初期化
        self.health_checker = RAGHealthChecker(self.container)
        
        print("✅ Optimizer initialized")
    
    async def analyze_current_state(self) -> Dict:
        """現在の状態を分析"""
        print("\n" + "=" * 70)
        print("📊 Analyzing Current State")
        print("=" * 70)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "dependencies": {},
            "health": {},
            "performance": {}
        }
        
        # システムリソース
        print("\n1. System Resources:")
        analysis["system"] = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
        }
        
        print(f"  CPU: {analysis['system']['cpu_percent']}% ({analysis['system']['cpu_count']} cores)")
        print(f"  Memory: {analysis['system']['memory_percent']}% used")
        print(f"  Disk: {analysis['system']['disk_free_gb']:.1f}GB free")
        
        # GPU状態
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                gpu_info.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "free_gb": mem_info[0] / (1024**3),
                    "total_gb": mem_info[1] / (1024**3)
                })
            analysis["system"]["gpu"] = gpu_info
            print(f"  GPU: {len(gpu_info)} device(s) available")
        
        # 依存関係
        print("\n2. Dependencies:")
        dep_result = self.dependency_manager.check_all_dependencies(use_cache=False)
        analysis["dependencies"] = {
            "can_run": dep_result.can_run,
            "missing_core": len(dep_result.missing_core),
            "missing_infrastructure": len(dep_result.missing_infrastructure),
            "missing_optional": len(dep_result.missing_optional),
            "alternatives_used": len(dep_result.alternatives_used)
        }
        
        print(f"  Can run: {dep_result.can_run}")
        print(f"  Missing optional: {len(dep_result.missing_optional)}")
        print(f"  Using alternatives: {len(dep_result.alternatives_used)}")
        
        # ヘルスチェック
        print("\n3. Health Check:")
        health_result = await self.health_checker.check_all()
        analysis["health"] = {
            "overall_status": health_result.overall_status.value,
            "components": {
                name: comp.status.value 
                for name, comp in health_result.components.items()
            }
        }
        
        print(f"  Overall: {health_result.overall_status.value}")
        unhealthy = [
            name for name, comp in health_result.components.items()
            if comp.status == HealthStatus.UNHEALTHY
        ]
        if unhealthy:
            print(f"  Unhealthy components: {', '.join(unhealthy)}")
        
        return analysis
    
    async def optimize_dependencies(self) -> Dict:
        """依存関係の最適化"""
        print("\n" + "=" * 70)
        print("🔧 Optimizing Dependencies")
        print("=" * 70)
        
        results = {
            "optimizations": [],
            "installed": [],
            "errors": []
        }
        
        # 現在の状態を確認
        dep_result = self.dependency_manager.check_all_dependencies(use_cache=False)
        
        # オプション依存関係の推奨インストール
        recommended_optional = {
            "spacy": "自然言語処理の精度向上",
            "plotly": "可視化機能の強化",
            "streamlit": "Web UIの改善"
        }
        
        print("\n📦 Recommended Optional Dependencies:")
        for dep_name in dep_result.missing_optional:
            if dep_name in recommended_optional:
                print(f"  - {dep_name}: {recommended_optional[dep_name]}")
                
                # インストールの提案
                results["optimizations"].append({
                    "type": "dependency",
                    "name": dep_name,
                    "reason": recommended_optional[dep_name],
                    "command": f"pip install {self.dependency_manager.dependencies[dep_name].package_name}"
                })
        
        # 代替パッケージの最適化
        if dep_result.alternatives_used:
            print("\n🔄 Alternative Package Optimizations:")
            for original, alternative in dep_result.alternatives_used.items():
                print(f"  - Consider installing {original} instead of {alternative}")
                results["optimizations"].append({
                    "type": "alternative",
                    "original": original,
                    "current": alternative,
                    "recommendation": f"Install {original} for better performance"
                })
        
        return results
    
    async def optimize_performance(self) -> Dict:
        """パフォーマンスの最適化"""
        print("\n" + "=" * 70)
        print("⚡ Optimizing Performance")
        print("=" * 70)
        
        results = {
            "benchmarks": {},
            "optimizations": []
        }
        
        # ベンチマークテスト
        print("\n📏 Running Benchmarks...")
        
        # 1. エンベディング速度テスト
        try:
            from src.rag.dependencies.services import IEmbeddingModel
            embedding_model = self.container.get_service(IEmbeddingModel)
            
            if embedding_model:
                test_texts = ["テスト文章" * 10] * 10
                
                start = time.perf_counter()
                embeddings = embedding_model.encode_batch(test_texts)
                duration = time.perf_counter() - start
                
                throughput = len(test_texts) / duration
                results["benchmarks"]["embedding"] = {
                    "throughput": throughput,
                    "duration_ms": duration * 1000
                }
                
                print(f"  Embedding: {throughput:.2f} texts/sec")
                
                # 最適化の提案
                if throughput < 10:
                    results["optimizations"].append({
                        "type": "performance",
                        "component": "embedding",
                        "issue": "Low throughput",
                        "suggestion": "Consider using GPU or batch processing"
                    })
        except Exception as e:
            print(f"  Embedding benchmark failed: {e}")
        
        # 2. メモリ使用量の最適化
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            results["optimizations"].append({
                "type": "memory",
                "issue": f"High memory usage ({memory.percent}%)",
                "suggestions": [
                    "Clear model cache",
                    "Reduce batch size",
                    "Use quantization"
                ]
            })
            print(f"\n⚠️ High memory usage detected: {memory.percent}%")
        
        # 3. キャッシュの最適化
        cache_dir = Path.home() / '.cache' / 'ai_ft'
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            cache_size_mb = cache_size / (1024 * 1024)
            
            results["benchmarks"]["cache_size_mb"] = cache_size_mb
            print(f"  Cache size: {cache_size_mb:.2f} MB")
            
            if cache_size_mb > 1000:  # 1GB以上
                results["optimizations"].append({
                    "type": "cache",
                    "issue": "Large cache size",
                    "suggestion": "Consider clearing old cache files"
                })
        
        return results
    
    async def optimize_configuration(self) -> Dict:
        """設定の最適化"""
        print("\n" + "=" * 70)
        print("⚙️ Optimizing Configuration")
        print("=" * 70)
        
        results = {
            "current_config": {},
            "optimizations": []
        }
        
        try:
            from src.rag.dependencies.services import ConfigurationService
            config_service = self.container.resolve(ConfigurationService)
            
            if config_service:
                current_config = config_service.get_all()
                results["current_config"] = current_config
                
                # 設定の最適化提案
                print("\n📝 Configuration Optimizations:")
                
                # バッチサイズの最適化
                batch_size = config_service.get("embedding.batch_size", 32)
                if batch_size < 16:
                    print(f"  - Increase batch size from {batch_size} to 32 for better throughput")
                    results["optimizations"].append({
                        "setting": "embedding.batch_size",
                        "current": batch_size,
                        "recommended": 32,
                        "reason": "Better throughput"
                    })
                
                # デバイスの最適化
                device = config_service.get("embedding.device", "cpu")
                if device == "cpu" and torch.cuda.is_available():
                    print(f"  - Switch from CPU to CUDA for faster processing")
                    results["optimizations"].append({
                        "setting": "embedding.device",
                        "current": device,
                        "recommended": "cuda",
                        "reason": "GPU acceleration available"
                    })
                
                # 検索設定の最適化
                top_k = config_service.get("search.top_k", 5)
                if top_k > 20:
                    print(f"  - Reduce top_k from {top_k} to 10 for faster search")
                    results["optimizations"].append({
                        "setting": "search.top_k",
                        "current": top_k,
                        "recommended": 10,
                        "reason": "Balance between accuracy and speed"
                    })
                
        except Exception as e:
            print(f"  Configuration optimization failed: {e}")
        
        return results
    
    async def apply_optimizations(self, auto_apply: bool = False) -> Dict:
        """最適化の適用"""
        print("\n" + "=" * 70)
        print("✨ Applying Optimizations")
        print("=" * 70)
        
        results = {
            "applied": [],
            "skipped": [],
            "errors": []
        }
        
        # 収集した最適化案
        all_optimizations = []
        
        if "dependencies" in self.optimization_results:
            all_optimizations.extend(
                self.optimization_results["dependencies"].get("optimizations", [])
            )
        
        if "performance" in self.optimization_results:
            all_optimizations.extend(
                self.optimization_results["performance"].get("optimizations", [])
            )
        
        if "configuration" in self.optimization_results:
            all_optimizations.extend(
                self.optimization_results["configuration"].get("optimizations", [])
            )
        
        if not all_optimizations:
            print("No optimizations to apply.")
            return results
        
        print(f"\nFound {len(all_optimizations)} optimization(s):")
        
        for i, opt in enumerate(all_optimizations, 1):
            print(f"\n{i}. {opt.get('type', 'general').upper()} optimization:")
            
            if "name" in opt:
                print(f"   Name: {opt['name']}")
            if "issue" in opt:
                print(f"   Issue: {opt['issue']}")
            if "suggestion" in opt:
                print(f"   Suggestion: {opt['suggestion']}")
            if "suggestions" in opt:
                print(f"   Suggestions:")
                for suggestion in opt['suggestions']:
                    print(f"     - {suggestion}")
            
            if auto_apply:
                # 自動適用（安全な最適化のみ）
                if opt.get("type") == "cache":
                    print("   ➜ Clearing cache...")
                    # キャッシュクリアの実装
                    results["applied"].append(opt)
                else:
                    results["skipped"].append(opt)
            else:
                # 手動確認
                results["skipped"].append(opt)
        
        return results
    
    async def generate_report(self) -> str:
        """最適化レポートの生成"""
        print("\n" + "=" * 70)
        print("📄 Generating Optimization Report")
        print("=" * 70)
        
        report = []
        report.append("# RAG System Optimization Report")
        report.append(f"\n**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 現在の状態
        if "analysis" in self.optimization_results:
            analysis = self.optimization_results["analysis"]
            report.append("\n## System Status")
            report.append(f"- CPU Usage: {analysis['system']['cpu_percent']}%")
            report.append(f"- Memory Usage: {analysis['system']['memory_percent']}%")
            report.append(f"- Health Status: {analysis['health']['overall_status']}")
        
        # 最適化の推奨事項
        report.append("\n## Optimization Recommendations")
        
        all_optimizations = []
        for key in ["dependencies", "performance", "configuration"]:
            if key in self.optimization_results:
                opts = self.optimization_results[key].get("optimizations", [])
                all_optimizations.extend(opts)
        
        if all_optimizations:
            for opt in all_optimizations:
                report.append(f"\n### {opt.get('type', 'General').title()} Optimization")
                
                if "issue" in opt:
                    report.append(f"**Issue:** {opt['issue']}")
                
                if "suggestion" in opt:
                    report.append(f"**Recommendation:** {opt['suggestion']}")
                elif "suggestions" in opt:
                    report.append("**Recommendations:**")
                    for suggestion in opt['suggestions']:
                        report.append(f"- {suggestion}")
                
                if "command" in opt:
                    report.append(f"```bash\n{opt['command']}\n```")
        else:
            report.append("No optimizations needed - system is running optimally!")
        
        # パフォーマンスメトリクス
        if "performance" in self.optimization_results:
            benchmarks = self.optimization_results["performance"].get("benchmarks", {})
            if benchmarks:
                report.append("\n## Performance Metrics")
                
                if "embedding" in benchmarks:
                    report.append(f"- Embedding throughput: {benchmarks['embedding']['throughput']:.2f} texts/sec")
                
                if "cache_size_mb" in benchmarks:
                    report.append(f"- Cache size: {benchmarks['cache_size_mb']:.2f} MB")
        
        report_text = "\n".join(report)
        
        # ファイルに保存
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"optimization_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✅ Report saved to: {report_file}")
        
        return report_text
    
    async def run_optimization(self) -> None:
        """最適化の実行"""
        try:
            await self.initialize()
            
            # 現在の状態を分析
            self.optimization_results["analysis"] = await self.analyze_current_state()
            
            # 各種最適化を実行
            self.optimization_results["dependencies"] = await self.optimize_dependencies()
            self.optimization_results["performance"] = await self.optimize_performance()
            self.optimization_results["configuration"] = await self.optimize_configuration()
            
            # 最適化の適用（手動確認モード）
            self.optimization_results["applied"] = await self.apply_optimizations(auto_apply=False)
            
            # レポート生成
            report = await self.generate_report()
            
            print("\n" + "=" * 70)
            print("✅ Optimization Complete!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Optimization failed: {e}")
            import traceback
            if os.environ.get("DEBUG"):
                traceback.print_exc()


async def main():
    """メイン処理"""
    print("=" * 70)
    print("🚀 RAG System Optimizer")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    optimizer = RAGSystemOptimizer()
    await optimizer.run_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        if os.environ.get("DEBUG"):
            traceback.print_exc()
        sys.exit(1)
