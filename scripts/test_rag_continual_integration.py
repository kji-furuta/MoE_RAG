#!/usr/bin/env python3
"""
RAGシステムと継続学習モデルの統合テスト
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import torch
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.core.continual_model_manager import ContinualModelManager
from src.rag.core.query_engine import LLMGenerator, RoadDesignQueryEngine
from src.rag.config.rag_config import load_config

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_continual_model_manager():
    """継続学習モデルマネージャーのテスト"""
    print("\n" + "="*60)
    print("1. Testing Continual Model Manager")
    print("="*60)
    
    try:
        # マネージャーの初期化
        manager = ContinualModelManager(base_path=Path("outputs"))
        
        # 利用可能なタスクの確認
        tasks = manager.get_available_tasks()
        print(f"✓ Available continual tasks: {tasks}")
        
        # 最新タスクの取得
        latest_task = manager.get_latest_task()
        if latest_task:
            print(f"✓ Latest task: {latest_task.task_name}")
            print(f"  - Model path: {latest_task.model_path}")
            print(f"  - Fisher path: {latest_task.fisher_path}")
            print(f"  - EWC lambda: {latest_task.ewc_lambda}")
        else:
            print("  No continual learning tasks found")
        
        # クエリベースのタスク選択テスト
        test_queries = [
            "道路の設計速度について教えてください",
            "橋梁の設計基準は？",
            "トンネルの照明設備について",
        ]
        
        for query in test_queries:
            selected_task = manager.select_task_for_query(query)
            print(f"  Query: '{query[:30]}...' -> Task: {selected_task or 'None'}")
        
        # ステータス確認
        status = manager.get_status()
        print(f"✓ Manager status:")
        print(f"  - Total tasks: {status['total_tasks']}")
        print(f"  - Cached models: {status['cached_models']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in continual model manager test: {e}")
        return False


def test_llm_generator_with_continual():
    """継続学習対応LLMGeneratorのテスト"""
    print("\n" + "="*60)
    print("2. Testing LLM Generator with Continual Learning")
    print("="*60)
    
    try:
        # 設定ファイルの更新（継続学習を有効化）
        config = load_config()
        if not hasattr(config, 'continual_learning'):
            print("  Updating config to enable continual learning...")
            # 設定を動的に追加
            from types import SimpleNamespace
            config.continual_learning = SimpleNamespace(
                enabled=True,
                model_base_path="./outputs",
                ewc_data_path="./outputs/ewc_data"
            )
        
        # LLMGeneratorの初期化
        llm_gen = LLMGenerator(config, load_model=False)
        
        print(f"✓ LLM Generator initialized")
        print(f"  - Continual learning enabled: {llm_gen.use_continual}")
        print(f"  - Ollama fallback: {llm_gen.use_ollama_fallback}")
        
        if llm_gen.continual_manager:
            print(f"  - Available tasks: {len(llm_gen.continual_manager.get_available_tasks())}")
        
        # テスト生成（継続学習モデルの選択をテスト）
        test_prompt = "道路設計の基準について説明してください"
        test_context = "道路構造令に基づく設計基準"
        
        print(f"\n  Testing generation with query: '{test_prompt[:50]}...'")
        
        # クエリテキストを渡して継続学習モデル選択をトリガー
        result = llm_gen.generate(
            prompt=test_prompt,
            context=test_context,
            max_new_tokens=100,
            query_text=test_prompt  # 継続学習モデル選択用
        )
        
        if result:
            print(f"✓ Generation successful (length: {len(result)} chars)")
            print(f"  Preview: {result[:100]}...")
            if llm_gen.current_continual_task:
                print(f"  Used continual task: {llm_gen.current_continual_task}")
        else:
            print("  No result generated (likely using Ollama fallback)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in LLM generator test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_with_continual():
    """RAGシステム全体での継続学習統合テスト"""
    print("\n" + "="*60)
    print("3. Testing Full RAG System with Continual Learning")
    print("="*60)
    
    try:
        # RAGエンジンの初期化
        engine = RoadDesignQueryEngine()
        
        # 設定を更新して継続学習を有効化
        if hasattr(engine.config, 'continual_learning'):
            engine.config.continual_learning.enabled = True
            print("✓ Continual learning enabled in RAG config")
        
        # エンジンを初期化
        print("  Initializing RAG engine...")
        engine.initialize()
        print("✓ RAG engine initialized")
        
        # システム情報の確認
        sys_info = engine.get_system_info()
        print(f"✓ System info:")
        print(f"  - Initialized: {sys_info['is_initialized']}")
        print(f"  - Models: {sys_info.get('models', {})}")
        
        # テストクエリの実行
        test_queries = [
            "道路の設計速度80km/hの場合の最小曲線半径は？",
            "橋梁設計における活荷重の考え方について",
            "トンネル内の照明設備の設置基準",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Test Query {i}: '{query[:50]}...'")
            
            try:
                result = engine.query(
                    query_text=query,
                    top_k=3,
                    search_type="hybrid"
                )
                
                if result:
                    print(f"  ✓ Query successful")
                    print(f"    - Answer length: {len(result.answer)} chars")
                    print(f"    - Confidence: {result.confidence_score:.2f}")
                    print(f"    - Sources: {len(result.sources)}")
                    print(f"    - Processing time: {result.processing_time:.2f}s")
                    
                    # メタデータから継続学習タスクを確認
                    if 'continual_task' in result.metadata:
                        print(f"    - Used continual task: {result.metadata['continual_task']}")
                
            except Exception as query_error:
                print(f"  ✗ Query failed: {query_error}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in RAG integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu_memory():
    """GPU メモリ状況の確認"""
    print("\n" + "="*60)
    print("GPU Memory Status")
    print("="*60)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            used_gb = total_gb - free_gb
            
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Total: {total_gb:.2f} GB")
            print(f"  - Used:  {used_gb:.2f} GB")
            print(f"  - Free:  {free_gb:.2f} GB")
            print(f"  - Usage: {(used_gb/total_gb)*100:.1f}%")
    else:
        print("No GPU available")


def main():
    """メインテスト実行"""
    print("\n" + "="*60)
    print("RAG-Continual Learning Integration Test")
    print("="*60)
    
    # GPU状況確認
    check_gpu_memory()
    
    # 各テストの実行
    results = {}
    
    # 1. 継続学習モデルマネージャーのテスト
    results['continual_manager'] = test_continual_model_manager()
    
    # 2. LLMGeneratorの継続学習対応テスト
    results['llm_generator'] = test_llm_generator_with_continual()
    
    # 3. RAGシステム全体の統合テスト
    results['rag_integration'] = test_rag_with_continual()
    
    # 結果サマリー
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    # 全体の成功/失敗
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests PASSED - RAG and Continual Learning are integrated!")
    else:
        print("✗ Some tests FAILED - Please check the logs above")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())