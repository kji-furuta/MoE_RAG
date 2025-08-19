#!/usr/bin/env python3
"""
MoE-RAG統合テストスクリプト
RAGシステムでMoEモデルを選択して検索を実行するテスト
"""

import sys
import json
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# 必要なモジュールのインポート
from src.rag.config.rag_config import RAGConfig, load_config, save_config
from src.rag.core.query_engine import RoadDesignQueryEngine


def test_moe_integration():
    """MoE統合テスト"""
    
    print("=" * 60)
    print("MoE-RAG統合テスト開始")
    print("=" * 60)
    
    # 1. RAG設定を読み込み
    print("\n1. RAG設定を読み込み中...")
    config = load_config()
    
    # 2. MoE設定を有効化
    print("\n2. MoE統合を有効化...")
    config.llm.use_moe = True
    config.llm.moe_num_experts = 8
    config.llm.moe_experts_per_token = 2
    
    # 設定を保存
    save_config(config, "src/rag/config/rag_config_moe_test.yaml")
    print("  MoE設定を保存しました")
    
    # 3. QueryEngineを初期化
    print("\n3. QueryEngineを初期化中...")
    query_engine = RoadDesignQueryEngine(
        config_path="src/rag/config/rag_config_moe_test.yaml"
    )
    query_engine.initialize()
    
    # MoE統合が有効かチェック
    if query_engine.use_moe:
        print("  ✓ MoE統合が有効になりました")
    else:
        print("  ✗ MoE統合が有効になりませんでした")
        return
    
    # 4. テストクエリを実行
    print("\n4. テストクエリを実行中...")
    test_queries = [
        "設計速度80km/hの道路の最小曲線半径は？",
        "橋梁の耐震設計における照査内容について教えてください",
        "アスファルト舗装の品質管理基準を教えてください"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n  クエリ {i}: {query}")
        
        try:
            # クエリ実行
            start_time = time.time()
            result = query_engine.query(
                query_text=query,
                top_k=3,
                search_type="hybrid"
            )
            elapsed_time = time.time() - start_time
            
            # 結果を表示
            print(f"    処理時間: {elapsed_time:.2f}秒")
            print(f"    信頼度: {result.confidence_score:.2f}")
            
            # MoE情報を表示
            if 'moe_experts' in result.metadata:
                print(f"    選択されたエキスパート: {', '.join(result.metadata['moe_experts'])}")
            
            print(f"    回答（最初の200文字）: {result.answer[:200]}...")
            
        except Exception as e:
            print(f"    エラー: {e}")
    
    # 5. MoE-RAGエンドポイントのテスト
    print("\n\n5. MoE-RAGエンドポイントのテスト...")
    
    import requests
    
    # エキスパート一覧取得
    print("\n  エキスパート一覧を取得中...")
    try:
        response = requests.get("http://localhost:8050/api/moe-rag/experts")
        if response.status_code == 200:
            data = response.json()
            print(f"    利用可能なエキスパート数: {data['total']}")
            for expert in data['experts'][:3]:
                print(f"      - {expert['name']}: {expert['description']}")
        else:
            print(f"    エラー: ステータスコード {response.status_code}")
    except Exception as e:
        print(f"    エラー: {e}")
    
    # クエリ分析
    print("\n  クエリ分析を実行中...")
    try:
        response = requests.post(
            "http://localhost:8050/api/moe-rag/analyze",
            params={"query": "道路の設計速度について教えてください"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"    主要エキスパート: {', '.join(data['primary_experts'])}")
            print(f"    信頼度: {data['confidence']:.2f}")
        else:
            print(f"    エラー: ステータスコード {response.status_code}")
    except Exception as e:
        print(f"    エラー: {e}")
    
    # ハイブリッド検索
    print("\n  MoE-RAGハイブリッド検索を実行中...")
    try:
        response = requests.post(
            "http://localhost:8050/api/moe-rag/query",
            params={
                "query": "橋梁の耐震設計基準",
                "top_k": 3,
                "use_moe": True
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"    選択されたエキスパート: {', '.join(data['selected_experts'])}")
            print(f"    信頼度: {data['confidence']:.2f}")
            print(f"    取得ドキュメント数: {len(data['documents'])}")
        else:
            print(f"    エラー: ステータスコード {response.status_code}")
    except Exception as e:
        print(f"    エラー: {e}")
    
    print("\n" + "=" * 60)
    print("MoE-RAG統合テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_moe_integration()
    except Exception as e:
        print(f"\nテスト失敗: {e}")
        import traceback
        traceback.print_exc()