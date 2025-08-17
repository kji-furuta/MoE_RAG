#!/usr/bin/env python3
"""
MoE-RAG統合システムテストスクリプト
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, Any
import requests

# パス追加
sys.path.append(str(Path(__file__).parent.parent))

# MoE-RAG統合モジュール
from src.moe_rag_integration.hybrid_query_engine import HybridMoERAGEngine
from src.moe_rag_integration.expert_router import ExpertRouter
from src.moe_rag_integration.response_fusion import ResponseFusion


class MoERAGIntegrationTester:
    """MoE-RAG統合テスター"""
    
    def __init__(self):
        self.base_url = "http://localhost:8050"
        self.test_results = []
        
    async def test_components(self):
        """コンポーネント単体テスト"""
        print("\n" + "="*50)
        print("🧪 コンポーネント単体テスト")
        print("="*50)
        
        # 1. エキスパートルーターテスト
        print("\n1. エキスパートルーター")
        try:
            router = ExpertRouter()
            
            test_queries = [
                "設計速度80km/hの道路における最小曲線半径は？",
                "橋梁の耐震設計について教えてください",
                "コンクリートの配合設計の基本",
                "トンネル掘削時の注意点"
            ]
            
            for query in test_queries:
                decision = router.route(query)
                print(f"  Query: {query[:30]}...")
                print(f"  → Primary: {decision.primary_experts}")
                print(f"  → Strategy: {decision.routing_strategy}")
                print(f"  → Confidence: {decision.confidence:.2f}")
            
            self.test_results.append(("Expert Router", "PASS"))
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            self.test_results.append(("Expert Router", f"FAIL: {e}"))
        
        # 2. レスポンス融合テスト
        print("\n2. レスポンス融合")
        try:
            fusion = ResponseFusion(fusion_strategy="adaptive")
            
            moe_response = "道路構造令により、設計速度80km/hの最小曲線半径は280mです。"
            rag_response = "設計速度80km/hでは最小曲線半径280m、特例値230mが適用可能です。"
            
            result = fusion.fuse(
                moe_response=moe_response,
                rag_response=rag_response,
                moe_confidence=0.85,
                rag_confidence=0.75,
                query_type="technical"
            )
            
            print(f"  Fusion Method: {result.fusion_method}")
            print(f"  Quality Score: {result.quality_score:.2f}")
            print(f"  MoE Contribution: {result.source_contributions.get('moe', 0):.2f}")
            print(f"  RAG Contribution: {result.source_contributions.get('rag', 0):.2f}")
            
            self.test_results.append(("Response Fusion", "PASS"))
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            self.test_results.append(("Response Fusion", f"FAIL: {e}"))
    
    async def test_integration(self):
        """統合テスト"""
        print("\n" + "="*50)
        print("🔗 統合テスト")
        print("="*50)
        
        try:
            # ハイブリッドエンジンの初期化
            print("\nハイブリッドエンジン初期化中...")
            engine = HybridMoERAGEngine(
                moe_weight=0.4,
                rag_weight=0.6
            )
            
            # テストクエリ
            test_cases = [
                {
                    "query": "設計速度60km/hの道路の最小曲線半径について、地形条件を考慮した設計方法を教えてください",
                    "expected_experts": ["道路設計", "法規・基準"],
                    "top_k": 3
                },
                {
                    "query": "橋梁基礎の設計における地盤調査の重要性と液状化対策",
                    "expected_experts": ["構造設計", "地盤工学"],
                    "top_k": 5
                },
                {
                    "query": "高強度コンクリートの配合設計と施工時の品質管理方法",
                    "expected_experts": ["材料工学", "施工管理"],
                    "top_k": 3
                }
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\nテストケース {i}:")
                print(f"  Query: {test_case['query'][:50]}...")
                
                start_time = time.time()
                
                # クエリ実行
                result = await engine.query(
                    query=test_case['query'],
                    top_k=test_case['top_k'],
                    use_reranking=True
                )
                
                elapsed_time = time.time() - start_time
                
                # 結果検証
                print(f"  処理時間: {elapsed_time:.2f}秒")
                print(f"  選択エキスパート: {result.moe_result.selected_experts}")
                print(f"  期待エキスパート: {test_case['expected_experts']}")
                print(f"  信頼度: {result.confidence_score:.2f}")
                print(f"  取得文書数: {len(result.rag_documents)}")
                
                # 期待エキスパートとの一致確認
                matched = any(
                    expert in result.moe_result.selected_experts 
                    for expert in test_case['expected_experts']
                )
                
                if matched:
                    print(f"  ✅ エキスパート選択: 成功")
                    self.test_results.append((f"Integration Test {i}", "PASS"))
                else:
                    print(f"  ⚠️  エキスパート選択: 不一致")
                    self.test_results.append((f"Integration Test {i}", "PARTIAL"))
                
                # 回答の品質チェック
                if result.answer and len(result.answer) > 50:
                    print(f"  ✅ 回答生成: 成功")
                else:
                    print(f"  ❌ 回答生成: 不十分")
                
        except Exception as e:
            print(f"\n❌ 統合テストエラー: {e}")
            self.test_results.append(("Integration", f"FAIL: {e}"))
    
    async def test_api(self):
        """API統合テスト"""
        print("\n" + "="*50)
        print("🌐 API統合テスト")
        print("="*50)
        
        # APIヘルスチェック
        print("\n1. ヘルスチェック")
        try:
            response = requests.get(f"{self.base_url}/api/moe-rag/health")
            if response.status_code == 200:
                data = response.json()
                print(f"  Status: {data['status']}")
                print(f"  Components: {data.get('components', {})}")
                self.test_results.append(("API Health", "PASS"))
            else:
                print(f"  ❌ ステータスコード: {response.status_code}")
                self.test_results.append(("API Health", f"FAIL: {response.status_code}"))
        except Exception as e:
            print(f"  ❌ 接続エラー: {e}")
            print("  → APIサーバーが起動していない可能性があります")
            self.test_results.append(("API Health", f"FAIL: {e}"))
            return
        
        # クエリ実行テスト
        print("\n2. クエリ実行")
        try:
            payload = {
                "query": "道路設計における視距の確保方法",
                "top_k": 5,
                "use_reranking": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/moe-rag/query",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Query ID: {data.get('query_id', 'N/A')}")
                print(f"  Confidence: {data.get('confidence_score', 0):.2f}")
                print(f"  Experts: {data.get('selected_experts', [])}")
                print(f"  Processing Time: {data.get('processing_time', 0):.2f}秒")
                self.test_results.append(("API Query", "PASS"))
            else:
                print(f"  ❌ ステータスコード: {response.status_code}")
                self.test_results.append(("API Query", f"FAIL: {response.status_code}"))
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            self.test_results.append(("API Query", f"FAIL: {e}"))
        
        # エキスパート情報取得
        print("\n3. エキスパート情報")
        try:
            response = requests.get(f"{self.base_url}/api/moe-rag/experts")
            if response.status_code == 200:
                data = response.json()
                print(f"  Total Experts: {data.get('total_experts', 0)}")
                print(f"  Active: {len(data.get('active_experts', []))}")
                self.test_results.append(("API Experts", "PASS"))
            else:
                print(f"  ❌ ステータスコード: {response.status_code}")
                self.test_results.append(("API Experts", f"FAIL: {response.status_code}"))
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            self.test_results.append(("API Experts", f"FAIL: {e}"))
    
    def print_summary(self):
        """テスト結果サマリー表示"""
        print("\n" + "="*50)
        print("📊 テスト結果サマリー")
        print("="*50)
        
        passed = sum(1 for _, status in self.test_results if status == "PASS")
        partial = sum(1 for _, status in self.test_results if status == "PARTIAL")
        failed = sum(1 for _, status in self.test_results if "FAIL" in status)
        total = len(self.test_results)
        
        print(f"\n合計: {total}件")
        print(f"✅ 成功: {passed}件")
        print(f"⚠️  部分成功: {partial}件")
        print(f"❌ 失敗: {failed}件")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"\n成功率: {success_rate:.1f}%")
        
        print("\n詳細:")
        for test_name, status in self.test_results:
            icon = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
            print(f"  {icon} {test_name}: {status}")
        
        # 推奨事項
        print("\n" + "="*50)
        print("💡 推奨事項")
        print("="*50)
        
        if failed > 0:
            print("• 失敗したテストの詳細ログを確認してください")
            print("• 必要な依存関係がインストールされているか確認してください")
            print("• APIサーバーが正しく起動しているか確認してください")
        
        if partial > 0:
            print("• エキスパート選択の精度向上のため、ルーティングロジックの調整を検討してください")
        
        if passed == total:
            print("✨ すべてのテストが成功しました！")
            print("• 本番環境への展開準備が整いました")


async def main():
    """メイン実行関数"""
    print("\n" + "="*50)
    print("🚀 MoE-RAG統合システムテスト開始")
    print("="*50)
    
    tester = MoERAGIntegrationTester()
    
    # コンポーネントテスト
    await tester.test_components()
    
    # 統合テスト
    await tester.test_integration()
    
    # APIテスト（サーバー起動時のみ）
    print("\nAPIテストを実行しますか？（サーバーが起動している必要があります）")
    print("実行する場合は5秒以内にCtrl+Cで中断してください...")
    
    try:
        await asyncio.sleep(5)
        await tester.test_api()
    except KeyboardInterrupt:
        print("\nAPIテストをスキップしました")
    
    # サマリー表示
    tester.print_summary()
    
    print("\n" + "="*50)
    print("✅ テスト完了")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())