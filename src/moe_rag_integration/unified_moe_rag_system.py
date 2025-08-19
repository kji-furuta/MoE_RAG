"""
統合MoE-RAGシステム
MoEエキスパート選択とRAG検索を完全に統合したシステム
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import logging
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

# RAGシステムのインポート
try:
    from src.rag.core.query_engine import QueryEngine
    from src.rag.retrieval.hybrid_search import HybridSearcher
    from src.rag.indexing.vector_store import VectorStore
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    QueryEngine = None
    HybridSearcher = None
    VectorStore = None

# MoEシステムのインポート
from src.moe.moe_architecture import MoEConfig, MoELayer
from src.moe_rag_integration.expert_router import ExpertRouter, RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class UnifiedQueryResult:
    """統合クエリ結果"""
    answer: str
    selected_experts: List[str]
    expert_scores: Dict[str, float]
    retrieved_documents: List[Dict[str, Any]]
    vector_similarity_scores: List[float]
    keyword_matches: List[str]
    fusion_strategy: str
    confidence: float
    metadata: Dict[str, Any]


class UnifiedMoERAGSystem:
    """
    完全統合型MoE-RAGシステム
    
    特徴:
    1. RAGのベクトル検索結果を使用してエキスパートを選択
    2. エキスパート特化のドキュメントフィルタリング
    3. エキスパートごとのベクトル埋め込み空間
    4. 動的な重み調整による結果融合
    """
    
    def __init__(
        self,
        rag_config_path: str = "config/rag_config.yaml",
        moe_config: Optional[MoEConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            rag_config_path: RAG設定ファイルパス
            moe_config: MoE設定
            device: 実行デバイス
        """
        self.device = device
        
        # MoE設定
        if moe_config is None:
            moe_config = MoEConfig(
                num_experts=8,
                num_experts_per_tok=2,
                hidden_size=768,
                domain_specific_routing=True
            )
        self.moe_config = moe_config
        
        # エキスパートルーターの初期化
        self.expert_router = ExpertRouter(
            num_experts_per_tok=moe_config.num_experts_per_tok
        )
        
        # RAGシステムの初期化
        if RAG_AVAILABLE:
            self.query_engine = QueryEngine(config_path=rag_config_path)
            self.vector_store = VectorStore()
            self.hybrid_searcher = HybridSearcher(self.vector_store)
            logger.info("RAG system initialized successfully")
        else:
            self.query_engine = None
            self.vector_store = None
            self.hybrid_searcher = None
            logger.warning("RAG system not available - using fallback mode")
        
        # エキスパート別のドキュメントタグマッピング
        self.expert_document_tags = {
            "構造設計": ["structural", "耐震", "荷重計算", "構造解析"],
            "道路設計": ["road", "道路構造令", "線形設計", "交差点設計"],
            "地盤工学": ["geotechnical", "土質", "地盤改良", "支持力"],
            "水理・排水": ["hydraulics", "排水", "雨水処理", "流量計算"],
            "材料工学": ["materials", "コンクリート", "アスファルト", "材料試験"],
            "施工管理": ["construction", "施工計画", "品質管理", "工程管理"],
            "法規・基準": ["regulations", "基準", "仕様書", "法令"],
            "環境・維持管理": ["environmental", "維持管理", "環境影響", "点検"]
        }
        
        # エキスパート別埋め込みモデル（将来的な拡張用）
        self.expert_embeddings = {}
        
        logger.info(f"Unified MoE-RAG System initialized on {device}")
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        expert_override: Optional[List[str]] = None
    ) -> UnifiedQueryResult:
        """
        統合クエリ処理
        
        Args:
            query: 検索クエリ
            top_k: 取得ドキュメント数
            use_reranking: リランキングを使用するか
            expert_override: エキスパートの手動指定
            
        Returns:
            統合クエリ結果
        """
        # 1. RAG検索を実行してコンテキストを取得
        rag_results = await self._execute_rag_search(query, top_k * 2)  # 多めに取得
        
        # 2. RAG結果を使用してエキスパートを選択
        if expert_override:
            selected_experts = expert_override
            routing_decision = None
        else:
            routing_decision = await self._select_experts_with_rag_context(
                query, rag_results
            )
            selected_experts = routing_decision.primary_experts
        
        logger.info(f"Selected experts: {selected_experts}")
        
        # 3. エキスパート特化のドキュメントフィルタリング
        expert_documents = self._filter_documents_by_experts(
            rag_results.get('documents', []),
            selected_experts
        )
        
        # 4. エキスパート別のクエリ拡張
        expanded_queries = self._expand_query_for_experts(query, selected_experts)
        
        # 5. エキスパート特化の追加検索（必要に応じて）
        if len(expert_documents) < top_k:
            additional_docs = await self._search_expert_specific_documents(
                expanded_queries, selected_experts, top_k - len(expert_documents)
            )
            expert_documents.extend(additional_docs)
        
        # 6. 結果の生成と統合
        answer = await self._generate_unified_answer(
            query, expert_documents, selected_experts
        )
        
        # 7. メタデータの構築
        metadata = self._build_metadata(
            query, rag_results, routing_decision, selected_experts
        )
        
        return UnifiedQueryResult(
            answer=answer,
            selected_experts=selected_experts,
            expert_scores=routing_decision.expert_scores if routing_decision else {},
            retrieved_documents=expert_documents[:top_k],
            vector_similarity_scores=self._extract_similarity_scores(expert_documents),
            keyword_matches=self._extract_keyword_matches(expert_documents),
            fusion_strategy=metadata.get('fusion_strategy', 'unified'),
            confidence=self._calculate_confidence(expert_documents, selected_experts),
            metadata=metadata
        )
    
    async def _execute_rag_search(
        self, 
        query: str, 
        top_k: int
    ) -> Dict[str, Any]:
        """RAG検索を実行"""
        if not RAG_AVAILABLE or self.query_engine is None:
            # フォールバック: ダミーデータを返す
            return {
                'documents': [
                    {
                        'content': f"ダミードキュメント {i+1}",
                        'metadata': {'source': 'dummy', 'score': 0.5},
                        'tags': []
                    }
                    for i in range(top_k)
                ],
                'scores': [0.5] * top_k
            }
        
        try:
            # 実際のRAG検索
            result = self.query_engine.search(
                query=query,
                top_k=top_k,
                use_hybrid=True  # ベクトル検索とキーワード検索の両方を使用
            )
            
            return {
                'documents': result.documents,
                'scores': result.scores,
                'metadata': result.metadata
            }
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return {'documents': [], 'scores': []}
    
    async def _select_experts_with_rag_context(
        self,
        query: str,
        rag_results: Dict[str, Any]
    ) -> RoutingDecision:
        """RAGコンテキストを使用してエキスパートを選択"""
        
        # 基本的なルーティング決定
        base_routing = self.expert_router.route(query)
        
        # RAGドキュメントからエキスパート関連度を計算
        document_expert_scores = self._analyze_documents_for_experts(
            rag_results.get('documents', [])
        )
        
        # スコアの統合
        combined_scores = {}
        for expert in self.expert_router.expert_domains.keys():
            base_score = base_routing.expert_scores.get(expert, 0.0)
            doc_score = document_expert_scores.get(expert, 0.0)
            
            # 重み付き平均（RAG結果を重視）
            combined_scores[expert] = 0.4 * base_score + 0.6 * doc_score
        
        # Top-Kエキスパートの選択
        sorted_experts = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        primary_experts = [
            expert for expert, _ in sorted_experts[:self.moe_config.num_experts_per_tok]
        ]
        secondary_experts = [
            expert for expert, _ in sorted_experts[
                self.moe_config.num_experts_per_tok:self.moe_config.num_experts_per_tok + 2
            ]
        ]
        
        return RoutingDecision(
            primary_experts=primary_experts,
            secondary_experts=secondary_experts,
            expert_scores=combined_scores,
            routing_strategy="rag_enhanced",
            confidence=np.mean([score for _, score in sorted_experts[:2]]),
            keywords_detected=base_routing.keywords_detected
        )
    
    def _analyze_documents_for_experts(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """ドキュメントを分析してエキスパート関連度を計算"""
        expert_scores = {expert: 0.0 for expert in self.expert_document_tags.keys()}
        
        if not documents:
            return expert_scores
        
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_tags = doc.get('tags', [])
            doc_metadata = doc.get('metadata', {})
            
            for expert, tags in self.expert_document_tags.items():
                score = 0.0
                
                # タグマッチング
                for tag in tags:
                    if tag in doc_tags:
                        score += 2.0  # タグの完全一致は高スコア
                    elif tag.lower() in content:
                        score += 1.0  # コンテンツ内のキーワードマッチ
                
                # メタデータからのカテゴリマッチング
                if doc_metadata.get('category') == expert:
                    score += 3.0
                
                # 正規化してスコアに追加
                expert_scores[expert] += score / (len(documents) * 5.0)  # 最大5点で正規化
        
        return expert_scores
    
    def _filter_documents_by_experts(
        self,
        documents: List[Dict[str, Any]],
        selected_experts: List[str]
    ) -> List[Dict[str, Any]]:
        """選択されたエキスパートに関連するドキュメントをフィルタリング"""
        filtered_docs = []
        expert_tags = set()
        
        # 選択されたエキスパートのタグを収集
        for expert in selected_experts:
            expert_tags.update(self.expert_document_tags.get(expert, []))
        
        for doc in documents:
            doc_content = doc.get('content', '').lower()
            doc_tags = set(doc.get('tags', []))
            doc_category = doc.get('metadata', {}).get('category', '')
            
            # エキスパートタグとの関連性をチェック
            relevance_score = 0
            for tag in expert_tags:
                if tag in doc_tags or tag.lower() in doc_content:
                    relevance_score += 1
            
            if doc_category in selected_experts:
                relevance_score += 3
            
            # 関連性がある場合は追加
            if relevance_score > 0:
                doc['expert_relevance_score'] = relevance_score
                filtered_docs.append(doc)
        
        # 関連性スコアでソート
        filtered_docs.sort(key=lambda x: x.get('expert_relevance_score', 0), reverse=True)
        
        return filtered_docs
    
    def _expand_query_for_experts(
        self,
        query: str,
        selected_experts: List[str]
    ) -> Dict[str, str]:
        """エキスパート別にクエリを拡張"""
        expanded_queries = {}
        
        for expert in selected_experts:
            # エキスパート特有のキーワードを追加
            expert_keywords = self.expert_router.expert_domains.get(
                expert, {}
            ).get('keywords', [])
            
            # クエリ拡張
            expanded_query = f"{query} {' '.join(expert_keywords[:3])}"
            expanded_queries[expert] = expanded_query
        
        return expanded_queries
    
    async def _search_expert_specific_documents(
        self,
        expanded_queries: Dict[str, str],
        selected_experts: List[str],
        num_docs: int
    ) -> List[Dict[str, Any]]:
        """エキスパート特化の追加ドキュメント検索"""
        additional_docs = []
        
        if not RAG_AVAILABLE or self.hybrid_searcher is None:
            return additional_docs
        
        docs_per_expert = max(1, num_docs // len(selected_experts))
        
        for expert, expanded_query in expanded_queries.items():
            try:
                # エキスパート特化検索
                result = await self._execute_rag_search(
                    expanded_query, 
                    docs_per_expert
                )
                
                # エキスパートタグを追加
                for doc in result.get('documents', []):
                    doc['expert'] = expert
                    additional_docs.append(doc)
                    
            except Exception as e:
                logger.error(f"Expert-specific search error for {expert}: {e}")
        
        return additional_docs
    
    async def _generate_unified_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        selected_experts: List[str]
    ) -> str:
        """統合回答の生成"""
        
        # ドキュメントからコンテキストを構築
        context = self._build_context_from_documents(documents[:5])
        
        # エキスパート情報を含めた回答生成
        expert_intro = f"【専門分野】{', '.join(selected_experts)}\n\n"
        
        # 基本的な回答テンプレート
        answer = f"""{expert_intro}【回答】
{query}について、以下の情報をもとに回答いたします。

{context}

【要約】
選択されたエキスパート（{', '.join(selected_experts)}）の知見に基づき、
上記の情報から総合的に判断した結果をお伝えしました。

詳細な技術情報が必要な場合は、関連する基準書や仕様書をご参照ください。
"""
        
        return answer
    
    def _build_context_from_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> str:
        """ドキュメントからコンテキストを構築"""
        if not documents:
            return "関連する情報が見つかりませんでした。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')[:500]  # 最初の500文字
            source = doc.get('metadata', {}).get('source', '不明')
            expert = doc.get('expert', '一般')
            
            context_parts.append(
                f"{i}. [{expert}専門知識] {content}\n   出典: {source}"
            )
        
        return "\n\n".join(context_parts)
    
    def _extract_similarity_scores(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[float]:
        """ドキュメントから類似度スコアを抽出"""
        scores = []
        for doc in documents:
            score = doc.get('metadata', {}).get('similarity_score', 0.0)
            scores.append(score)
        return scores
    
    def _extract_keyword_matches(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """ドキュメントからキーワードマッチを抽出"""
        keywords = set()
        for doc in documents:
            doc_keywords = doc.get('metadata', {}).get('matched_keywords', [])
            keywords.update(doc_keywords)
        return list(keywords)
    
    def _calculate_confidence(
        self,
        documents: List[Dict[str, Any]],
        selected_experts: List[str]
    ) -> float:
        """信頼度スコアを計算"""
        if not documents:
            return 0.0
        
        # ドキュメントの関連性スコアの平均
        doc_scores = [
            doc.get('expert_relevance_score', 0.0) 
            for doc in documents
        ]
        avg_doc_score = np.mean(doc_scores) if doc_scores else 0.0
        
        # エキスパート数による信頼度調整
        expert_factor = min(1.0, len(selected_experts) / 3.0)
        
        # 総合信頼度
        confidence = min(1.0, avg_doc_score * 0.7 + expert_factor * 0.3)
        
        return confidence
    
    def _build_metadata(
        self,
        query: str,
        rag_results: Dict[str, Any],
        routing_decision: Optional[RoutingDecision],
        selected_experts: List[str]
    ) -> Dict[str, Any]:
        """メタデータを構築"""
        metadata = {
            'query': query,
            'selected_experts': selected_experts,
            'num_documents': len(rag_results.get('documents', [])),
            'fusion_strategy': 'unified',
            'rag_available': RAG_AVAILABLE
        }
        
        if routing_decision:
            metadata.update({
                'routing_strategy': routing_decision.routing_strategy,
                'routing_confidence': routing_decision.confidence,
                'keywords_detected': routing_decision.keywords_detected
            })
        
        if rag_results:
            metadata['rag_metadata'] = rag_results.get('metadata', {})
        
        return metadata


# 使用例
async def main():
    """テスト用メイン関数"""
    system = UnifiedMoERAGSystem()
    
    # テストクエリ
    test_queries = [
        "設計速度80km/hの道路の最小曲線半径は？",
        "橋梁の耐震設計における照査内容について教えてください",
        "アスファルト舗装の品質管理基準を教えてください"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = await system.query(query, top_k=5)
        
        print(f"Selected Experts: {', '.join(result.selected_experts)}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Retrieved Documents: {len(result.retrieved_documents)}")
        print(f"\nAnswer:\n{result.answer[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())