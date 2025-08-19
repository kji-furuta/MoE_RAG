"""
MoE-RAGハイブリッドクエリエンジン
MoEエキスパート推論とRAG検索を統合したハイブリッド検索エンジン
"""

import torch
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys
import os
import logging

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MoEモジュール
from .moe_serving import MoEModelServer, MoEInferenceResult

# RAGモジュール（簡易実装）
# from rag.core.query_engine import QueryEngine
# from rag.core.types import QueryResult, Document
# from rag.retrieval.hybrid_search import HybridSearcher
# from rag.generation.llm_generator import LLMGenerator

# 簡易的なダミークラス（実際のRAGが利用できない場合）
@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    
    def dict(self):
        return {
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score
        }

@dataclass
class QueryResult:
    answer: str
    documents: List[Document]
    metadata: Dict[str, Any]

class QueryEngine:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        
    def query(self, query: str, top_k: int = 5, use_reranking: bool = True) -> QueryResult:
        # ダミー実装
        return QueryResult(
            answer=f"RAGシステムからの回答: {query}に関する情報です。",
            documents=[
                Document(
                    content=f"文書{i}: {query}に関連する内容",
                    metadata={'filename': f'doc_{i}.pdf', 'page': i},
                    score=0.9 - i*0.1
                ) for i in range(min(top_k, 3))
            ],
            metadata={'confidence': 0.75}
        )

logger = logging.getLogger(__name__)


@dataclass
class HybridQueryResult:
    """ハイブリッドクエリ結果"""
    answer: str
    rag_documents: List[Document]
    moe_result: MoEInferenceResult
    confidence_score: float
    expert_contributions: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer': self.answer,
            'rag_documents': [doc.dict() for doc in self.rag_documents],
            'moe_result': self.moe_result.to_dict(),
            'confidence_score': self.confidence_score,
            'expert_contributions': self.expert_contributions,
            'metadata': self.metadata
        }


class HybridMoERAGEngine:
    """MoE-RAGハイブリッドクエリエンジン"""
    
    def __init__(
        self,
        moe_model_path: Optional[str] = None,
        rag_config_path: str = "src/rag/config/rag_config.yaml",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        moe_weight: float = 0.4,
        rag_weight: float = 0.6
    ):
        """
        Args:
            moe_model_path: MoEモデルのパス
            rag_config_path: RAG設定ファイルのパス
            device: 実行デバイス
            moe_weight: MoE結果の重み
            rag_weight: RAG結果の重み
        """
        self.device = device
        self.moe_weight = moe_weight
        self.rag_weight = rag_weight
        
        # MoEサーバーの初期化
        logger.info("Initializing MoE Model Server...")
        self.moe_server = MoEModelServer(
            model_path=moe_model_path,
            device=device
        )
        
        # RAGエンジンの初期化
        logger.info("Initializing RAG Query Engine...")
        self.rag_engine = QueryEngine(config_path=rag_config_path)
        
        # エキスパート専門分野のマッピング
        self.expert_domains = {
            "構造設計": ["構造", "荷重", "応力", "変形", "耐震", "設計"],
            "道路設計": ["道路", "曲線", "勾配", "視距", "設計速度", "交差点"],
            "地盤工学": ["地盤", "土質", "支持力", "沈下", "液状化", "地質"],
            "水理・排水": ["水理", "排水", "流量", "管渠", "雨水", "下水"],
            "材料工学": ["材料", "コンクリート", "鋼材", "強度", "耐久性"],
            "施工管理": ["施工", "工程", "品質", "安全", "管理", "検査"],
            "法規・基準": ["法規", "基準", "規格", "仕様", "規定"],
            "環境・維持管理": ["環境", "維持", "管理", "点検", "補修"]
        }
        
        logger.info(f"Hybrid MoE-RAG Engine initialized on {device}")
        logger.info(f"Weights: MoE={moe_weight}, RAG={rag_weight}")
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        stream: bool = False
    ) -> HybridQueryResult:
        """
        ハイブリッドクエリを実行
        
        Args:
            query: クエリテキスト
            top_k: 取得する文書数
            use_reranking: リランキングを使用するか
            stream: ストリーミングモードか
            
        Returns:
            ハイブリッドクエリ結果
        """
        # 1. エキスパート選択
        selected_experts, expert_scores = self.moe_server.get_expert_for_query(query)
        logger.info(f"Selected experts: {selected_experts}")
        
        # 2. 並列実行: MoE推論とRAG検索
        moe_task = self._run_moe_inference(query, selected_experts)
        rag_task = self._run_rag_search(query, top_k, use_reranking)
        
        moe_result, rag_result = await asyncio.gather(moe_task, rag_task)
        
        # 3. 結果の統合
        hybrid_answer = await self._fuse_results(
            query, moe_result, rag_result, selected_experts
        )
        
        # 4. 信頼度スコアの計算
        confidence_score = self._calculate_hybrid_confidence(
            moe_result.confidence,
            rag_result.metadata.get('confidence', 0.5)
        )
        
        # 5. エキスパート貢献度の計算
        expert_contributions = self._calculate_expert_contributions(
            moe_result.expert_scores,
            rag_result.documents,
            selected_experts
        )
        
        return HybridQueryResult(
            answer=hybrid_answer,
            rag_documents=rag_result.documents,
            moe_result=moe_result,
            confidence_score=confidence_score,
            expert_contributions=expert_contributions,
            metadata={
                'selected_experts': selected_experts,
                'moe_weight': self.moe_weight,
                'rag_weight': self.rag_weight,
                'rag_metadata': rag_result.metadata,
                'query_type': self._classify_query(query)
            }
        )
    
    async def _run_moe_inference(
        self,
        query: str,
        selected_experts: List[str]
    ) -> MoEInferenceResult:
        """MoE推論を実行"""
        try:
            # 非同期でMoE推論を実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.moe_server.infer,
                query,
                512,  # max_length
                True,  # return_hidden_states
                0.7   # temperature
            )
            return result
        except Exception as e:
            logger.error(f"MoE inference error: {e}")
            # フォールバック結果を返す
            return MoEInferenceResult(
                text="MoE推論エラー",
                expert_scores={},
                selected_experts=selected_experts,
                confidence=0.0
            )
    
    async def _run_rag_search(
        self,
        query: str,
        top_k: int,
        use_reranking: bool
    ) -> QueryResult:
        """RAG検索を実行"""
        try:
            # 非同期でRAG検索を実行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.rag_engine.query,
                query,
                top_k,
                use_reranking
            )
            return result
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            # フォールバック結果を返す
            return QueryResult(
                answer="RAG検索エラー",
                documents=[],
                metadata={'error': str(e)}
            )
    
    async def _fuse_results(
        self,
        query: str,
        moe_result: MoEInferenceResult,
        rag_result: QueryResult,
        selected_experts: List[str]
    ) -> str:
        """
        MoEとRAGの結果を統合
        
        Args:
            query: 元のクエリ
            moe_result: MoE推論結果
            rag_result: RAG検索結果
            selected_experts: 選択されたエキスパート
            
        Returns:
            統合された回答
        """
        # 基本的な統合戦略
        fusion_strategy = self._determine_fusion_strategy(
            query, moe_result, rag_result
        )
        
        if fusion_strategy == "moe_dominant":
            # MoEの専門知識を優先
            base_answer = moe_result.text
            supporting_info = self._extract_supporting_info(rag_result.documents)
            
            return f"""
{base_answer}

【参考情報】
{supporting_info}

【専門エキスパート】
{', '.join(selected_experts)}
"""
        
        elif fusion_strategy == "rag_dominant":
            # RAGの検索結果を優先
            base_answer = rag_result.answer
            expert_insight = moe_result.text if moe_result.confidence > 0.3 else ""
            
            return f"""
{base_answer}

{f'【専門家の見解】{expert_insight}' if expert_insight else ''}

【情報源】
{self._format_sources(rag_result.documents[:3])}
"""
        
        else:  # balanced
            # バランスの取れた統合
            combined_context = self._create_combined_context(
                moe_result, rag_result
            )
            
            # LLMを使用して最終的な回答を生成
            final_answer = await self._generate_unified_answer(
                query, combined_context, selected_experts
            )
            
            return final_answer
    
    def _determine_fusion_strategy(
        self,
        query: str,
        moe_result: MoEInferenceResult,
        rag_result: QueryResult
    ) -> str:
        """結果統合戦略を決定"""
        # 信頼度に基づく戦略選択
        moe_confidence = moe_result.confidence
        rag_confidence = rag_result.metadata.get('confidence', 0.5)
        
        # クエリタイプの分析
        query_type = self._classify_query(query)
        
        if query_type == "specialized" and moe_confidence > 0.7:
            return "moe_dominant"
        elif query_type == "factual" and rag_confidence > 0.7:
            return "rag_dominant"
        else:
            return "balanced"
    
    def _classify_query(self, query: str) -> str:
        """クエリのタイプを分類"""
        # 専門用語の検出
        specialized_keywords = [
            "設計", "施工", "構造", "地盤", "材料",
            "基準", "規格", "計算", "解析"
        ]
        
        factual_keywords = [
            "とは", "定義", "種類", "一覧", "数値",
            "仕様", "規定値"
        ]
        
        query_lower = query.lower()
        
        specialized_count = sum(
            1 for kw in specialized_keywords if kw in query_lower
        )
        factual_count = sum(
            1 for kw in factual_keywords if kw in query_lower
        )
        
        if specialized_count > factual_count:
            return "specialized"
        elif factual_count > specialized_count:
            return "factual"
        else:
            return "general"
    
    def _calculate_hybrid_confidence(
        self,
        moe_confidence: float,
        rag_confidence: float
    ) -> float:
        """ハイブリッド信頼度スコアを計算"""
        # 重み付け平均
        weighted_confidence = (
            self.moe_weight * moe_confidence +
            self.rag_weight * rag_confidence
        )
        
        # 両方の信頼度が高い場合はボーナス
        if moe_confidence > 0.7 and rag_confidence > 0.7:
            weighted_confidence = min(weighted_confidence * 1.1, 1.0)
        
        return float(weighted_confidence)
    
    def _calculate_expert_contributions(
        self,
        expert_scores: Dict[str, float],
        documents: List[Document],
        selected_experts: List[str]
    ) -> Dict[str, float]:
        """エキスパートの貢献度を計算"""
        contributions = {expert: 0.0 for expert in selected_experts}
        
        # MoEスコアから初期貢献度
        for expert in selected_experts:
            if expert in expert_scores:
                contributions[expert] = expert_scores[expert] * self.moe_weight
        
        # 文書との関連性から追加貢献度
        for doc in documents:
            doc_text = doc.content.lower()
            for expert, keywords in self.expert_domains.items():
                if expert in contributions:
                    keyword_matches = sum(
                        1 for kw in keywords if kw in doc_text
                    )
                    if keyword_matches > 0:
                        contributions[expert] += (
                            (keyword_matches / len(keywords)) *
                            self.rag_weight / len(documents)
                        )
        
        # 正規化
        total = sum(contributions.values())
        if total > 0:
            contributions = {
                k: v / total for k, v in contributions.items()
            }
        
        return contributions
    
    def _extract_supporting_info(self, documents: List[Document]) -> str:
        """文書から補足情報を抽出"""
        if not documents:
            return "関連文書が見つかりませんでした。"
        
        info_parts = []
        for i, doc in enumerate(documents[:3], 1):
            # 重要な部分を抽出（最初の200文字）
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            info_parts.append(f"{i}. {snippet}")
        
        return "\n".join(info_parts)
    
    def _format_sources(self, documents: List[Document]) -> str:
        """文書ソースをフォーマット"""
        if not documents:
            return "なし"
        
        sources = []
        for doc in documents:
            source = f"- {doc.metadata.get('filename', '不明')}"
            if 'page' in doc.metadata:
                source += f" (p.{doc.metadata['page']})"
            sources.append(source)
        
        return "\n".join(sources)
    
    def _create_combined_context(
        self,
        moe_result: MoEInferenceResult,
        rag_result: QueryResult
    ) -> str:
        """統合コンテキストを作成"""
        context_parts = [
            "【MoEエキスパート分析】",
            moe_result.text,
            "",
            "【RAG検索結果】",
            rag_result.answer,
            "",
            "【関連文書】"
        ]
        
        for doc in rag_result.documents[:3]:
            context_parts.append(f"- {doc.content[:150]}...")
        
        return "\n".join(context_parts)
    
    async def _generate_unified_answer(
        self,
        query: str,
        combined_context: str,
        selected_experts: List[str]
    ) -> str:
        """統合された最終回答を生成"""
        try:
            # LLMGeneratorを使用して回答生成
            prompt = f"""
以下のコンテキストを基に、質問に対する包括的な回答を生成してください。

質問: {query}

選択されたエキスパート: {', '.join(selected_experts)}

コンテキスト:
{combined_context}

回答:
"""
            
            # ここでは簡易的な実装
            # 実際にはLLMGeneratorを使用
            return f"""
{combined_context}

【統合回答】
質問「{query}」について、{', '.join(selected_experts)}の専門知識とRAG検索結果を統合して回答しました。
"""
        
        except Exception as e:
            logger.error(f"Failed to generate unified answer: {e}")
            return combined_context
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        return {
            'moe_info': self.moe_server.get_model_info(),
            'rag_info': {
                'vector_store': 'Qdrant',
                'embedding_model': 'multilingual-e5-large',
                'hybrid_search': True
            },
            'integration_config': {
                'moe_weight': self.moe_weight,
                'rag_weight': self.rag_weight,
                'device': str(self.device)
            }
        }


# 使用例
async def main():
    # ハイブリッドエンジンの初期化
    engine = HybridMoERAGEngine()
    
    # クエリ実行
    query = "設計速度80km/hの道路における最小曲線半径について、安全性を考慮した設計方法を教えてください"
    
    result = await engine.query(query, top_k=5)
    
    print(f"Query: {query}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Expert Contributions: {result.expert_contributions}")
    print(f"Selected Experts: {result.moe_result.selected_experts}")


if __name__ == "__main__":
    asyncio.run(main())