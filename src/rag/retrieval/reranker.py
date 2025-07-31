"""
リランキングモジュール
検索結果の精度向上のための再順位付けシステム
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from loguru import logger
import re

from .hybrid_search import HybridSearchResult


@dataclass
class RerankedResult:
    """リランキング済み結果"""
    original_result: HybridSearchResult
    rerank_score: float
    final_score: float
    rank: int
    explanation: Optional[str] = None
    
    @property
    def score(self) -> float:
        """スコアのエイリアス（互換性のため）"""
        return self.final_score


class CrossEncoderReranker:
    """Cross-Encoderベースのリランカー"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 device: Optional[str] = None):
        """
        Args:
            model_name: リランキングモデル名
            device: 使用デバイス
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info(f"CrossEncoder loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder {model_name}: {e}")
            self.model = None
            
    def rerank(self, 
              query: str, 
              results: List[HybridSearchResult],
              top_k: Optional[int] = None) -> List[RerankedResult]:
        """検索結果をリランキング"""
        
        if not self.model:
            logger.warning("CrossEncoder not available, skipping reranking")
            return self._fallback_rerank(query, results, top_k)
            
        if not results:
            return []
            
        # クエリと文書のペアを作成
        query_doc_pairs = [(query, result.text) for result in results]
        
        # リランキングスコアを計算
        try:
            rerank_scores = self.model.predict(query_doc_pairs)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._fallback_rerank(query, results, top_k)
            
        # 結果を統合
        reranked_results = []
        for i, (result, rerank_score) in enumerate(zip(results, rerank_scores)):
            # 最終スコア = 元スコア * リランキングスコア
            final_score = result.hybrid_score * (1.0 + float(rerank_score))
            
            reranked_result = RerankedResult(
                original_result=result,
                rerank_score=float(rerank_score),
                final_score=final_score,
                rank=i + 1  # 後で更新
            )
            reranked_results.append(reranked_result)
            
        # 最終スコアでソート
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # ランクを更新
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
            
        # 上位k件を返す
        if top_k:
            reranked_results = reranked_results[:top_k]
            
        logger.info(f"Reranked {len(results)} results")
        return reranked_results
        
    def _fallback_rerank(self, 
                        query: str, 
                        results: List[HybridSearchResult],
                        top_k: Optional[int] = None) -> List[RerankedResult]:
        """フォールバックリランキング"""
        
        reranked_results = []
        for i, result in enumerate(results):
            reranked_result = RerankedResult(
                original_result=result,
                rerank_score=0.0,
                final_score=result.hybrid_score,
                rank=i + 1
            )
            reranked_results.append(reranked_result)
            
        if top_k:
            reranked_results = reranked_results[:top_k]
            
        return reranked_results


class TechnicalTermReranker:
    """技術用語特化型リランカー"""
    
    def __init__(self):
        """道路設計専門用語リランカーの初期化"""
        
        # 道路設計の重要用語と重み
        self.technical_weights = {
            # 高優先度用語
            '設計速度': 3.0,
            '曲線半径': 3.0,
            '縦断勾配': 3.0,
            '横断勾配': 2.5,
            '車道幅員': 2.5,
            '歩道幅員': 2.0,
            '路肩幅員': 2.0,
            '中央分離帯': 2.0,
            '制動距離': 2.5,
            '視距': 2.5,
            
            # 中優先度用語
            '道路構造令': 2.0,
            '設計基準': 2.0,
            '安全性': 1.8,
            '交通容量': 1.8,
            '荷重': 1.5,
            '材料': 1.5,
            '施工': 1.3,
            '維持管理': 1.3,
            
            # 数値・単位関連
            'km/h': 2.0,
            'メートル': 1.5,
            'パーセント': 1.5,
            '最小': 1.8,
            '最大': 1.8,
            '標準': 1.5,
            '推奨': 1.5
        }
        
        # 数値パターン
        self.numerical_patterns = [
            r'\d+(?:\.\d+)?\s*km/h',  # 速度
            r'\d+(?:\.\d+)?\s*m(?:\s|$)',  # 距離
            r'\d+(?:\.\d+)?\s*%',  # 勾配
            r'\d+(?:\.\d+)?\s*度',  # 角度
            r'半径\s*[:：]?\s*\d+(?:\.\d+)?\s*m',  # 曲線半径
            r'勾配\s*[:：]?\s*\d+(?:\.\d+)?\s*%',  # 勾配値
            r'速度\s*[:：]?\s*\d+(?:\.\d+)?\s*km/h'  # 設計速度
        ]
        
    def calculate_technical_relevance(self, query: str, text: str) -> float:
        """技術的関連度を計算"""
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        relevance_score = 0.0
        
        # 技術用語のマッチング
        for term, weight in self.technical_weights.items():
            term_lower = term.lower()
            
            # クエリに含まれる用語の重み付け
            if term_lower in query_lower:
                if term_lower in text_lower:
                    # 完全一致の場合は高いスコア
                    relevance_score += weight * 2.0
                    
                    # 出現頻度も考慮
                    count = text_lower.count(term_lower)
                    relevance_score += min(count - 1, 2) * weight * 0.5
                    
        # 数値パターンのマッチング
        query_numbers = self._extract_numbers(query)
        text_numbers = self._extract_numbers(text)
        
        for q_num, q_unit in query_numbers:
            for t_num, t_unit in text_numbers:
                if q_unit == t_unit:  # 同じ単位
                    if abs(q_num - t_num) / max(q_num, t_num) < 0.1:  # 10%以内の差
                        relevance_score += 2.0
                    elif abs(q_num - t_num) / max(q_num, t_num) < 0.3:  # 30%以内の差
                        relevance_score += 1.0
                        
        # 正規化
        return min(relevance_score / 10.0, 1.0)
        
    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """テキストから数値と単位を抽出"""
        numbers = []
        
        for pattern in self.numerical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                match_text = match.group(0)
                
                # 数値と単位を分離
                num_match = re.search(r'\d+(?:\.\d+)?', match_text)
                if num_match:
                    number = float(num_match.group(0))
                    
                    # 単位を推定
                    if 'km/h' in match_text:
                        unit = 'speed'
                    elif 'm' in match_text:
                        unit = 'length'
                    elif '%' in match_text:
                        unit = 'percentage'
                    elif '度' in match_text:
                        unit = 'angle'
                    else:
                        unit = 'unknown'
                        
                    numbers.append((number, unit))
                    
        return numbers


class ContextualReranker:
    """文脈理解型リランカー"""
    
    def __init__(self):
        """文脈理解リランカーの初期化"""
        
        # 文脈パターン
        self.context_patterns = {
            # 設計関連
            'design': [
                r'設計\s*(?:に|の|で|する|した|による)',
                r'計画\s*(?:に|の|で|する|した)',
                r'基準\s*(?:に|の|で|による|として)',
            ],
            
            # 計算関連
            'calculation': [
                r'計算\s*(?:に|の|で|する|した|による)',
                r'算出\s*(?:に|の|で|する|した)',
                r'求める\s*(?:に|の|で|場合)',
            ],
            
            # 規定関連
            'regulation': [
                r'規定\s*(?:に|の|で|による|として)',
                r'定める\s*(?:に|の|で|場合)',
                r'適用\s*(?:に|の|で|する|した)',
            ],
            
            # 条件関連
            'condition': [
                r'場合\s*(?:に|の|で|には)',
                r'条件\s*(?:に|の|で|として)',
                r'状況\s*(?:に|の|で)',
            ]
        }
        
    def calculate_contextual_relevance(self, query: str, text: str) -> float:
        """文脈的関連度を計算"""
        
        relevance_score = 0.0
        
        # クエリの文脈タイプを特定
        query_contexts = self._identify_contexts(query)
        text_contexts = self._identify_contexts(text)
        
        # 共通する文脈タイプに対してスコア加算
        for context in query_contexts:
            if context in text_contexts:
                relevance_score += 0.5
                
        # 文章構造の類似性
        query_structure = self._analyze_sentence_structure(query)
        text_structure = self._analyze_sentence_structure(text)
        
        structure_similarity = self._calculate_structure_similarity(
            query_structure, text_structure
        )
        relevance_score += structure_similarity * 0.3
        
        return min(relevance_score, 1.0)
        
    def _identify_contexts(self, text: str) -> List[str]:
        """文脈タイプを特定"""
        contexts = []
        
        for context_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    contexts.append(context_type)
                    break
                    
        return contexts
        
    def _analyze_sentence_structure(self, text: str) -> Dict[str, int]:
        """文章構造を解析"""
        structure = {
            'questions': len(re.findall(r'[？?]', text)),
            'conditions': len(re.findall(r'(?:場合|条件|とき)', text)),
            'numbers': len(re.findall(r'\d+(?:\.\d+)?', text)),
            'technical_terms': len(re.findall(r'(?:設計|基準|規定|計算)', text))
        }
        return structure
        
    def _calculate_structure_similarity(self, 
                                      struct1: Dict[str, int], 
                                      struct2: Dict[str, int]) -> float:
        """構造の類似度を計算"""
        similarity = 0.0
        total_features = len(struct1)
        
        for key in struct1:
            if key in struct2:
                # 正規化された差分
                val1 = struct1[key]
                val2 = struct2[key]
                max_val = max(val1, val2, 1)
                
                similarity += 1.0 - abs(val1 - val2) / max_val
                
        return similarity / total_features


class HybridReranker:
    """統合リランカー"""
    
    def __init__(self, 
                 use_cross_encoder: bool = True,
                 cross_encoder_weight: float = 0.4,
                 technical_weight: float = 0.4,
                 contextual_weight: float = 0.2):
        """
        Args:
            use_cross_encoder: CrossEncoderを使用するか
            cross_encoder_weight: CrossEncoderの重み
            technical_weight: 技術用語リランカーの重み
            contextual_weight: 文脈リランカーの重み
        """
        self.cross_encoder_weight = cross_encoder_weight
        self.technical_weight = technical_weight
        self.contextual_weight = contextual_weight
        
        # 各リランカーの初期化
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        else:
            self.cross_encoder = None
            
        self.technical_reranker = TechnicalTermReranker()
        self.contextual_reranker = ContextualReranker()
        
    def rerank(self, 
              query: str, 
              results: List[HybridSearchResult],
              top_k: Optional[int] = None) -> List[RerankedResult]:
        """統合リランキングを実行"""
        
        if not results:
            return []
            
        logger.info(f"Starting hybrid reranking for {len(results)} results")
        
        # 各リランカーのスコアを計算
        reranked_results = []
        
        for i, result in enumerate(results):
            # CrossEncoderスコア
            if self.cross_encoder:
                ce_results = self.cross_encoder.rerank(query, [result])
                ce_score = ce_results[0].rerank_score if ce_results else 0.0
            else:
                ce_score = 0.0
                
            # 技術用語スコア
            tech_score = self.technical_reranker.calculate_technical_relevance(
                query, result.text
            )
            
            # 文脈スコア
            context_score = self.contextual_reranker.calculate_contextual_relevance(
                query, result.text
            )
            
            # 統合スコア計算
            rerank_score = (
                self.cross_encoder_weight * ce_score +
                self.technical_weight * tech_score +
                self.contextual_weight * context_score
            )
            
            # 最終スコア = 元スコア + リランキングブースト
            final_score = result.hybrid_score * (1.0 + rerank_score)
            
            # 説明文生成
            explanation = f"CE:{ce_score:.3f}, Tech:{tech_score:.3f}, Context:{context_score:.3f}"
            
            reranked_result = RerankedResult(
                original_result=result,
                rerank_score=rerank_score,
                final_score=final_score,
                rank=i + 1,  # 後で更新
                explanation=explanation
            )
            reranked_results.append(reranked_result)
            
        # 最終スコアでソート
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # ランクを更新
        for i, result in enumerate(reranked_results):
            result.rank = i + 1
            
        # 上位k件を返す
        if top_k:
            reranked_results = reranked_results[:top_k]
            
        logger.info(f"Hybrid reranking completed, top score: {reranked_results[0].final_score:.3f}")
        return reranked_results


# 便利な関数
def rerank_search_results(query: str,
                         results: List[HybridSearchResult],
                         reranker_type: str = "hybrid",
                         **kwargs) -> List[RerankedResult]:
    """検索結果をリランキング（便利関数）"""
    
    if reranker_type == "cross_encoder":
        reranker = CrossEncoderReranker(**kwargs)
    elif reranker_type == "technical":
        reranker = TechnicalTermReranker()
        # TechnicalTermRerankerはrerank関数がないので、HybridRerankerを使用
        reranker = HybridReranker(use_cross_encoder=False, technical_weight=1.0)
    else:  # hybrid
        reranker = HybridReranker(**kwargs)
        
    return reranker.rerank(query, results)