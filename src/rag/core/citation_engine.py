"""
引用機能付き生成エンジン
正確な出典情報を持つ回答生成システム
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

from ..retrieval.hybrid_search import HybridSearchResult
from ..retrieval.reranker import RerankedResult


@dataclass
class Citation:
    """引用情報"""
    id: str
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    document_title: Optional[str] = None
    document_version: Optional[str] = None
    document_type: Optional[str] = None
    confidence_score: float = 1.0
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    
    def to_reference_string(self) -> str:
        """参考文献形式の文字列を生成"""
        parts = []
        
        if self.document_title:
            parts.append(self.document_title)
            
        if self.chapter:
            parts.append(f"第{self.chapter}章")
            
        if self.section:
            parts.append(f"第{self.section}節")
            
        if self.page:
            parts.append(f"p.{self.page}")
            
        if self.document_version:
            parts.append(f"({self.document_version})")
            
        return " - ".join(parts) if parts else self.source
        
    def to_inline_citation(self) -> str:
        """インライン引用形式を生成"""
        if self.document_title and self.page:
            return f"[{self.document_title}, p.{self.page}]"
        elif self.source:
            return f"[{self.source}]"
        else:
            return f"[出典{self.id}]"


@dataclass
class GeneratedResponse:
    """生成された回答"""
    answer: str
    citations: List[Citation]
    source_chunks: List[Union[HybridSearchResult, RerankedResult]]
    confidence_score: float
    generation_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'answer': self.answer,
            'citations': [asdict(cite) for cite in self.citations],
            'source_chunks': [
                result.original_result.to_dict() if hasattr(result, 'original_result') 
                else result.to_dict() if hasattr(result, 'to_dict')
                else str(result)
                for result in self.source_chunks
            ],
            'confidence_score': self.confidence_score,
            'generation_metadata': self.generation_metadata
        }


class CitationExtractor:
    """引用情報抽出器"""
    
    def __init__(self):
        """引用抽出器の初期化"""
        
        # 引用パターン
        self.citation_patterns = [
            # 直接引用パターン
            r'「([^」]+)」',  # 日本語の引用符
            r'"([^"]+)"',    # 英語の引用符
            r'『([^』]+)』',  # 書籍タイトル風
            
            # 数値引用パターン
            r'(\d+(?:\.\d+)?\s*(?:m|km|mm|cm|km/h|%|度|°))',
            
            # 基準値引用パターン
            r'((?:最小|最大|標準|推奨)[:：]?\s*\d+(?:\.\d+)?[^。]*)',
            
            # 条文引用パターン
            r'(第[0-9０-９]+[章節項条](?:[^。]*?))',
            
            # 表・図引用パターン
            r'((?:表|図|Table|Figure)\s*[0-9\-\.]+[^。]*)',
        ]
        
    def extract_citations_from_text(self, 
                                   text: str, 
                                   source_chunks: List[Union[HybridSearchResult, RerankedResult]]) -> List[Citation]:
        """テキストから引用を抽出"""
        
        citations = []
        citation_id = 1
        
        # 各パターンで引用を検索
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                quoted_text = match.group(1) if match.groups() else match.group(0)
                
                # ソースチャンクから最も適合するものを見つける
                best_source = self._find_best_source(quoted_text, source_chunks)
                
                if best_source:
                    citation = self._create_citation(
                        citation_id, quoted_text, best_source, match
                    )
                    citations.append(citation)
                    citation_id += 1
                    
        return citations
        
    def _find_best_source(self, 
                         quoted_text: str, 
                         source_chunks: List[Union[HybridSearchResult, RerankedResult]]) -> Optional[Union[HybridSearchResult, RerankedResult]]:
        """引用テキストに最も適合するソースチャンクを探す"""
        
        best_source = None
        best_score = 0.0
        
        for chunk in source_chunks:
            # チャンクテキストを取得
            if hasattr(chunk, 'original_result'):
                chunk_text = chunk.original_result.text
                metadata = chunk.original_result.metadata
            else:
                chunk_text = chunk.text
                metadata = chunk.metadata
                
            # 類似度を計算
            similarity_score = self._calculate_text_similarity(quoted_text, chunk_text)
            
            # メタデータのマッチングも考慮
            metadata_score = self._calculate_metadata_match(quoted_text, metadata)
            
            total_score = similarity_score + metadata_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_source = chunk
                
        return best_source if best_score > 0.3 else None
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # 完全一致
        if text1_lower in text2_lower:
            return 1.0
            
        # 部分一致
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_metadata_match(self, quoted_text: str, metadata: Dict[str, Any]) -> float:
        """メタデータとの一致度を計算"""
        score = 0.0
        quoted_lower = quoted_text.lower()
        
        # セクション情報のマッチング
        if 'section_title' in metadata:
            section_title = str(metadata['section_title']).lower()
            if any(word in section_title for word in quoted_lower.split()):
                score += 0.5
                
        # 文書タイプのマッチング
        if 'document_type' in metadata:
            doc_type = str(metadata['document_type']).lower()
            if 'standard' in doc_type and '基準' in quoted_lower:
                score += 0.3
            elif 'regulation' in doc_type and '規定' in quoted_lower:
                score += 0.3
                
        return score
        
    def _create_citation(self, 
                        citation_id: int, 
                        quoted_text: str, 
                        source_chunk: Union[HybridSearchResult, RerankedResult],
                        match_obj) -> Citation:
        """引用オブジェクトを作成"""
        
        # ソースチャンクから情報を抽出
        if hasattr(source_chunk, 'original_result'):
            chunk_data = source_chunk.original_result
            confidence = source_chunk.final_score
        else:
            chunk_data = source_chunk
            confidence = getattr(source_chunk, 'hybrid_score', 1.0)
            
        metadata = getattr(chunk_data, 'metadata', {}) or {}
        
        return Citation(
            id=str(citation_id),
            text=quoted_text,
            source=chunk_data.id,
            page=metadata.get('page'),
            section=metadata.get('section_number'),
            chapter=metadata.get('chapter_number'), 
            document_title=metadata.get('filename', '').replace('.pdf', ''),
            document_version=metadata.get('version'),
            document_type=metadata.get('document_type'),
            confidence_score=float(confidence),
            char_start=match_obj.start(),
            char_end=match_obj.end()
        )


class ResponseGenerator:
    """回答生成器"""
    
    def __init__(self):
        """回答生成器の初期化"""
        self.citation_extractor = CitationExtractor()
        
    def generate_response(self, 
                         query: str,
                         search_results: List[Union[HybridSearchResult, RerankedResult]],
                         llm_response: str,
                         include_sources: bool = True) -> GeneratedResponse:
        """引用付き回答を生成"""
        
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # 引用を抽出
        citations = self.citation_extractor.extract_citations_from_text(
            llm_response, search_results
        )
        
        # 引用付きの回答を生成
        annotated_answer = self._add_inline_citations(llm_response, citations)
        
        # ソース情報を追加
        if include_sources:
            annotated_answer = self._add_source_section(annotated_answer, citations)
            
        # 信頼度スコアを計算
        confidence_score = self._calculate_response_confidence(
            llm_response, citations, search_results
        )
        
        # メタデータを生成
        generation_metadata = {
            'query': query,
            'num_sources': len(search_results),
            'num_citations': len(citations),
            'generation_timestamp': datetime.now().isoformat(),
            'has_high_confidence_citations': any(c.confidence_score > 0.8 for c in citations)
        }
        
        response = GeneratedResponse(
            answer=annotated_answer,
            citations=citations,
            source_chunks=search_results,
            confidence_score=confidence_score,
            generation_metadata=generation_metadata
        )
        
        logger.info(f"Response generated with {len(citations)} citations, confidence: {confidence_score:.3f}")
        return response
        
    def _add_inline_citations(self, text: str, citations: List[Citation]) -> str:
        """インライン引用を追加"""
        
        annotated_text = text
        
        # 引用を文字位置の逆順でソート（後ろから処理）
        sorted_citations = sorted(citations, key=lambda c: c.char_start or 0, reverse=True)
        
        for citation in sorted_citations:
            if citation.char_start is not None and citation.char_end is not None:
                # 引用位置にインライン引用を挿入
                inline_citation = citation.to_inline_citation()
                annotated_text = (
                    annotated_text[:citation.char_end] + 
                    inline_citation + 
                    annotated_text[citation.char_end:]
                )
                
        return annotated_text
        
    def _add_source_section(self, text: str, citations: List[Citation]) -> str:
        """参考文献セクションを追加"""
        
        if not citations:
            return text
            
        source_section = "\n\n【参考文献】\n"
        
        for i, citation in enumerate(citations, 1):
            reference = citation.to_reference_string()
            source_section += f"{i}. {reference}\n"
            
        return text + source_section
        
    def _calculate_response_confidence(self, 
                                     response_text: str,
                                     citations: List[Citation],
                                     search_results: List[Union[HybridSearchResult, RerankedResult]]) -> float:
        """回答の信頼度を計算"""
        
        confidence_factors = []
        
        # 引用の質
        if citations:
            avg_citation_confidence = sum(c.confidence_score for c in citations) / len(citations)
            confidence_factors.append(avg_citation_confidence * 0.4)
            
        # 引用の数
        citation_coverage = min(len(citations) / 3, 1.0)  # 3個以上の引用で満点
        confidence_factors.append(citation_coverage * 0.2)
        
        # 検索結果の質
        if search_results:
            if hasattr(search_results[0], 'final_score'):
                avg_search_score = sum(r.final_score for r in search_results) / len(search_results)
            else:
                avg_search_score = sum(r.hybrid_score for r in search_results) / len(search_results)
            confidence_factors.append(avg_search_score * 0.3)
            
        # 回答の長さ（適度な長さが良い）
        response_length = len(response_text)
        if 100 <= response_length <= 1000:
            length_score = 1.0
        elif response_length < 100:
            length_score = response_length / 100
        else:
            length_score = max(0.5, 1000 / response_length)
        confidence_factors.append(length_score * 0.1)
        
        return sum(confidence_factors) if confidence_factors else 0.5


class CitationQueryEngine:
    """引用機能付きクエリエンジン"""
    
    def __init__(self, 
                 hybrid_search_engine,
                 reranker=None,
                 llm_generator=None):
        """
        Args:
            hybrid_search_engine: ハイブリッド検索エンジン
            reranker: リランカー
            llm_generator: LLM生成器
        """
        self.search_engine = hybrid_search_engine
        self.reranker = reranker
        self.llm_generator = llm_generator
        self.response_generator = ResponseGenerator()
        
    def query(self, 
             query_text: str,
             top_k: int = 5,
             include_sources: bool = True,
             **kwargs) -> GeneratedResponse:
        """引用付きクエリを実行"""
        
        logger.info(f"Processing query: {query_text}")
        
        # 1. 検索を実行
        from ..retrieval.hybrid_search import SearchQuery
        search_query = SearchQuery(text=query_text, **kwargs)
        search_results = self.search_engine.search(search_query, top_k=top_k * 2)
        
        # 2. リランキング（オプション）
        if self.reranker:
            reranked_results = self.reranker.rerank(query_text, search_results, top_k)
            final_results = reranked_results
        else:
            final_results = search_results[:top_k]
            
        # 3. コンテキストを構築
        context_text = self._build_context(final_results)
        
        # 4. LLM生成
        if self.llm_generator:
            logger.info(f"Using LLM generator for query: {query_text[:50]}...")
            logger.info(f"Context length: {len(context_text)} characters")
            llm_response = self.llm_generator.generate(query_text, context_text)
            logger.info(f"LLM response length: {len(llm_response)} characters")
        else:
            logger.warning("No LLM generator available, using fallback response")
            llm_response = self._generate_fallback_response(query_text, final_results)
            
        # 5. 引用付き回答を生成
        response = self.response_generator.generate_response(
            query_text, final_results, llm_response, include_sources
        )
        
        return response
        
    def _build_context(self, results: List[Union[HybridSearchResult, RerankedResult]]) -> str:
        """検索結果からコンテキストを構築"""
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            if hasattr(result, 'original_result'):
                text = result.original_result.text
                metadata = result.original_result.metadata
            else:
                text = result.text
                metadata = result.metadata
                
            # メタデータ情報を含めたコンテキスト
            source_info = f"[出典{i}]"
            if 'filename' in metadata:
                source_info += f" {metadata['filename']}"
            if 'page' in metadata:
                source_info += f" p.{metadata['page']}"
                
            context_parts.append(f"{source_info}\n{text}\n")
            
        return "\n".join(context_parts)
        
    def _generate_fallback_response(self, 
                                   query: str, 
                                   results: List[Union[HybridSearchResult, RerankedResult]]) -> str:
        """フォールバック回答生成（LLMが利用できない場合）"""
        
        if not results:
            return "申し訳ございませんが、ご質問に関する情報が見つかりませんでした。"
            
        # 最も関連性の高い結果から回答を構築
        best_result = results[0]
        
        if hasattr(best_result, 'original_result'):
            text = best_result.original_result.text
        else:
            text = best_result.text
            
        # 簡易的な回答生成
        response = f"ご質問に関して、以下の情報が見つかりました：\n\n{text[:500]}"
        
        if len(text) > 500:
            response += "..."
            
        return response


# 便利な関数
def create_citation_query_engine(hybrid_search_engine, **kwargs) -> CitationQueryEngine:
    """引用機能付きクエリエンジンを作成"""
    return CitationQueryEngine(hybrid_search_engine, **kwargs)


def extract_citations_from_response(response_text: str,
                                   source_chunks: List[Union[HybridSearchResult, RerankedResult]]) -> List[Citation]:
    """回答から引用を抽出（便利関数）"""
    extractor = CitationExtractor()
    return extractor.extract_citations_from_text(response_text, source_chunks)