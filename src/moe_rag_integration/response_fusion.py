"""
レスポンス融合
MoEとRAGの結果を統合して最適な回答を生成
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """融合結果"""
    fused_answer: str
    fusion_method: str
    source_contributions: Dict[str, float]
    quality_score: float
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fused_answer': self.fused_answer,
            'fusion_method': self.fusion_method,
            'source_contributions': self.source_contributions,
            'quality_score': self.quality_score,
            'citations': self.citations,
            'metadata': self.metadata
        }


class ResponseFusion:
    """レスポンス融合クラス"""
    
    def __init__(self, fusion_strategy: str = "adaptive"):
        """
        Args:
            fusion_strategy: 融合戦略 (adaptive, weighted, hierarchical)
        """
        self.fusion_strategy = fusion_strategy
        
        # 融合テンプレート
        self.templates = {
            "technical": """
【技術的回答】
{main_content}

【詳細説明】
{detailed_explanation}

【参考情報】
{references}

【専門家による補足】
{expert_notes}
""",
            "comprehensive": """
{introduction}

【主要な内容】
{main_points}

【関連情報】
{related_info}

【まとめ】
{conclusion}
""",
            "concise": """
{direct_answer}

【根拠】
{evidence}

【出典】
{sources}
"""
        }
        
        logger.info(f"Response Fusion initialized with strategy: {fusion_strategy}")
    
    def fuse(
        self,
        moe_response: str,
        rag_response: str,
        moe_confidence: float,
        rag_confidence: float,
        query_type: str = "general",
        expert_info: Optional[Dict[str, Any]] = None,
        documents: Optional[List[Any]] = None
    ) -> FusionResult:
        """
        MoEとRAGのレスポンスを融合
        
        Args:
            moe_response: MoEの回答
            rag_response: RAGの回答
            moe_confidence: MoEの信頼度
            rag_confidence: RAGの信頼度
            query_type: クエリタイプ
            expert_info: エキスパート情報
            documents: RAG検索文書
            
        Returns:
            融合結果
        """
        # 1. 融合方法の決定
        fusion_method = self._determine_fusion_method(
            moe_confidence, rag_confidence, query_type
        )
        
        # 2. コンテンツ分析
        moe_analysis = self._analyze_content(moe_response)
        rag_analysis = self._analyze_content(rag_response)
        
        # 3. 重複除去と統合
        merged_content = self._merge_content(
            moe_response, rag_response,
            moe_analysis, rag_analysis,
            fusion_method
        )
        
        # 4. 引用情報の生成
        citations = self._generate_citations(documents, expert_info)
        
        # 5. 最終回答の構成
        fused_answer = self._compose_final_answer(
            merged_content, citations, query_type
        )
        
        # 6. ソース貢献度の計算
        source_contributions = self._calculate_contributions(
            moe_confidence, rag_confidence, fusion_method
        )
        
        # 7. 品質スコアの計算
        quality_score = self._calculate_quality_score(
            fused_answer, moe_confidence, rag_confidence
        )
        
        return FusionResult(
            fused_answer=fused_answer,
            fusion_method=fusion_method,
            source_contributions=source_contributions,
            quality_score=quality_score,
            citations=citations,
            metadata={
                'moe_confidence': moe_confidence,
                'rag_confidence': rag_confidence,
                'query_type': query_type,
                'content_overlap': self._calculate_overlap(moe_response, rag_response)
            }
        )
    
    def _determine_fusion_method(
        self,
        moe_confidence: float,
        rag_confidence: float,
        query_type: str
    ) -> str:
        """融合方法を決定"""
        if self.fusion_strategy == "adaptive":
            # 信頼度差に基づく選択
            confidence_diff = abs(moe_confidence - rag_confidence)
            
            if confidence_diff > 0.3:
                # 一方が明確に優れている
                return "dominant_source"
            elif moe_confidence > 0.7 and rag_confidence > 0.7:
                # 両方高信頼度
                return "complementary"
            elif query_type in ["technical", "specialized"]:
                # 技術的クエリ
                return "expert_weighted"
            else:
                # バランス融合
                return "balanced"
        
        elif self.fusion_strategy == "weighted":
            return "weighted_average"
        
        elif self.fusion_strategy == "hierarchical":
            return "hierarchical"
        
        else:
            return "simple_merge"
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """コンテンツを分析"""
        # 文の分割
        sentences = re.split(r'[。！？\n]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # キーポイントの抽出
        key_points = []
        for sentence in sentences:
            # 数値を含む文
            if re.search(r'\d+', sentence):
                key_points.append(sentence)
            # 定義文
            elif any(word in sentence for word in ['とは', 'である', '定義']):
                key_points.append(sentence)
            # 重要キーワード
            elif any(word in sentence for word in ['必要', '重要', '注意', '推奨']):
                key_points.append(sentence)
        
        # 技術用語の抽出
        technical_terms = self._extract_technical_terms(content)
        
        # 構造分析
        has_list = '・' in content or '1.' in content or '①' in content
        has_sections = '【' in content or '■' in content
        
        return {
            'sentences': sentences,
            'key_points': key_points,
            'technical_terms': technical_terms,
            'length': len(content),
            'has_list': has_list,
            'has_sections': has_sections,
            'sentiment': self._analyze_sentiment(content)
        }
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """技術用語を抽出"""
        # 簡易的な技術用語パターン
        patterns = [
            r'[ァ-ヶー]+(?:設計|工法|構造|材料|基準)',  # カタカナ+技術語
            r'\b[A-Z]{2,}\b',  # 大文字略語
            r'\d+\s*(?:mm|cm|m|km|kg|kN|MPa|N/mm2)',  # 数値+単位
        ]
        
        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            terms.extend(matches)
        
        return list(set(terms))
    
    def _analyze_sentiment(self, content: str) -> str:
        """文の論調を分析"""
        if any(word in content for word in ['推奨', '望ましい', '適切']):
            return "positive"
        elif any(word in content for word in ['注意', '避ける', '問題']):
            return "cautionary"
        elif any(word in content for word in ['必須', '必要', '義務']):
            return "mandatory"
        else:
            return "neutral"
    
    def _merge_content(
        self,
        moe_response: str,
        rag_response: str,
        moe_analysis: Dict[str, Any],
        rag_analysis: Dict[str, Any],
        fusion_method: str
    ) -> Dict[str, Any]:
        """コンテンツをマージ"""
        merged = {
            'main_content': '',
            'supporting_details': [],
            'unique_moe': [],
            'unique_rag': []
        }
        
        if fusion_method == "dominant_source":
            # 信頼度の高い方を主とする
            if moe_analysis['length'] > rag_analysis['length']:
                merged['main_content'] = moe_response
                merged['supporting_details'] = rag_analysis['key_points']
            else:
                merged['main_content'] = rag_response
                merged['supporting_details'] = moe_analysis['key_points']
        
        elif fusion_method == "complementary":
            # 相補的に統合
            # 重複を除去しながら統合
            all_sentences = set(moe_analysis['sentences'] + rag_analysis['sentences'])
            
            # 重要度でソート
            scored_sentences = []
            for sentence in all_sentences:
                score = 0
                if sentence in moe_analysis['key_points']:
                    score += 2
                if sentence in rag_analysis['key_points']:
                    score += 2
                if any(term in sentence for term in moe_analysis['technical_terms']):
                    score += 1
                scored_sentences.append((sentence, score))
            
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            merged['main_content'] = '。'.join([s[0] for s in scored_sentences[:5]]) + '。'
            merged['supporting_details'] = [s[0] for s in scored_sentences[5:10]]
        
        elif fusion_method == "expert_weighted":
            # エキスパート重視
            merged['main_content'] = moe_response
            # RAGから補足情報を抽出
            for point in rag_analysis['key_points']:
                if point not in moe_response:
                    merged['supporting_details'].append(point)
        
        else:  # balanced or simple_merge
            # バランス融合
            merged['main_content'] = f"{moe_response}\n\n{rag_response}"
        
        # ユニークな内容の特定
        moe_unique = set(moe_analysis['sentences']) - set(rag_analysis['sentences'])
        rag_unique = set(rag_analysis['sentences']) - set(moe_analysis['sentences'])
        
        merged['unique_moe'] = list(moe_unique)[:3]
        merged['unique_rag'] = list(rag_unique)[:3]
        
        return merged
    
    def _generate_citations(
        self,
        documents: Optional[List[Any]],
        expert_info: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """引用情報を生成"""
        citations = []
        
        # 文書からの引用
        if documents:
            for i, doc in enumerate(documents[:5], 1):
                citation = {
                    'type': 'document',
                    'source': getattr(doc, 'metadata', {}).get('filename', f'文書{i}'),
                    'page': getattr(doc, 'metadata', {}).get('page'),
                    'relevance': getattr(doc, 'score', 0.0)
                }
                citations.append(citation)
        
        # エキスパートからの引用
        if expert_info:
            for expert_name, contribution in expert_info.get('contributions', {}).items():
                if contribution > 0.1:
                    citation = {
                        'type': 'expert',
                        'source': expert_name,
                        'contribution': contribution
                    }
                    citations.append(citation)
        
        return citations
    
    def _compose_final_answer(
        self,
        merged_content: Dict[str, Any],
        citations: List[Dict[str, Any]],
        query_type: str
    ) -> str:
        """最終回答を構成"""
        # クエリタイプに応じたテンプレート選択
        if query_type in ["technical", "specialized"]:
            template_name = "technical"
        elif query_type == "comprehensive":
            template_name = "comprehensive"
        else:
            template_name = "concise"
        
        template = self.templates[template_name]
        
        # テンプレート変数の準備
        if template_name == "technical":
            main_content = merged_content['main_content']
            detailed_explanation = '\n'.join(merged_content.get('supporting_details', []))
            references = self._format_citations(citations, 'document')
            expert_notes = self._format_citations(citations, 'expert')
            
            return template.format(
                main_content=main_content,
                detailed_explanation=detailed_explanation or "詳細情報なし",
                references=references or "参考文献なし",
                expert_notes=expert_notes or "専門家の補足なし"
            )
        
        elif template_name == "comprehensive":
            # 導入部の生成
            introduction = merged_content['main_content'].split('。')[0] + '。' if merged_content['main_content'] else ""
            
            # 主要ポイント
            main_points = '\n'.join([
                f"• {point}" for point in merged_content.get('supporting_details', [])[:5]
            ])
            
            # 関連情報
            related_info = '\n'.join([
                f"- {info}" for info in merged_content.get('unique_rag', [])[:3]
            ])
            
            # まとめ
            conclusion = "以上の情報を総合すると、適切な判断と実施が可能です。"
            
            return template.format(
                introduction=introduction,
                main_points=main_points or "主要ポイントなし",
                related_info=related_info or "関連情報なし",
                conclusion=conclusion
            )
        
        else:  # concise
            direct_answer = merged_content['main_content'][:500] + "..." if len(merged_content['main_content']) > 500 else merged_content['main_content']
            evidence = '\n'.join([f"• {detail}" for detail in merged_content.get('supporting_details', [])[:3]])
            sources = self._format_all_citations(citations)
            
            return template.format(
                direct_answer=direct_answer,
                evidence=evidence or "根拠情報なし",
                sources=sources or "出典なし"
            )
    
    def _format_citations(self, citations: List[Dict[str, Any]], citation_type: str) -> str:
        """引用をフォーマット"""
        filtered = [c for c in citations if c.get('type') == citation_type]
        
        if not filtered:
            return ""
        
        formatted = []
        for citation in filtered[:5]:
            if citation_type == 'document':
                text = f"- {citation['source']}"
                if citation.get('page'):
                    text += f" (p.{citation['page']})"
                formatted.append(text)
            elif citation_type == 'expert':
                text = f"- {citation['source']}: 貢献度 {citation['contribution']:.1%}"
                formatted.append(text)
        
        return '\n'.join(formatted)
    
    def _format_all_citations(self, citations: List[Dict[str, Any]]) -> str:
        """すべての引用をフォーマット"""
        if not citations:
            return ""
        
        formatted = []
        for i, citation in enumerate(citations[:7], 1):
            if citation['type'] == 'document':
                text = f"[{i}] {citation['source']}"
                if citation.get('page'):
                    text += f" p.{citation['page']}"
            else:
                text = f"[{i}] {citation['source']} (エキスパート)"
            formatted.append(text)
        
        return '\n'.join(formatted)
    
    def _calculate_contributions(
        self,
        moe_confidence: float,
        rag_confidence: float,
        fusion_method: str
    ) -> Dict[str, float]:
        """ソース貢献度を計算"""
        if fusion_method == "dominant_source":
            if moe_confidence > rag_confidence:
                return {'moe': 0.8, 'rag': 0.2}
            else:
                return {'moe': 0.2, 'rag': 0.8}
        
        elif fusion_method == "expert_weighted":
            return {'moe': 0.7, 'rag': 0.3}
        
        elif fusion_method == "complementary":
            return {'moe': 0.5, 'rag': 0.5}
        
        else:  # balanced or weighted
            total = moe_confidence + rag_confidence
            if total > 0:
                return {
                    'moe': moe_confidence / total,
                    'rag': rag_confidence / total
                }
            else:
                return {'moe': 0.5, 'rag': 0.5}
    
    def _calculate_quality_score(
        self,
        fused_answer: str,
        moe_confidence: float,
        rag_confidence: float
    ) -> float:
        """品質スコアを計算"""
        # 基本スコア（信頼度の平均）
        base_score = (moe_confidence + rag_confidence) / 2
        
        # 長さボーナス（適切な長さ）
        answer_length = len(fused_answer)
        if 200 <= answer_length <= 1000:
            length_bonus = 0.1
        elif 100 <= answer_length < 200 or 1000 < answer_length <= 2000:
            length_bonus = 0.05
        else:
            length_bonus = 0.0
        
        # 構造ボーナス（セクション分けされている）
        structure_bonus = 0.1 if '【' in fused_answer else 0.0
        
        # 引用ボーナス
        citation_bonus = 0.05 if '出典' in fused_answer or '参考' in fused_answer else 0.0
        
        quality_score = min(
            base_score + length_bonus + structure_bonus + citation_bonus,
            1.0
        )
        
        return float(quality_score)
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """テキスト間の重複度を計算"""
        # 単語レベルでの重複を計算
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


# 使用例
if __name__ == "__main__":
    fusion = ResponseFusion(fusion_strategy="adaptive")
    
    # サンプルレスポンス
    moe_response = "道路の設計速度80km/hにおける最小曲線半径は、道路構造令により280mと定められています。これは安全性と快適性を考慮した基準値です。"
    
    rag_response = "設計速度80km/hの道路では、最小曲線半径280mが標準となります。ただし、地形条件により困難な場合は、特例値として230mまで縮小可能です。"
    
    result = fusion.fuse(
        moe_response=moe_response,
        rag_response=rag_response,
        moe_confidence=0.85,
        rag_confidence=0.75,
        query_type="technical"
    )
    
    print("Fused Answer:")
    print(result.fused_answer)
    print(f"\nFusion Method: {result.fusion_method}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Source Contributions: {result.source_contributions}")