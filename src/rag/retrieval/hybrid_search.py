"""
ハイブリッド検索エンジン
ベクトル検索とキーワード検索を統合した高精度検索システム
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..indexing.vector_store import QdrantVectorStore, SearchResult
from ..indexing.embedding_model import EmbeddingModel


@dataclass
class SearchQuery:
    """検索クエリを表すデータクラス"""
    text: str
    filters: Optional[Dict[str, Any]] = None
    boost_terms: Optional[List[str]] = None
    search_type: str = "hybrid"  # hybrid, vector, keyword
    
    
@dataclass
class HybridSearchResult:
    """ハイブリッド検索結果"""
    id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    keyword_score: float
    hybrid_score: float
    rank: int
    
    @classmethod
    def from_search_result(cls, 
                          result: SearchResult, 
                          vector_score: float,
                          keyword_score: float,
                          hybrid_score: float,
                          rank: int) -> 'HybridSearchResult':
        """SearchResultから変換"""
        return cls(
            id=result.id,
            text=result.text,
            metadata=result.metadata,
            vector_score=vector_score,
            keyword_score=keyword_score,
            hybrid_score=hybrid_score,
            rank=rank
        )


class TechnicalTermExtractor:
    """技術用語抽出器"""
    
    def __init__(self, model_name: str = "ja_core_news_lg"):
        """
        Args:
            model_name: Spacyモデル名
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Spacy model {model_name} not found, using fallback")
            self.nlp = None
            
        # 道路設計関連の専門用語パターン
        self.technical_patterns = [
            # 数値と単位
            r'\d+(?:\.\d+)?\s*(?:m|km|mm|cm|km/h|%|度|°)',
            # 設計基準値
            r'(?:最小|最大|標準|推奨)[:：]?\s*\d+(?:\.\d+)?',
            # 道路部位
            r'(?:車道|歩道|中央分離帯|路肩|側道)(?:幅員?)?',
            # 技術用語
            r'(?:設計速度|曲線半径|縦断勾配|横断勾配|視距|制動距離)',
            # 基準・規格
            r'(?:道路構造令|設計基準|技術基準|JIS|ISO)\s*[A-Z0-9\-]*',
            # 表・図の参照
            r'(?:表|図|Table|Figure)\s*[0-9\-\.]+',
        ]
        
        # 重要キーワード辞書
        self.important_keywords = {
            '設計': 2.0,
            '基準': 2.0,
            '規定': 1.8,
            '計算': 1.5,
            '安全': 1.8,
            '構造': 1.5,
            '幅員': 1.5,
            '勾配': 1.5,
            '半径': 1.5,
            '速度': 1.4,
            '距離': 1.3,
            '高さ': 1.3,
            '荷重': 1.4,
            '材料': 1.3,
            '施工': 1.2,
            '検査': 1.2,
            '維持': 1.1,
            '管理': 1.1
        }
        
    def extract_technical_terms(self, text: str) -> List[Tuple[str, float]]:
        """技術用語を抽出し重みを付与"""
        terms = []
        
        # パターンマッチング
        for pattern in self.technical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(0)
                weight = 2.0  # 技術パターンは高い重み
                terms.append((term, weight))
                
        # 重要キーワードのマッチング
        for keyword, weight in self.important_keywords.items():
            if keyword in text:
                terms.append((keyword, weight))
                
        # Spacyを使った名詞・専門用語抽出
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                # 名詞で長さが2文字以上
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    len(token.text) >= 2 and 
                    not token.is_stop):
                    terms.append((token.text, 1.0))
                    
        # 重複除去と正規化
        term_weights = {}
        for term, weight in terms:
            term = term.strip()
            if term and len(term) >= 2:
                if term in term_weights:
                    term_weights[term] = max(term_weights[term], weight)
                else:
                    term_weights[term] = weight
                    
        return list(term_weights.items())


class KeywordSearchEngine:
    """キーワード検索エンジン"""
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2):
        """
        Args:
            max_features: TF-IDFの最大特徴数
            ngram_range: N-gramの範囲
            min_df: 最小文書頻度
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=None,  # 日本語なのでNone
            token_pattern=r'(?u)\b\w+\b'
        )
        
        self.term_extractor = TechnicalTermExtractor()
        self.corpus_vectors = None
        self.corpus_texts = None
        self.corpus_ids = None
        self.is_fitted = False
        
    def fit(self, texts: List[str], ids: List[str]):
        """文書コーパスでTF-IDFモデルを学習"""
        self.corpus_texts = texts
        self.corpus_ids = ids
        
        logger.info(f"Fitting keyword search on {len(texts)} documents...")
        
        if len(texts) == 0:
            # 空のコーパスの場合
            logger.warning("Empty corpus provided, keyword search will be disabled")
            self.corpus_vectors = None
            self.is_fitted = True
        else:
            # TF-IDF行列を計算
            self.corpus_vectors = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            logger.info(f"Keyword search fitted with {self.corpus_vectors.shape[1]} features")
        
    def search(self, 
              query: str, 
              top_k: int = 10,
              boost_technical_terms: bool = True) -> List[Tuple[str, float]]:
        """キーワード検索を実行"""
        
        if not self.is_fitted:
            raise RuntimeError("KeywordSearchEngine must be fitted before search")
            
        if self.corpus_vectors is None:
            # 空のコーパスの場合は空の結果を返す
            logger.debug("Empty corpus, returning empty keyword search results")
            return []
            
        # クエリのベクトル化
        query_vector = self.vectorizer.transform([query])
        
        # コサイン類似度を計算
        similarities = cosine_similarity(query_vector, self.corpus_vectors).flatten()
        
        # 技術用語ブースト
        if boost_technical_terms:
            similarities = self._apply_technical_boost(query, similarities)
            
        # スコアでソート
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(ranked_indices):
            if similarities[idx] > 0:  # スコアが0より大きいもののみ
                results.append((self.corpus_ids[idx], float(similarities[idx])))
                
        return results
        
    def _apply_technical_boost(self, query: str, similarities: np.ndarray) -> np.ndarray:
        """技術用語に基づくスコアブースト"""
        
        # 空のコーパスの場合はそのまま返す
        if not self.corpus_texts or len(self.corpus_texts) == 0:
            return similarities
        
        # クエリから技術用語を抽出
        technical_terms = self.term_extractor.extract_technical_terms(query)
        
        if not technical_terms:
            return similarities
            
        # 各文書で技術用語のマッチング度を計算
        boost_scores = np.zeros(len(similarities))
        
        for i, text in enumerate(self.corpus_texts):
            doc_boost = 0.0
            text_lower = text.lower()
            
            for term, weight in technical_terms:
                term_lower = term.lower()
                # 用語の出現回数に重みを掛ける
                count = text_lower.count(term_lower)
                if count > 0:
                    doc_boost += weight * min(count, 3)  # 最大3回まで
                    
            boost_scores[i] = doc_boost
            
        # ブーストスコアを正規化
        if boost_scores.max() > 0:
            boost_scores = boost_scores / boost_scores.max() * 0.3  # 最大30%のブースト
            
        return similarities * (1.0 + boost_scores)


class HybridSearchEngine:
    """ハイブリッド検索エンジン（ベクトル + キーワード）"""
    
    def __init__(self,
                 vector_store: QdrantVectorStore,
                 embedding_model: EmbeddingModel,
                 vector_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        Args:
            vector_store: ベクトルストア
            embedding_model: 埋め込みモデル
            vector_weight: ベクトル検索の重み
            keyword_weight: キーワード検索の重み
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        self.keyword_engine = KeywordSearchEngine()
        self.term_extractor = TechnicalTermExtractor()
        self.is_ready = False
        
    def initialize(self, corpus_texts: List[str], corpus_ids: List[str]):
        """検索エンジンを初期化"""
        logger.info("Initializing hybrid search engine...")
        
        # キーワード検索エンジンを学習
        self.keyword_engine.fit(corpus_texts, corpus_ids)
        self.is_ready = True
        
        logger.info("Hybrid search engine ready")
        
    def search(self, 
              query: SearchQuery,
              top_k: int = 10) -> List[HybridSearchResult]:
        """ハイブリッド検索を実行"""
        
        if not self.is_ready:
            raise RuntimeError("HybridSearchEngine must be initialized before search")
            
        query_text = query.text
        filters = query.filters
        
        # クエリから技術用語を抽出
        technical_terms = self.term_extractor.extract_technical_terms(query_text)
        keywords = [term for term, _ in technical_terms[:10]]  # 上位10個
        
        logger.info(f"Searching for: '{query_text}' with {len(keywords)} keywords")
        
        if query.search_type == "vector":
            return self._vector_only_search(query_text, filters, top_k)
        elif query.search_type == "keyword":
            return self._keyword_only_search(query_text, top_k)
        else:  # hybrid
            return self._hybrid_search(query_text, keywords, filters, top_k)
            
    def _vector_only_search(self, 
                           query_text: str, 
                           filters: Optional[Dict[str, Any]], 
                           top_k: int) -> List[HybridSearchResult]:
        """ベクトル検索のみ"""
        
        # クエリの埋め込みを生成
        query_embedding = self.embedding_model.encode([query_text], is_query=True)[0]
        
        # ベクトル検索を実行
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # 結果を変換
        hybrid_results = []
        for i, result in enumerate(vector_results):
            hybrid_result = HybridSearchResult.from_search_result(
                result=result,
                vector_score=result.score,
                keyword_score=0.0,
                hybrid_score=result.score,
                rank=i + 1
            )
            hybrid_results.append(hybrid_result)
            
        return hybrid_results
        
    def _keyword_only_search(self, query_text: str, top_k: int) -> List[HybridSearchResult]:
        """キーワード検索のみ"""
        
        # キーワード検索を実行
        keyword_results = self.keyword_engine.search(query_text, top_k=top_k)
        
        # 結果を変換
        hybrid_results = []
        for i, (doc_id, score) in enumerate(keyword_results):
            # ベクトルストアから文書情報を取得
            # 簡易実装: IDで検索（実際にはより効率的な方法が必要）
            hybrid_result = HybridSearchResult(
                id=doc_id,
                text="",  # キーワード検索では詳細テキストは取得しない
                metadata={},
                vector_score=0.0,
                keyword_score=score,
                hybrid_score=score,
                rank=i + 1
            )
            hybrid_results.append(hybrid_result)
            
        return hybrid_results
        
    def _hybrid_search(self, 
                      query_text: str, 
                      keywords: List[str],
                      filters: Optional[Dict[str, Any]], 
                      top_k: int) -> List[HybridSearchResult]:
        """ハイブリッド検索（ベクトル + キーワード）"""
        
        # ベクトル検索を実行
        query_embedding = self.embedding_model.encode([query_text], is_query=True)[0]
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # リランキングのため多めに取得
            filters=filters
        )
        
        # キーワード検索を実行
        keyword_results = self.keyword_engine.search(query_text, top_k=top_k * 2)
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        
        # スコアを統合
        hybrid_results = []
        processed_ids = set()
        
        for result in vector_results:
            if result.id in processed_ids:
                continue
                
            vector_score = result.score
            keyword_score = keyword_scores.get(result.id, 0.0)
            
            # ハイブリッドスコアを計算
            hybrid_score = (
                self.vector_weight * vector_score + 
                self.keyword_weight * keyword_score
            )
            
            # 技術用語マッチングによるブースト
            tech_boost = self._calculate_technical_boost(result.text, keywords)
            hybrid_score = hybrid_score * (1.0 + tech_boost)
            
            hybrid_result = HybridSearchResult.from_search_result(
                result=result,
                vector_score=vector_score,
                keyword_score=keyword_score,
                hybrid_score=hybrid_score,
                rank=0  # 後で設定
            )
            hybrid_results.append(hybrid_result)
            processed_ids.add(result.id)
            
        # キーワード検索のみの結果も追加
        for doc_id, keyword_score in keyword_results:
            if doc_id not in processed_ids:
                # ベクトルストアから詳細情報を取得する必要があるが、
                # ここでは簡易実装として空の結果を作成
                hybrid_result = HybridSearchResult(
                    id=doc_id,
                    text="",
                    metadata={},
                    vector_score=0.0,
                    keyword_score=keyword_score,
                    hybrid_score=self.keyword_weight * keyword_score,
                    rank=0
                )
                hybrid_results.append(hybrid_result)
                processed_ids.add(doc_id)
                
        # ハイブリッドスコアでソート
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # ランクを設定
        for i, result in enumerate(hybrid_results[:top_k]):
            result.rank = i + 1
            
        return hybrid_results[:top_k]
        
    def _calculate_technical_boost(self, text: str, keywords: List[str]) -> float:
        """技術用語マッチングによるブースト計算"""
        if not keywords:
            return 0.0
            
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
                
        # マッチング率に基づいてブースト（最大20%）
        match_ratio = matches / len(keywords)
        return min(match_ratio * 0.2, 0.2)
        
    def explain_search(self, query: SearchQuery, top_k: int = 3) -> Dict[str, Any]:
        """検索結果の説明を生成"""
        
        results = self.search(query, top_k)
        
        explanation = {
            'query': query.text,
            'search_type': query.search_type,
            'weights': {
                'vector': self.vector_weight,
                'keyword': self.keyword_weight
            },
            'results': []
        }
        
        for result in results:
            result_explanation = {
                'rank': result.rank,
                'id': result.id,
                'scores': {
                    'vector': result.vector_score,
                    'keyword': result.keyword_score,
                    'hybrid': result.hybrid_score
                },
                'text_preview': result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            explanation['results'].append(result_explanation)
            
        return explanation


# 便利な関数
def create_hybrid_search_engine(vector_store: QdrantVectorStore,
                               embedding_model: EmbeddingModel,
                               **kwargs) -> HybridSearchEngine:
    """ハイブリッド検索エンジンを作成"""
    return HybridSearchEngine(vector_store, embedding_model, **kwargs)


def search_documents(search_engine: HybridSearchEngine,
                    query_text: str,
                    **kwargs) -> List[HybridSearchResult]:
    """文書を検索（便利関数）"""
    query = SearchQuery(text=query_text, **kwargs)
    return search_engine.search(query)