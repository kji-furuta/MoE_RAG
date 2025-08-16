"""
道路設計文書専用のチャンキング戦略モジュール
技術文書の構造を理解した意味的な分割を実行
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    """文書チャンクを表すデータクラス"""
    id: str
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    section_title: Optional[str] = None
    chunk_type: str = "paragraph"  # paragraph, table, figure, list, heading


class ChunkingStrategy(ABC):
    """チャンキング戦略の基底クラス"""
    
    @abstractmethod
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """テキストをチャンクに分割"""
        pass


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """固定サイズチャンキング戦略"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """固定サイズでテキストを分割"""
        chunks = []
        text_length = len(text)
        
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # 文境界で調整
            if end < text_length:
                # 最後の句点または改行で分割
                for i in range(end, start, -1):
                    if text[i] in '。\n':
                        end = i + 1
                        break
                        
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = DocumentChunk(
                    id=f"{metadata.get('doc_id', 'doc')}_{chunk_id}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_id += 1
                
            start = max(start + self.chunk_size - self.overlap, end)
            
        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """意味的チャンキング戦略（Spacyベース）"""
    
    def __init__(self, 
                 model_name: str = "ja_core_news_lg",
                 similarity_threshold: float = 0.7,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000):
        """
        Args:
            model_name: Spacyモデル名
            similarity_threshold: 意味的類似度の閾値
            min_chunk_size: 最小チャンクサイズ
            max_chunk_size: 最大チャンクサイズ
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Spacy model {model_name} not found. Using fallback.")
            self.nlp = None
            
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """意味的類似度に基づいてテキストを分割"""
        if not self.nlp:
            # フォールバック: 文単位分割
            return self._sentence_based_chunking(text, metadata)
            
        # 文単位に分割
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return []
            
        # 文ベクトルを計算
        sentence_vectors = [sent.vector for sent in doc.sents if sent.text.strip()]
        
        # 意味的類似度に基づいてグループ化
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_start = 0
        
        for i in range(1, len(sentences)):
            # 前の文との類似度を計算
            similarity = cosine_similarity(
                [sentence_vectors[i-1]], 
                [sentence_vectors[i]]
            )[0][0]
            
            current_chunk_text = ' '.join(current_chunk_sentences)
            
            # 類似度が閾値を下回るか、最大サイズを超えた場合は新しいチャンクを作成
            if (similarity < self.similarity_threshold or 
                len(current_chunk_text) > self.max_chunk_size):
                
                if len(current_chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk_sentences,
                        current_chunk_start,
                        text,
                        metadata,
                        len(chunks)
                    )
                    chunks.append(chunk)
                    
                current_chunk_sentences = [sentences[i]]
                current_chunk_start = i
            else:
                current_chunk_sentences.append(sentences[i])
                
        # 最後のチャンクを追加
        if current_chunk_sentences:
            current_chunk_text = ' '.join(current_chunk_sentences)
            if len(current_chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_chunk_sentences,
                    current_chunk_start,
                    text,
                    metadata,
                    len(chunks)
                )
                chunks.append(chunk)
                
        return chunks
        
    def _sentence_based_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """文ベースのフォールバックチャンキング"""
        # 簡易的な文分割
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if (current_size + len(sentence) > self.max_chunk_size and 
                current_chunk and 
                current_size >= self.min_chunk_size):
                
                chunk_text = ''.join(current_chunk)
                chunk = DocumentChunk(
                    id=f"{metadata.get('doc_id', 'doc')}_{len(chunks)}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text)
                )
                chunks.append(chunk)
                
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
                
        # 最後のチャンク
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = DocumentChunk(
                    id=f"{metadata.get('doc_id', 'doc')}_{len(chunks)}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text)
                )
                chunks.append(chunk)
                
        return chunks
        
    def _create_chunk(self, 
                     sentences: List[str], 
                     sentence_start_idx: int,
                     full_text: str, 
                     metadata: Dict[str, Any], 
                     chunk_idx: int) -> DocumentChunk:
        """チャンクを作成"""
        chunk_text = ' '.join(sentences)
        start_char = full_text.find(sentences[0])
        end_char = start_char + len(chunk_text)
        
        return DocumentChunk(
            id=f"{metadata.get('doc_id', 'doc')}_{chunk_idx}",
            text=chunk_text,
            metadata=metadata.copy(),
            start_char=start_char,
            end_char=end_char
        )


class RoadDesignDocumentChunker(ChunkingStrategy):
    """道路設計文書専用のチャンキング戦略"""
    
    def __init__(self,
                 chunk_size: int = 512,
                 overlap: int = 128,
                 min_chunk_size: int = 100):
        """
        Args:
            chunk_size: 標準チャンクサイズ
            overlap: オーバーラップサイズ
            min_chunk_size: 最小チャンクサイズ
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # 道路設計文書の構造パターン
        self.section_patterns = [
            (r'^第([0-9０-９]+)[章][\s　]*(.+)', 1, 'chapter'),
            (r'^第([0-9０-９]+)[節][\s　]*(.+)', 2, 'section'),
            (r'^第([0-9０-９]+)[項][\s　]*(.+)', 3, 'subsection'),
            (r'^(\d+)\.\s*(.+)', 1, 'numbered_section'),
            (r'^(\d+\.\d+)\s*(.+)', 2, 'numbered_subsection'),
            (r'^[（(]([0-9０-９]+)[）)]\s*(.+)', 3, 'item'),
        ]
        
        # 表・図の参照パターン
        self.table_patterns = [
            r'表[\s　]*[0-9０-９\-\.]+',
            r'Table[\s　]*[0-9\-\.]+',
            r'第[0-9０-９]+表'
        ]
        
        self.figure_patterns = [
            r'図[\s　]*[0-9０-９\-\.]+',
            r'Figure[\s　]*[0-9\-\.]+',
            r'第[0-9０-９]+図'
        ]
        
        # 数値・基準値パターン
        self.numerical_patterns = [
            r'\d+(?:\.\d+)?\s*[m|km|mm|cm](?![a-zA-Z])',  # 距離
            r'\d+(?:\.\d+)?\s*%',  # パーセント
            r'\d+(?:\.\d+)?\s*km/h',  # 速度
            r'半径\s*:\s*\d+(?:\.\d+)?\s*m',  # 曲線半径
            r'勾配\s*:\s*\d+(?:\.\d+)?\s*%',  # 勾配
        ]
        
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """道路設計文書の構造を考慮したチャンキング"""
        chunks = []
        lines = text.split('\n')
        
        current_section = None
        current_chunk_lines = []
        current_chunk_start = 0
        line_start_pos = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # セクションヘッダーの検出
            section_info = self._detect_section(line)
            
            # セクションが変わった場合、現在のチャンクを保存
            if section_info and current_chunk_lines:
                chunk = self._create_chunk_from_lines(
                    current_chunk_lines,
                    current_chunk_start,
                    line_start_pos,
                    metadata,
                    len(chunks),
                    current_section
                )
                if chunk:
                    chunks.append(chunk)
                    
                current_chunk_lines = []
                current_chunk_start = line_start_pos
                current_section = section_info
                
            current_chunk_lines.append(line)
            
            # チャンクサイズが上限に達した場合
            current_text = '\n'.join(current_chunk_lines)
            if len(current_text) > self.chunk_size:
                # 表や図の参照を含む場合は特別処理
                if self._contains_table_or_figure_reference(current_text):
                    chunk_type = 'reference'
                else:
                    chunk_type = 'paragraph'
                    
                chunk = self._create_chunk_from_lines(
                    current_chunk_lines[:-1],  # 最後の行は次のチャンクに
                    current_chunk_start,
                    line_start_pos,
                    metadata,
                    len(chunks),
                    current_section,
                    chunk_type
                )
                if chunk:
                    chunks.append(chunk)
                    
                # オーバーラップを考慮して次のチャンクを開始
                overlap_lines = self._get_overlap_lines(
                    current_chunk_lines, 
                    self.overlap
                )
                current_chunk_lines = overlap_lines + [line]
                current_chunk_start = line_start_pos - sum(len(l) for l in overlap_lines)
                
            line_start_pos += len(line) + 1  # +1 for newline
            
        # 最後のチャンクを処理
        if current_chunk_lines:
            chunk = self._create_chunk_from_lines(
                current_chunk_lines,
                current_chunk_start,
                line_start_pos,
                metadata,
                len(chunks),
                current_section
            )
            if chunk:
                chunks.append(chunk)
                
        return chunks
        
    def _detect_section(self, line: str) -> Optional[Dict[str, Any]]:
        """セクションヘッダーを検出"""
        for pattern, level, section_type in self.section_patterns:
            match = re.match(pattern, line)
            if match:
                return {
                    'level': level,
                    'type': section_type,
                    'title': match.group(2) if len(match.groups()) > 1 else line,
                    'number': match.group(1)
                }
        return None
        
    def _contains_table_or_figure_reference(self, text: str) -> bool:
        """表や図の参照を含むかチェック"""
        for pattern in self.table_patterns + self.figure_patterns:
            if re.search(pattern, text):
                return True
        return False
        
    def _contains_numerical_data(self, text: str) -> bool:
        """数値データを含むかチェック"""
        for pattern in self.numerical_patterns:
            if re.search(pattern, text):
                return True
        return False
        
    def _get_overlap_lines(self, lines: List[str], overlap_chars: int) -> List[str]:
        """オーバーラップ用の行を取得"""
        overlap_lines = []
        total_chars = 0
        
        for line in reversed(lines):
            if total_chars + len(line) > overlap_chars:
                break
            overlap_lines.insert(0, line)
            total_chars += len(line)
            
        return overlap_lines
        
    def _create_chunk_from_lines(self,
                                lines: List[str],
                                start_pos: int,
                                end_pos: int,
                                base_metadata: Dict[str, Any],
                                chunk_idx: int,
                                section_info: Optional[Dict[str, Any]] = None,
                                chunk_type: str = "paragraph") -> Optional[DocumentChunk]:
        """行リストからチャンクを作成"""
        text = '\n'.join(lines).strip()
        
        if len(text) < self.min_chunk_size:
            return None
            
        metadata = base_metadata.copy()
        
        # セクション情報を追加
        if section_info:
            metadata.update({
                'section_level': section_info['level'],
                'section_type': section_info['type'],
                'section_title': section_info['title'],
                'section_number': section_info['number']
            })
            
        # 特徴的な内容を検出してメタデータに追加
        if self._contains_numerical_data(text):
            metadata['contains_numerical_data'] = True
            
        if self._contains_table_or_figure_reference(text):
            metadata['contains_references'] = True
            
        # 重要度スコアを計算
        importance_score = self._calculate_importance_score(text)
        metadata['importance_score'] = importance_score
        
        return DocumentChunk(
            id=f"{base_metadata.get('doc_id', 'doc')}_{chunk_idx}",
            text=text,
            metadata=metadata,
            start_char=start_pos,
            end_char=end_pos,
            section_title=section_info['title'] if section_info else None,
            chunk_type=chunk_type
        )
        
    def _calculate_importance_score(self, text: str) -> float:
        """チャンクの重要度スコアを計算"""
        score = 0.0
        
        # 数値データの存在
        if self._contains_numerical_data(text):
            score += 0.3
            
        # 表・図の参照
        if self._contains_table_or_figure_reference(text):
            score += 0.2
            
        # キーワードベースの重要度
        important_keywords = [
            '基準', '規定', '規格', '仕様', '設計', '計算',
            '最小', '最大', '標準', '推奨', '必須', '義務'
        ]
        
        keyword_count = sum(1 for keyword in important_keywords if keyword in text)
        score += min(keyword_count * 0.1, 0.3)
        
        # 文章の長さによる調整（中程度の長さが最適）
        text_length = len(text)
        if 200 <= text_length <= 800:
            score += 0.2
        elif text_length < 100:
            score -= 0.1
            
        return min(score, 1.0)


# ファクトリークラス
class ChunkingStrategyFactory:
    """チャンキング戦略のファクトリクラス"""
    
    STRATEGIES = {
        'fixed': FixedSizeChunkingStrategy,
        'semantic': SemanticChunkingStrategy,
        'road_design': RoadDesignDocumentChunker
    }
    
    @classmethod
    def create(cls, strategy_type: str = 'road_design', **kwargs) -> ChunkingStrategy:
        """チャンキング戦略のインスタンスを作成"""
        if strategy_type not in cls.STRATEGIES:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available strategies: {list(cls.STRATEGIES.keys())}"
            )
            
        strategy_class = cls.STRATEGIES[strategy_type]
        return strategy_class(**kwargs)


# 便利な関数
def chunk_document(text: str, 
                  metadata: Dict[str, Any],
                  strategy: str = 'road_design',
                  **kwargs) -> List[DocumentChunk]:
    """文書をチャンクに分割"""
    chunker = ChunkingStrategyFactory.create(strategy, **kwargs)
    return chunker.create_chunks(text, metadata)