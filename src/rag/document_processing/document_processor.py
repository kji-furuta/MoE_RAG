"""
統合文書前処理システム
PDF処理、テーブル抽出、OCR、チャンキングを統合
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import uuid
from datetime import datetime
from loguru import logger
import hashlib

from .pdf_processor import PDFProcessor, DocumentSection, ExtractedTable, ExtractedFigure
from .table_extractor import TableExtractor, ExtractedTable as TableExtractorTable
from .ocr_processor import OCRProcessor, OCRResult
from .chunking_strategy import ChunkingStrategyFactory, DocumentChunk


@dataclass
class ProcessedDocument:
    """処理済み文書を表すデータクラス"""
    id: str
    filename: str
    file_path: str
    file_hash: str
    processing_timestamp: str
    
    # メタデータ
    metadata: Dict[str, Any]
    
    # 抽出されたコンテンツ
    sections: List[DocumentSection]
    tables: List[ExtractedTable]
    figures: List[ExtractedFigure]
    ocr_results: List[OCRResult]
    chunks: List[DocumentChunk]
    
    # 統計情報
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'processing_timestamp': self.processing_timestamp,
            'metadata': self.metadata,
            'sections': [asdict(section) for section in self.sections],
            'tables': [table.to_dict() for table in self.tables],
            'figures': [figure.to_dict() for figure in self.figures],
            'ocr_results': [asdict(ocr) for ocr in self.ocr_results],
            'chunks': [asdict(chunk) for chunk in self.chunks],
            'stats': self.stats
        }
        
    def save_to_file(self, output_path: str):
        """ファイルに保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            
        logger.info(f"Processed document saved to: {output_path}")
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ProcessedDocument':
        """ファイルから読み込み"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # データクラスに変換
        sections = [DocumentSection(**section) for section in data['sections']]
        
        # テーブルの復元
        tables = []
        for table_data in data['tables']:
            if 'data' in table_data and isinstance(table_data['data'], list):
                # DataFrameを復元
                import pandas as pd
                df = pd.DataFrame(table_data['data'])
                
                table = ExtractedTable(
                    id=table_data['id'],
                    page=table_data['page'],
                    caption=table_data['caption'],
                    data=df,
                    bbox=tuple(table_data['bbox']),
                    confidence_score=table_data.get('confidence_score', 1.0),
                    extraction_method=table_data.get('extraction_method', 'unknown'),
                    metadata=table_data.get('metadata', {})
                )
                tables.append(table)
                
        figures = [ExtractedFigure(**figure) for figure in data['figures']]
        ocr_results = [OCRResult(**ocr) for ocr in data['ocr_results']]
        chunks = [DocumentChunk(**chunk) for chunk in data['chunks']]
        
        return cls(
            id=data['id'],
            filename=data['filename'],
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            processing_timestamp=data['processing_timestamp'],
            metadata=data['metadata'],
            sections=sections,
            tables=tables,
            figures=figures,
            ocr_results=ocr_results,
            chunks=chunks,
            stats=data['stats']
        )


class RoadDesignDocumentProcessor:
    """道路設計文書専用の統合処理システム"""
    
    def __init__(self,
                 chunking_strategy: str = 'road_design',
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 extract_tables: bool = True,
                 extract_figures: bool = True,
                 perform_ocr: bool = True,
                 ocr_languages: List[str] = ['ja', 'en'],
                 output_dir: str = './outputs/processed_documents'):
        """
        Args:
            chunking_strategy: チャンキング戦略
            chunk_size: チャンクサイズ
            chunk_overlap: チャンクのオーバーラップ
            extract_tables: テーブル抽出の実行
            extract_figures: 図表抽出の実行
            perform_ocr: OCR処理の実行
            ocr_languages: OCR対応言語
            output_dir: 出力ディレクトリ
        """
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.perform_ocr = perform_ocr
        self.ocr_languages = ocr_languages
        self.output_dir = Path(output_dir)
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各処理モジュールの初期化
        self.pdf_processor = PDFProcessor(
            extract_images=extract_figures,
            extract_tables=extract_tables
        )
        
        if extract_tables:
            self.table_extractor = TableExtractor()
            
        if perform_ocr:
            self.ocr_processor = OCRProcessor(languages=ocr_languages, gpu=False)
            
        self.chunker = ChunkingStrategyFactory.create(
            chunking_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        logger.info(f"RoadDesignDocumentProcessor initialized")
        
    def process_document(self, 
                        pdf_path: str,
                        document_metadata: Optional[Dict[str, Any]] = None) -> Optional[ProcessedDocument]:
        """文書を統合処理"""
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            logger.info(f"Processing document: {pdf_path}")
            
            # 基本メタデータの設定
            doc_id = str(uuid.uuid4())
            file_hash = self._calculate_file_hash(pdf_path)
            processing_timestamp = datetime.now().isoformat()
            
            base_metadata = {
                'doc_id': doc_id,
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'file_hash': file_hash,
                'processing_timestamp': processing_timestamp,
                'document_type': 'road_design_standard',
                'language': 'ja'
            }
            
            if document_metadata:
                base_metadata.update(document_metadata)
                
            # ステップ1: PDF基本処理
            logger.info("Step 1: PDF processing...")
            pdf_result = self.pdf_processor.process_document(str(pdf_path))
            
            sections = pdf_result['sections']
            basic_tables = pdf_result['tables']
            figures = pdf_result['figures']
            full_text = pdf_result['full_text']
            
            # ステップ2: 高度なテーブル抽出
            enhanced_tables = basic_tables  # デフォルトは基本抽出結果
            if self.extract_tables:
                logger.info("Step 2: Enhanced table extraction...")
                try:
                    extracted_tables = self.table_extractor.extract_tables_from_pdf(str(pdf_path))
                    enhanced_tables = self._merge_table_results(basic_tables, extracted_tables)
                except Exception as e:
                    logger.warning(f"Enhanced table extraction failed: {e}")
                    
            # ステップ3: OCR処理
            ocr_results = []
            if self.perform_ocr and figures:
                logger.info("Step 3: OCR processing...")
                for figure in figures:
                    try:
                        figure_ocr = self.ocr_processor.extract_text_from_image(figure.image)
                        # OCR結果に図のメタデータを追加
                        for ocr_result in figure_ocr:
                            ocr_result.metadata = {
                                'figure_id': figure.id,
                                'figure_page': figure.page,
                                'figure_caption': figure.caption
                            }
                        ocr_results.extend(figure_ocr)
                        
                        # 図のocr_textフィールドを更新
                        figure.ocr_text = ' '.join([ocr.text for ocr in figure_ocr])
                        
                    except Exception as e:
                        logger.warning(f"OCR failed for figure {figure.id}: {e}")
                        
            # ステップ4: テキスト統合とチャンキング
            logger.info("Step 4: Text chunking...")
            
            # 全テキストを統合
            integrated_text = self._integrate_text_content(
                full_text, enhanced_tables, ocr_results
            )
            
            # テキストの長さをログ出力
            logger.info(f"Integrated text length: {len(integrated_text)} characters")
            logger.info(f"First 500 chars: {integrated_text[:500]}...")
            
            # チャンキング実行
            chunks = self.chunker.create_chunks(integrated_text, base_metadata)
            logger.info(f"Generated {len(chunks)} chunks")
            
            # チャンクにセクション情報を付与
            chunks = self._enrich_chunks_with_context(chunks, sections, enhanced_tables, figures)
            
            # ステップ5: 統計情報の計算
            stats = self._calculate_statistics(
                full_text, sections, enhanced_tables, figures, ocr_results, chunks
            )
            
            # 処理済み文書オブジェクトの作成
            processed_doc = ProcessedDocument(
                id=doc_id,
                filename=pdf_path.name,
                file_path=str(pdf_path),
                file_hash=file_hash,
                processing_timestamp=processing_timestamp,
                metadata=base_metadata,
                sections=sections,
                tables=enhanced_tables,
                figures=figures,
                ocr_results=ocr_results,
                chunks=chunks,
                stats=stats
            )
            
            # 自動保存
            output_file = self.output_dir / f"{doc_id}.json"
            processed_doc.save_to_file(output_file)
            
            logger.info(f"Document processing completed: {doc_id}")
            return processed_doc
        
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    def process_multiple_documents(self, 
                                  pdf_paths: List[str],
                                  metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[ProcessedDocument]:
        """複数文書を一括処理"""
        
        if metadata_list is None:
            metadata_list = [{}] * len(pdf_paths)
            
        if len(pdf_paths) != len(metadata_list):
            raise ValueError("pdf_paths and metadata_list must have the same length")
            
        processed_docs = []
        
        for i, (pdf_path, metadata) in enumerate(zip(pdf_paths, metadata_list)):
            logger.info(f"Processing document {i+1}/{len(pdf_paths)}: {pdf_path}")
            
            try:
                processed_doc = self.process_document(pdf_path, metadata)
                processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                
        logger.info(f"Batch processing completed: {len(processed_docs)}/{len(pdf_paths)} documents")
        return processed_docs
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """ファイルのハッシュ値を計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _merge_table_results(self, 
                           basic_tables: List[ExtractedTable],
                           enhanced_tables: List[TableExtractorTable]) -> List[ExtractedTable]:
        """基本テーブル抽出結果と高度な抽出結果をマージ"""
        
        # 高度な抽出結果を優先し、基本結果で補完
        merged_tables = []
        
        # 高度な抽出結果を追加
        for table in enhanced_tables:
            # TableExtractorTableをExtractedTableに変換
            if hasattr(table, 'to_dict'):
                merged_table = ExtractedTable(
                    id=table.id,
                    page=table.page,
                    caption=table.caption,
                    data=table.data,
                    bbox=table.bbox
                )
                merged_tables.append(merged_table)
                
        # 基本結果で補完（重複しないもののみ）
        for basic_table in basic_tables:
            is_duplicate = any(
                table.page == basic_table.page and 
                abs(len(table.data) - len(basic_table.data)) < 2
                for table in merged_tables
            )
            
            if not is_duplicate:
                merged_tables.append(basic_table)
                
        return merged_tables
        
    def _integrate_text_content(self,
                               full_text: str,
                               tables: List[ExtractedTable],
                               ocr_results: List[OCRResult]) -> str:
        """テキスト、テーブル、OCR結果を統合"""
        
        integrated_parts = [full_text]
        
        # テーブルテキストを追加
        for table in tables:
            if hasattr(table, 'to_text'):
                table_text = table.to_text()
            else:
                # 基本的なテーブルテキスト変換
                table_text = f"表 {table.page}ページ: {table.caption}\n"
                table_text += table.data.to_string()
                
            integrated_parts.append(f"\n--- テーブル情報 ---\n{table_text}\n")
            
        # OCRテキストを追加
        if ocr_results:
            ocr_text = "\n--- 図表OCRテキスト ---\n"
            for ocr in ocr_results:
                if ocr.confidence > 0.7:  # 高信頼度のもののみ
                    ocr_text += f"{ocr.text}\n"
            integrated_parts.append(ocr_text)
            
        return '\n'.join(integrated_parts)
        
    def _enrich_chunks_with_context(self,
                                   chunks: List[DocumentChunk],
                                   sections: List[DocumentSection],
                                   tables: List[ExtractedTable],
                                   figures: List[ExtractedFigure]) -> List[DocumentChunk]:
        """チャンクにコンテキスト情報を付与"""
        
        enriched_chunks = []
        
        for chunk in chunks:
            enriched_metadata = chunk.metadata.copy()
            
            # 関連するセクションを特定
            related_sections = [
                section for section in sections
                if section.title.lower() in chunk.text.lower()
            ]
            
            if related_sections:
                enriched_metadata['related_sections'] = [
                    {'title': section.title, 'level': section.level}
                    for section in related_sections
                ]
                
            # 関連するテーブルを特定
            related_tables = [
                table for table in tables
                if any(keyword in chunk.text.lower() 
                       for keyword in ['表', 'table', table.caption.lower()])
            ]
            
            if related_tables:
                enriched_metadata['related_tables'] = [
                    {'id': table.id, 'caption': table.caption, 'page': table.page}
                    for table in related_tables
                ]
                
            # 関連する図を特定
            related_figures = [
                figure for figure in figures
                if any(keyword in chunk.text.lower()
                       for keyword in ['図', 'figure', figure.caption.lower()])
            ]
            
            if related_figures:
                enriched_metadata['related_figures'] = [
                    {'id': figure.id, 'caption': figure.caption, 'page': figure.page}
                    for figure in related_figures
                ]
                
            # 強化されたチャンクを作成
            enriched_chunk = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=enriched_metadata,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                section_title=chunk.section_title,
                chunk_type=chunk.chunk_type
            )
            
            enriched_chunks.append(enriched_chunk)
            
        return enriched_chunks
        
    def _calculate_statistics(self,
                            full_text: str,
                            sections: List[DocumentSection],
                            tables: List[ExtractedTable],
                            figures: List[ExtractedFigure],
                            ocr_results: List[OCRResult],
                            chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """処理統計を計算"""
        
        return {
            'text_stats': {
                'total_characters': len(full_text),
                'total_lines': len(full_text.split('\n')),
                'estimated_pages': len(full_text) // 3000  # 大まかな推定
            },
            'structure_stats': {
                'sections_count': len(sections),
                'tables_count': len(tables),
                'figures_count': len(figures)
            },
            'ocr_stats': {
                'ocr_results_count': len(ocr_results),
                'high_confidence_ocr': len([ocr for ocr in ocr_results if ocr.confidence > 0.8]),
                'average_ocr_confidence': sum(ocr.confidence for ocr in ocr_results) / len(ocr_results) if ocr_results else 0
            },
            'chunking_stats': {
                'chunks_count': len(chunks),
                'average_chunk_size': sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0,
                'chunks_with_tables': len([chunk for chunk in chunks if 'related_tables' in chunk.metadata]),
                'chunks_with_figures': len([chunk for chunk in chunks if 'related_figures' in chunk.metadata])
            }
        }


# 便利な関数
def process_road_design_document(pdf_path: str, **kwargs) -> ProcessedDocument:
    """道路設計文書を処理（便利関数）"""
    processor = RoadDesignDocumentProcessor(**kwargs)
    return processor.process_document(pdf_path)
    
    
def process_multiple_road_design_documents(pdf_paths: List[str], **kwargs) -> List[ProcessedDocument]:
    """複数の道路設計文書を一括処理（便利関数）"""
    processor = RoadDesignDocumentProcessor(**kwargs)
    return processor.process_multiple_documents(pdf_paths)