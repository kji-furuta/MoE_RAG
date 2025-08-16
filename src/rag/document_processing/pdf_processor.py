"""
PDF処理モジュール
道路設計文書のPDFから構造化データを抽出
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
import pandas as pd
from PIL import Image
import io
import hashlib


@dataclass
class DocumentSection:
    """文書のセクション情報"""
    title: str
    level: int  # 1: 章, 2: 節, 3: 項
    page_start: int
    page_end: int
    content: str
    
    
@dataclass
class ExtractedTable:
    """抽出されたテーブル情報"""
    id: str
    page: int
    caption: str
    data: pd.DataFrame
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（JSON serializable）"""
        return {
            'id': self.id,
            'page': self.page,
            'caption': self.caption,
            'data': self.data.to_dict('records') if isinstance(self.data, pd.DataFrame) else self.data,
            'bbox': list(self.bbox),
            'rows': len(self.data) if isinstance(self.data, pd.DataFrame) else 0,
            'columns': len(self.data.columns) if isinstance(self.data, pd.DataFrame) else 0
        }
    
    
@dataclass
class ExtractedFigure:
    """抽出された図表情報"""
    id: str
    page: int
    caption: str
    image: Image.Image
    bbox: Tuple[float, float, float, float]
    ocr_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（JSON serializable）"""
        # 画像は保存せず、メタデータのみを返す
        return {
            'id': self.id,
            'page': self.page,
            'caption': self.caption,
            'bbox': list(self.bbox),
            'ocr_text': self.ocr_text,
            'has_image': self.image is not None,
            'image_size': self.image.size if self.image else None
        }


class PDFProcessor:
    """道路設計文書用PDFプロセッサ"""
    
    def __init__(self, 
                 extract_images: bool = True,
                 extract_tables: bool = True,
                 dpi: int = 300):
        """
        Args:
            extract_images: 画像を抽出するか
            extract_tables: テーブルを抽出するか
            dpi: 画像抽出時のDPI
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.dpi = dpi
        
        # 道路設計文書の一般的な章節パターン
        self.section_patterns = [
            # 第1章、第2節、第3項などのパターン
            (r'^第([0-9０-９]+)[章][\s　]*(.+)', 1),
            (r'^第([0-9０-９]+)[節][\s　]*(.+)', 2),
            (r'^第([0-9０-９]+)[項][\s　]*(.+)', 3),
            # 1., 1.1, 1.1.1 などのパターン
            (r'^(\d+)\.\s*(.+)', 1),
            (r'^(\d+\.\d+)\s*(.+)', 2),
            (r'^(\d+\.\d+\.\d+)\s*(.+)', 3),
            # （1）、（2）などのパターン
            (r'^[（(]([0-9０-９]+)[）)]\s*(.+)', 3),
        ]
        
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDFドキュメントを処理して構造化データを抽出
        
        Returns:
            {
                'metadata': {...},
                'sections': [...],
                'tables': [...],
                'figures': [...],
                'full_text': str
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        # 結果を格納する辞書
        result = {
            'metadata': {
                'filename': pdf_path.name,
                'file_hash': self._calculate_file_hash(pdf_path),
                'pages': 0
            },
            'sections': [],
            'tables': [],
            'figures': [],
            'full_text': ''
        }
        
        # PyMuPDFで処理
        with fitz.open(pdf_path) as pdf_doc:
            result['metadata']['pages'] = len(pdf_doc)
            
            # ページごとにテキストを抽出
            page_texts = []
            for page_num, page in enumerate(pdf_doc):
                # より詳細なテキスト抽出オプションを使用
                page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                # テキストが空の場合はOCR処理を試行
                if not page_text.strip():
                    logger.warning(f"Page {page_num + 1} has no text, may need OCR")
                    # 代替抽出方法を試行
                    page_text = page.get_text("blocks")
                    if page_text:
                        page_text = '\n'.join([block[4] for block in page_text if block[6] == 0])
                
                page_texts.append(page_text)
                
                # 画像の抽出
                if self.extract_images:
                    figures = self._extract_figures_from_page(page, page_num + 1)
                    result['figures'].extend(figures)
                    
            result['full_text'] = '\n'.join(page_texts)
            
            # テキストが完全に空の場合は警告
            if not result['full_text'].strip():
                logger.error(f"No text extracted from PDF: {pdf_path}")
            
        # pdfplumberでテーブルを抽出
        if self.extract_tables:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = self._extract_tables_from_page(page, page_num + 1)
                    result['tables'].extend(tables)
                    
        # セクション構造を解析
        result['sections'] = self._extract_sections(result['full_text'], page_texts)
        
        logger.info(
            f"Extraction complete: "
            f"{len(result['sections'])} sections, "
            f"{len(result['tables'])} tables, "
            f"{len(result['figures'])} figures"
        )
        
        return result
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """ファイルのハッシュ値を計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _extract_sections(self, full_text: str, page_texts: List[str]) -> List[DocumentSection]:
        """テキストからセクション構造を抽出"""
        sections = []
        lines = full_text.split('\n')
        
        current_page = 1
        page_line_count = 0
        page_total_lines = len(page_texts[0].split('\n')) if page_texts else 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # ページ境界の追跡
            page_line_count += 1
            if page_line_count >= page_total_lines and current_page < len(page_texts):
                current_page += 1
                page_total_lines = len(page_texts[current_page - 1].split('\n'))
                page_line_count = 0
                
            # セクションヘッダーのマッチング
            for pattern, level in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_title = match.group(2) if len(match.groups()) > 1 else line
                    
                    # セクションの内容を収集
                    content_lines = []
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        
                        # 次のセクションが見つかったら終了
                        is_next_section = False
                        for p, _ in self.section_patterns:
                            if re.match(p, next_line):
                                is_next_section = True
                                break
                                
                        if is_next_section:
                            break
                            
                        content_lines.append(lines[j])
                        j += 1
                        
                    section = DocumentSection(
                        title=section_title,
                        level=level,
                        page_start=current_page,
                        page_end=current_page,  # 簡易実装
                        content='\n'.join(content_lines)
                    )
                    sections.append(section)
                    break
                    
        return sections
        
    def _extract_tables_from_page(self, page, page_num: int) -> List[ExtractedTable]:
        """ページからテーブルを抽出"""
        tables = []
        
        page_tables = page.extract_tables()
        for i, table_data in enumerate(page_tables):
            if not table_data or len(table_data) < 2:  # 最低2行必要
                continue
                
            try:
                # DataFrameに変換
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                
                # テーブルIDを生成
                table_id = f"table_p{page_num}_{i+1}"
                
                # キャプションを推定（テーブルの上部テキストから）
                caption = self._find_table_caption(page, i)
                
                table = ExtractedTable(
                    id=table_id,
                    page=page_num,
                    caption=caption,
                    data=df,
                    bbox=(0, 0, page.width, page.height)  # 簡易実装
                )
                tables.append(table)
                
            except Exception as e:
                logger.warning(f"Failed to process table on page {page_num}: {e}")
                
        return tables
        
    def _find_table_caption(self, page, table_index: int) -> str:
        """テーブルのキャプションを推定"""
        # テーブルの上部にあるテキストをキャプションとして扱う
        # 簡易実装
        text = page.extract_text()
        lines = text.split('\n')
        
        # "表" や "Table" を含む行を探す
        caption_patterns = [
            r'表[\s　]*[0-9０-９\-\.]+',
            r'Table[\s　]*[0-9\-\.]+',
            r'第[0-9０-９]+表'
        ]
        
        for line in lines:
            for pattern in caption_patterns:
                if re.search(pattern, line):
                    return line.strip()
                    
        return f"表 {table_index + 1}"
        
    def _extract_figures_from_page(self, page, page_num: int) -> List[ExtractedFigure]:
        """ページから図表を抽出"""
        figures = []
        
        # 画像リストを取得
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # 画像データを取得
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # PILイメージに変換
                image = Image.open(io.BytesIO(image_bytes))
                
                # 図のIDを生成
                figure_id = f"figure_p{page_num}_{img_index+1}"
                
                # キャプションを推定
                caption = self._find_figure_caption(page, img_index)
                
                figure = ExtractedFigure(
                    id=figure_id,
                    page=page_num,
                    caption=caption,
                    image=image,
                    bbox=(0, 0, image.width, image.height)
                )
                figures.append(figure)
                
            except Exception as e:
                logger.warning(f"Failed to extract image on page {page_num}: {e}")
                
        return figures
        
    def _find_figure_caption(self, page, figure_index: int) -> str:
        """図のキャプションを推定"""
        text = page.get_text()
        lines = text.split('\n')
        
        # "図" や "Figure" を含む行を探す
        caption_patterns = [
            r'図[\s　]*[0-9０-９\-\.]+',
            r'Figure[\s　]*[0-9\-\.]+',
            r'第[0-9０-９]+図'
        ]
        
        for line in lines:
            for pattern in caption_patterns:
                if re.search(pattern, line):
                    return line.strip()
                    
        return f"図 {figure_index + 1}"
        
    def extract_text_with_layout(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        レイアウト情報を保持したテキスト抽出
        段落、見出し、リストなどの構造を識別
        """
        pdf_path = Path(pdf_path)
        blocks = []
        
        with fitz.open(pdf_path) as pdf_doc:
            for page_num, page in enumerate(pdf_doc):
                # テキストブロックを取得
                page_blocks = page.get_text("dict")
                
                for block in page_blocks["blocks"]:
                    if "lines" in block:  # テキストブロック
                        block_text = ""
                        block_bbox = block["bbox"]
                        
                        # フォント情報を解析して見出しを識別
                        fonts = []
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                                fonts.append({
                                    "size": span["size"],
                                    "flags": span["flags"],  # 太字、斜体など
                                    "font": span["font"]
                                })
                                
                        if block_text.strip():
                            # ブロックタイプを推定
                            block_type = self._estimate_block_type(
                                block_text, fonts, block_bbox
                            )
                            
                            blocks.append({
                                "page": page_num + 1,
                                "text": block_text.strip(),
                                "type": block_type,
                                "bbox": block_bbox,
                                "fonts": fonts
                            })
                            
        return blocks
        
    def _estimate_block_type(self, text: str, fonts: List[Dict], bbox: Tuple) -> str:
        """ブロックのタイプを推定（見出し、本文、リストなど）"""
        # フォントサイズの平均を計算
        if fonts:
            avg_font_size = sum(f["size"] for f in fonts) / len(fonts)
        else:
            avg_font_size = 10
            
        # 太字フラグをチェック
        is_bold = any(f["flags"] & 2**4 for f in fonts)  # 16 = bold flag
        
        # テキストパターンをチェック
        if re.match(r'^第[0-9０-９]+[章節項]', text):
            return "heading1"
        elif re.match(r'^\d+\.', text) and avg_font_size > 12:
            return "heading2"
        elif re.match(r'^[・◆◇○●▪▫■□]', text):
            return "list_item"
        elif avg_font_size > 14 or is_bold:
            return "heading3"
        else:
            return "paragraph"