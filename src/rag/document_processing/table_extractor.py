"""
テーブル抽出モジュール
道路設計文書からテーブルを抽出・構造化
"""

import pandas as pd
import camelot
import tabula
import pdfplumber
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
from loguru import logger
import numpy as np
from PIL import Image
import cv2


@dataclass
class ExtractedTable:
    """抽出されたテーブル情報"""
    id: str
    page: int
    caption: str
    data: pd.DataFrame
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence_score: float
    extraction_method: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（JSON serializable）"""
        return {
            'id': self.id,
            'page': self.page,
            'caption': self.caption,
            'data': self.data.to_dict('records') if isinstance(self.data, pd.DataFrame) else self.data,
            'columns': list(self.data.columns) if isinstance(self.data, pd.DataFrame) else [],
            'shape': list(self.data.shape) if isinstance(self.data, pd.DataFrame) else [0, 0],
            'bbox': list(self.bbox),
            'confidence_score': self.confidence_score,
            'extraction_method': self.extraction_method,
            'metadata': self.metadata
        }
        
    def to_text(self) -> str:
        """テキスト形式に変換（検索用）"""
        text_parts = []
        
        # キャプション
        if self.caption:
            text_parts.append(f"表題: {self.caption}")
            
        # 列名
        text_parts.append(f"列名: {', '.join(self.data.columns)}")
        
        # データ
        for _, row in self.data.iterrows():
            row_text = ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_text)
            
        return '\n'.join(text_parts)


class TableExtractor:
    """テーブル抽出の統合クラス"""
    
    def __init__(self, 
                 use_camelot: bool = True,
                 use_tabula: bool = True,
                 use_pdfplumber: bool = True,
                 confidence_threshold: float = 0.5):
        """
        Args:
            use_camelot: Camelotを使用するか
            use_tabula: Tabulaを使用するか
            use_pdfplumber: pdfplumberを使用するか
            confidence_threshold: 信頼度の閾値
        """
        self.use_camelot = use_camelot
        self.use_tabula = use_tabula
        self.use_pdfplumber = use_pdfplumber
        self.confidence_threshold = confidence_threshold
        
        # テーブルキャプションのパターン
        self.caption_patterns = [
            r'表[\s　]*([0-9０-９\-\.]+)[\s　]*[：:]?\s*(.+)',
            r'Table[\s　]*([0-9\-\.]+)[\s　]*[：:]?\s*(.+)',
            r'第([0-9０-９]+)表[\s　]*[：:]?\s*(.+)',
            r'([表Table]+[\s　]*[0-9０-９\-\.]+.*?)(?=\n|$)'
        ]
        
    def extract_tables_from_pdf(self, pdf_path: str) -> List[ExtractedTable]:
        """PDFからテーブルを抽出"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Extracting tables from: {pdf_path}")
        
        all_tables = []
        
        # 各抽出手法を試行
        if self.use_camelot:
            camelot_tables = self._extract_with_camelot(pdf_path)
            all_tables.extend(camelot_tables)
            
        if self.use_tabula:
            tabula_tables = self._extract_with_tabula(pdf_path)
            all_tables.extend(tabula_tables)
            
        if self.use_pdfplumber:
            pdfplumber_tables = self._extract_with_pdfplumber(pdf_path)
            all_tables.extend(pdfplumber_tables)
            
        # 重複を除去し、信頼度で並び替え
        unique_tables = self._deduplicate_tables(all_tables)
        filtered_tables = [t for t in unique_tables if t.confidence_score >= self.confidence_threshold]
        
        logger.info(f"Extracted {len(filtered_tables)} tables from {pdf_path}")
        return filtered_tables
        
    def _extract_with_camelot(self, pdf_path: Path) -> List[ExtractedTable]:
        """Camelotでテーブル抽出"""
        tables = []
        
        try:
            # Camelotでテーブル抽出
            camelot_tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                if table.df.empty:
                    continue
                    
                # データフレームをクリーンアップ
                cleaned_df = self._clean_dataframe(table.df)
                
                if cleaned_df.empty:
                    continue
                    
                # キャプションを推定
                caption = self._find_table_caption(pdf_path, table.page, i)
                
                # 信頼度スコアを計算
                confidence = self._calculate_confidence_score(
                    cleaned_df, 
                    table.accuracy, 
                    'camelot'
                )
                
                extracted_table = ExtractedTable(
                    id=f"camelot_p{table.page}_{i}",
                    page=table.page,
                    caption=caption,
                    data=cleaned_df,
                    bbox=self._get_table_bbox(table),
                    confidence_score=confidence,
                    extraction_method='camelot',
                    metadata={
                        'accuracy': table.accuracy,
                        'whitespace': table.whitespace,
                        'order': table.order
                    }
                )
                tables.append(extracted_table)
                
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            
        return tables
        
    def _extract_with_tabula(self, pdf_path: Path) -> List[ExtractedTable]:
        """Tabulaでテーブル抽出"""
        tables = []
        
        try:
            # Tabulaでテーブル抽出
            tabula_tables = tabula.read_pdf(
                str(pdf_path), 
                pages='all', 
                multiple_tables=True,
                pandas_options={'header': 0}
            )
            
            for i, df in enumerate(tabula_tables):
                if df.empty:
                    continue
                    
                # データフレームをクリーンアップ
                cleaned_df = self._clean_dataframe(df)
                
                if cleaned_df.empty:
                    continue
                    
                # ページ番号を推定（Tabulaは正確なページ番号を返さない場合がある）
                page_num = i + 1  # 簡易的な推定
                
                # キャプションを推定
                caption = self._find_table_caption(pdf_path, page_num, i)
                
                # 信頼度スコアを計算
                confidence = self._calculate_confidence_score(
                    cleaned_df, 
                    None, 
                    'tabula'
                )
                
                extracted_table = ExtractedTable(
                    id=f"tabula_p{page_num}_{i}",
                    page=page_num,
                    caption=caption,
                    data=cleaned_df,
                    bbox=(0, 0, 0, 0),  # Tabulaでは正確なbboxが取得困難
                    confidence_score=confidence,
                    extraction_method='tabula',
                    metadata={'table_index': i}
                )
                tables.append(extracted_table)
                
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
            
        return tables
        
    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[ExtractedTable]:
        """pdfplumberでテーブル抽出"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for i, table_data in enumerate(page_tables):
                        if not table_data or len(table_data) < 2:
                            continue
                            
                        try:
                            # DataFrameに変換
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            
                            # データフレームをクリーンアップ
                            cleaned_df = self._clean_dataframe(df)
                            
                            if cleaned_df.empty:
                                continue
                                
                            # キャプションを推定
                            caption = self._find_table_caption_in_page(page, i)
                            
                            # 信頼度スコアを計算
                            confidence = self._calculate_confidence_score(
                                cleaned_df, 
                                None, 
                                'pdfplumber'
                            )
                            
                            extracted_table = ExtractedTable(
                                id=f"pdfplumber_p{page_num}_{i}",
                                page=page_num,
                                caption=caption,
                                data=cleaned_df,
                                bbox=(0, 0, page.width, page.height),
                                confidence_score=confidence,
                                extraction_method='pdfplumber',
                                metadata={'page_width': page.width, 'page_height': page.height}
                            )
                            tables.append(extracted_table)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process table on page {page_num}: {e}")
                            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            
        return tables
        
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレームをクリーンアップ"""
        if df.empty:
            return df
            
        # 空の行・列を削除
        df = df.dropna(how='all')  # 全てNaNの行を削除
        df = df.dropna(axis=1, how='all')  # 全てNaNの列を削除
        
        # 列名をクリーンアップ
        if not df.empty:
            df.columns = [self._clean_column_name(str(col)) for col in df.columns]
            
            # 重複する列名を処理
            df = self._handle_duplicate_columns(df)
            
            # 数値データの正規化
            df = self._normalize_numeric_data(df)
            
        return df
        
    def _clean_column_name(self, col_name: str) -> str:
        """列名をクリーンアップ"""
        # 改行文字を削除
        col_name = re.sub(r'\n+', ' ', col_name)
        
        # 余分な空白を削除
        col_name = re.sub(r'\s+', ' ', col_name).strip()
        
        # 空の列名に名前を付与
        if not col_name or col_name == 'Unnamed' or col_name.startswith('Unnamed:'):
            col_name = '列'
            
        return col_name
        
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """重複する列名を処理"""
        columns = df.columns.tolist()
        seen = {}
        
        for i, col in enumerate(columns):
            if col in seen:
                seen[col] += 1
                columns[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
                
        df.columns = columns
        return df
        
    def _normalize_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値データを正規化"""
        for col in df.columns:
            # 数値らしき文字列を数値に変換
            df[col] = df[col].apply(self._convert_to_numeric)
            
        return df
        
    def _convert_to_numeric(self, value) -> Any:
        """文字列を数値に変換（可能であれば）"""
        if pd.isna(value):
            return value
            
        value_str = str(value).strip()
        
        # 空文字列の場合
        if not value_str:
            return np.nan
            
        # 数値パターンをチェック
        numeric_patterns = [
            r'^[+-]?\d+$',  # 整数
            r'^[+-]?\d*\.\d+$',  # 小数
            r'^[+-]?\d+\.\d*$',  # 小数（末尾ピリオド）
            r'^[+-]?\d+(?:,\d{3})*(?:\.\d+)?$'  # カンマ区切り数値
        ]
        
        for pattern in numeric_patterns:
            if re.match(pattern, value_str):
                try:
                    # カンマを削除して変換
                    numeric_value = float(value_str.replace(',', ''))
                    return int(numeric_value) if numeric_value.is_integer() else numeric_value
                except ValueError:
                    break
                    
        return value
        
    def _find_table_caption(self, pdf_path: Path, page_num: int, table_index: int) -> str:
        """テーブルのキャプションを推定"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    return self._find_table_caption_in_page(page, table_index)
        except Exception as e:
            logger.warning(f"Failed to find caption: {e}")
            
        return f"表 {table_index + 1}"
        
    def _find_table_caption_in_page(self, page, table_index: int) -> str:
        """ページ内でテーブルのキャプションを検索"""
        text = page.extract_text()
        if not text:
            return f"表 {table_index + 1}"
            
        lines = text.split('\n')
        
        # キャプションパターンをマッチング
        for pattern in self.caption_patterns:
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    return line.strip()
                    
        return f"表 {table_index + 1}"
        
    def _get_table_bbox(self, camelot_table) -> Tuple[float, float, float, float]:
        """Camelotテーブルのbboxを取得"""
        try:
            return camelot_table._bbox
        except:
            return (0, 0, 0, 0)
            
    def _calculate_confidence_score(self, 
                                  df: pd.DataFrame, 
                                  accuracy: Optional[float],
                                  method: str) -> float:
        """信頼度スコアを計算"""
        score = 0.0
        
        # データフレームの品質に基づくスコア
        if not df.empty:
            # 行数・列数
            rows, cols = df.shape
            if rows >= 2 and cols >= 2:
                score += 0.3
                
            # 空白セルの割合
            total_cells = rows * cols
            non_null_cells = df.count().sum()
            fill_ratio = non_null_cells / total_cells if total_cells > 0 else 0
            score += fill_ratio * 0.3
            
            # 数値データの存在
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            if numeric_cols > 0:
                score += min(numeric_cols / cols * 0.2, 0.2)
                
        # 抽出手法固有のスコア
        if method == 'camelot' and accuracy is not None:
            score += accuracy * 0.2
        elif method == 'pdfplumber':
            score += 0.1  # pdfplumberは一般的に安定
        elif method == 'tabula':
            score += 0.05  # tabulaは精度が劣る場合がある
            
        return min(score, 1.0)
        
    def _deduplicate_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """重複テーブルを除去"""
        if not tables:
            return tables
            
        # 同じページの同じようなテーブルを統合
        page_tables = {}
        for table in tables:
            page = table.page
            if page not in page_tables:
                page_tables[page] = []
            page_tables[page].append(table)
            
        unique_tables = []
        for page, page_table_list in page_tables.items():
            # ページ内でのテーブル重複を除去
            unique_page_tables = self._deduplicate_page_tables(page_table_list)
            unique_tables.extend(unique_page_tables)
            
        return sorted(unique_tables, key=lambda x: x.confidence_score, reverse=True)
        
    def _deduplicate_page_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """同一ページ内のテーブル重複を除去"""
        if len(tables) <= 1:
            return tables
            
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            for unique_table in unique_tables:
                # 形状とデータの類似性をチェック
                if self._are_tables_similar(table, unique_table):
                    # より信頼度の高いテーブルを選択
                    if table.confidence_score > unique_table.confidence_score:
                        unique_tables.remove(unique_table)
                        unique_tables.append(table)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_tables.append(table)
                
        return unique_tables
        
    def _are_tables_similar(self, table1: ExtractedTable, table2: ExtractedTable) -> bool:
        """2つのテーブルが類似しているかチェック"""
        # 形状の比較
        if table1.data.shape != table2.data.shape:
            return False
            
        # データの類似性を簡単にチェック
        try:
            # 文字列データとして比較
            str1 = table1.data.to_string()
            str2 = table2.data.to_string()
            
            # 類似度を計算（簡易版）
            common_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2))
            similarity = common_chars / max(len(str1), len(str2))
            
            return similarity > 0.8
            
        except Exception:
            return False


# 便利な関数
def extract_tables_from_pdf(pdf_path: str, **kwargs) -> List[ExtractedTable]:
    """PDFからテーブルを抽出（便利関数）"""
    extractor = TableExtractor(**kwargs)
    return extractor.extract_tables_from_pdf(pdf_path)