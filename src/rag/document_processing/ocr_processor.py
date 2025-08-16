"""
OCR処理モジュール
道路設計文書の図表・画像からテキストを抽出
"""

import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path
from loguru import logger
import io
import base64


@dataclass
class OCRResult:
    """OCR結果を格納するデータクラス"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    language: str
    method: str
    
    
@dataclass
class ProcessedImage:
    """処理済み画像情報"""
    image: Image.Image
    preprocessing_steps: List[str]
    enhancement_applied: bool
    noise_reduction_applied: bool


class OCRProcessor:
    """統合OCR処理クラス"""
    
    def __init__(self,
                 languages: List[str] = ['ja', 'en'],
                 use_easyocr: bool = True,
                 use_tesseract: bool = True,
                 confidence_threshold: float = 0.5,
                 gpu: bool = True):
        """
        Args:
            languages: 対応言語リスト
            use_easyocr: EasyOCRを使用するか
            use_tesseract: Tesseractを使用するか
            confidence_threshold: 信頼度閾値
            gpu: GPU使用フラグ
        """
        self.languages = languages
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu
        
        # EasyOCRの初期化
        if self.use_easyocr:
            try:
                # Force CPU usage to avoid CUDA/NCCL errors
                self.easyocr_reader = easyocr.Reader(
                    languages, 
                    gpu=False
                )
                logger.info(f"EasyOCR initialized with languages: {languages} (CPU mode)")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.use_easyocr = False
                
        # Tesseractの設定
        if self.use_tesseract:
            try:
                # 日本語+英語の設定
                self.tesseract_lang = '+'.join(['jpn' if lang == 'ja' else 'eng' for lang in languages])
                # テスト実行
                pytesseract.get_tesseract_version()
                logger.info(f"Tesseract initialized with languages: {self.tesseract_lang}")
            except Exception as e:
                logger.warning(f"Tesseract initialization failed: {e}")
                self.use_tesseract = False
                
        # 道路設計文書特有のパターン
        self.technical_patterns = [
            # 数値と単位
            r'\d+(?:\.\d+)?\s*(?:m|km|mm|cm|km/h|%)',
            # 基準値
            r'[最大最小標準推奨]{1,2}[:：]\s*\d+(?:\.\d+)?',
            # 図表番号
            r'[図表]\s*\d+[\-\.]\d+',
            # 道路関連用語
            r'[車道歩道中央分離帯側道路肩]{2,}',
        ]
        
    def extract_text_from_image(self, 
                               image: Image.Image,
                               preprocess: bool = True) -> List[OCRResult]:
        """画像からテキストを抽出"""
        
        results = []
        
        # 前処理
        if preprocess:
            processed_image = self._preprocess_image(image)
        else:
            processed_image = ProcessedImage(
                image=image,
                preprocessing_steps=[],
                enhancement_applied=False,
                noise_reduction_applied=False
            )
            
        # EasyOCRでの抽出
        if self.use_easyocr:
            easyocr_results = self._extract_with_easyocr(processed_image.image)
            results.extend(easyocr_results)
            
        # Tesseractでの抽出
        if self.use_tesseract:
            tesseract_results = self._extract_with_tesseract(processed_image.image)
            results.extend(tesseract_results)
            
        # 結果の統合と重複除去
        final_results = self._merge_ocr_results(results)
        
        # 道路設計文書特有の後処理
        processed_results = self._post_process_technical_text(final_results)
        
        return processed_results
        
    def extract_text_from_file(self, image_path: str) -> List[OCRResult]:
        """ファイルからテキストを抽出"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path)
        return self.extract_text_from_image(image)
        
    def _preprocess_image(self, image: Image.Image) -> ProcessedImage:
        """画像の前処理"""
        steps = []
        enhanced = False
        noise_reduced = False
        
        # RGBに変換
        if image.mode != 'RGB':
            image = image.convert('RGB')
            steps.append('converted_to_rgb')
            
        # グレースケール変換
        gray_image = image.convert('L')
        steps.append('converted_to_grayscale')
        
        # コントラスト向上
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(1.5)
        enhanced = True
        steps.append('enhanced_contrast')
        
        # シャープネス向上
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
        sharp_image = sharpness_enhancer.enhance(2.0)
        steps.append('enhanced_sharpness')
        
        # ノイズ除去
        filtered_image = sharp_image.filter(ImageFilter.MedianFilter(size=3))
        noise_reduced = True
        steps.append('noise_reduction')
        
        # 解像度向上（小さい画像の場合）
        width, height = filtered_image.size
        if width < 500 or height < 500:
            scale_factor = max(500 / width, 500 / height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            filtered_image = filtered_image.resize(new_size, Image.LANCZOS)
            steps.append(f'upscaled_{scale_factor:.1f}x')
            
        return ProcessedImage(
            image=filtered_image,
            preprocessing_steps=steps,
            enhancement_applied=enhanced,
            noise_reduction_applied=noise_reduced
        )
        
    def _extract_with_easyocr(self, image: Image.Image) -> List[OCRResult]:
        """EasyOCRでテキスト抽出"""
        results = []
        
        try:
            # PIL ImageをNumPy配列に変換
            img_array = np.array(image)
            
            # EasyOCRで検出
            detections = self.easyocr_reader.readtext(
                img_array,
                detail=1,
                paragraph=False
            )
            
            for detection in detections:
                bbox_points, text, confidence = detection
                
                if confidence < self.confidence_threshold:
                    continue
                    
                # バウンディングボックスを計算
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                
                result = OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    language=self._detect_language(text),
                    method='easyocr'
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            
        return results
        
    def _extract_with_tesseract(self, image: Image.Image) -> List[OCRResult]:
        """Tesseractでテキスト抽出"""
        results = []
        
        try:
            # Tesseract設定
            custom_config = f'--oem 3 --psm 6 -l {self.tesseract_lang}'
            
            # テキスト抽出
            text = pytesseract.image_to_string(
                image, 
                config=custom_config
            ).strip()
            
            if text:
                # 詳細情報も取得
                data = pytesseract.image_to_data(
                    image,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # 単語レベルでの結果を組み立て
                word_results = []
                for i in range(len(data['text'])):
                    word_text = data['text'][i].strip()
                    confidence = int(data['conf'][i])
                    
                    if word_text and confidence > self.confidence_threshold * 100:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        word_result = OCRResult(
                            text=word_text,
                            confidence=confidence / 100.0,
                            bbox=(x, y, w, h),
                            language=self._detect_language(word_text),
                            method='tesseract'
                        )
                        word_results.append(word_result)
                        
                # 行レベルで統合
                line_results = self._merge_words_to_lines(word_results)
                results.extend(line_results)
                
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            
        return results
        
    def _merge_words_to_lines(self, word_results: List[OCRResult]) -> List[OCRResult]:
        """単語を行レベルで統合"""
        if not word_results:
            return []
            
        # Y座標でソート
        sorted_words = sorted(word_results, key=lambda x: x.bbox[1])
        
        lines = []
        current_line_words = [sorted_words[0]]
        current_y = sorted_words[0].bbox[1]
        y_tolerance = 10  # Y座標の許容範囲
        
        for word in sorted_words[1:]:
            word_y = word.bbox[1]
            
            # 同じ行とみなす
            if abs(word_y - current_y) <= y_tolerance:
                current_line_words.append(word)
            else:
                # 新しい行
                if current_line_words:
                    line_result = self._merge_word_list(current_line_words)
                    lines.append(line_result)
                    
                current_line_words = [word]
                current_y = word_y
                
        # 最後の行
        if current_line_words:
            line_result = self._merge_word_list(current_line_words)
            lines.append(line_result)
            
        return lines
        
    def _merge_word_list(self, words: List[OCRResult]) -> OCRResult:
        """単語リストを1つのOCRResultに統合"""
        if not words:
            return None
            
        # X座標でソート
        sorted_words = sorted(words, key=lambda x: x.bbox[0])
        
        # テキストを結合
        merged_text = ' '.join([word.text for word in sorted_words])
        
        # 信頼度の平均
        avg_confidence = sum([word.confidence for word in sorted_words]) / len(sorted_words)
        
        # バウンディングボックスを計算
        min_x = min([word.bbox[0] for word in sorted_words])
        min_y = min([word.bbox[1] for word in sorted_words])
        max_x = max([word.bbox[0] + word.bbox[2] for word in sorted_words])
        max_y = max([word.bbox[1] + word.bbox[3] for word in sorted_words])
        
        return OCRResult(
            text=merged_text,
            confidence=avg_confidence,
            bbox=(min_x, min_y, max_x - min_x, max_y - min_y),
            language=sorted_words[0].language,
            method=sorted_words[0].method
        )
        
    def _merge_ocr_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """複数のOCR結果を統合"""
        if not results:
            return []
            
        # 重複を除去
        unique_results = []
        
        for result in results:
            is_duplicate = False
            
            for unique_result in unique_results:
                # テキストが類似し、位置が近い場合は重複とみなす
                if self._are_results_similar(result, unique_result):
                    # より信頼度の高い結果を採用
                    if result.confidence > unique_result.confidence:
                        unique_results.remove(unique_result)
                        unique_results.append(result)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_results.append(result)
                
        return sorted(unique_results, key=lambda x: x.confidence, reverse=True)
        
    def _are_results_similar(self, result1: OCRResult, result2: OCRResult) -> bool:
        """2つのOCR結果が類似しているかチェック"""
        # テキストの類似度
        text_similarity = self._calculate_text_similarity(result1.text, result2.text)
        
        # 位置の類似度
        bbox_similarity = self._calculate_bbox_overlap(result1.bbox, result2.bbox)
        
        return text_similarity > 0.8 and bbox_similarity > 0.5
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算"""
        if not text1 or not text2:
            return 0.0
            
        # 簡易的なJaccard類似度
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """バウンディングボックスの重複率を計算"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 重複領域の計算
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # 全体領域
        area1 = w1 * h1
        area2 = w2 * h2
        total_area = area1 + area2 - overlap_area
        
        return overlap_area / total_area if total_area > 0 else 0.0
        
    def _detect_language(self, text: str) -> str:
        """テキストの言語を推定"""
        # 日本語文字が含まれているかチェック
        japanese_chars = re.findall(r'[ひらがなカタカナ漢字]', text)
        
        if japanese_chars:
            return 'ja'
        else:
            return 'en'
            
    def _post_process_technical_text(self, results: List[OCRResult]) -> List[OCRResult]:
        """技術文書特有の後処理"""
        processed_results = []
        
        for result in results:
            processed_text = result.text
            
            # 数値の正規化
            processed_text = self._normalize_numbers(processed_text)
            
            # 単位の正規化
            processed_text = self._normalize_units(processed_text)
            
            # 専門用語の修正
            processed_text = self._correct_technical_terms(processed_text)
            
            # OCR特有のエラー修正
            processed_text = self._fix_ocr_errors(processed_text)
            
            if processed_text.strip():  # 空でない場合のみ追加
                processed_result = OCRResult(
                    text=processed_text,
                    confidence=result.confidence,
                    bbox=result.bbox,
                    language=result.language,
                    method=result.method
                )
                processed_results.append(processed_result)
                
        return processed_results
        
    def _normalize_numbers(self, text: str) -> str:
        """数値の正規化"""
        # 全角数字を半角に変換
        text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        
        # スペースの正規化
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # 数字間のスペースを削除
        
        return text
        
    def _normalize_units(self, text: str) -> str:
        """単位の正規化"""
        unit_mappings = {
            'メートル': 'm',
            'キロメートル': 'km',
            'ミリメートル': 'mm',
            'センチメートル': 'cm',
            'パーセント': '%',
            'キロ毎時': 'km/h'
        }
        
        for japanese, english in unit_mappings.items():
            text = text.replace(japanese, english)
            
        return text
        
    def _correct_technical_terms(self, text: str) -> str:
        """専門用語の修正"""
        # よくあるOCRエラーの修正
        corrections = {
            '車道幅員': '車道幅員',
            '歩道幅員': '歩道幅員',
            '中央分離帯': '中央分離帯',
            '路肩幅員': '路肩幅員',
            '設計速度': '設計速度',
            '曲線半径': '曲線半径'
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
            
        return text
        
    def _fix_ocr_errors(self, text: str) -> str:
        """一般的なOCRエラーの修正"""
        # よくある文字の誤認識
        error_mappings = {
            'O': '0',  # 文脈によっては数字
            'l': '1',  # 文脈によっては数字
            'rn': 'm',
            '|': 'I'
        }
        
        # 数値文脈での修正
        if re.search(r'\d', text):
            for wrong, correct in error_mappings.items():
                text = text.replace(wrong, correct)
                
        return text


# 便利な関数
def extract_text_from_image(image_path: str, **kwargs) -> List[OCRResult]:
    """画像からテキストを抽出（便利関数）"""
    processor = OCRProcessor(**kwargs)
    return processor.extract_text_from_file(image_path)
    
    
def extract_text_from_pil_image(image: Image.Image, **kwargs) -> List[OCRResult]:
    """PIL Imageからテキストを抽出（便利関数）"""
    processor = OCRProcessor(**kwargs)
    return processor.extract_text_from_image(image)