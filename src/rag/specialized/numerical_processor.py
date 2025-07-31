"""
数値処理エンジン
道路設計における数値・計算式の処理に特化したモジュール
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
logger = logging.getLogger(__name__)
# import numpy as np  # Optional dependency


@dataclass
class NumericalValue:
    """数値データ"""
    value: float
    unit: str
    original_text: str
    confidence: float = 1.0
    context: Optional[str] = None
    value_type: Optional[str] = None  # speed, length, gradient, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'value': self.value,
            'unit': self.unit,
            'original_text': self.original_text,
            'confidence': self.confidence,
            'context': self.context,
            'value_type': self.value_type
        }


@dataclass
class Formula:
    """計算式データ"""
    expression: str
    variables: Dict[str, float]
    result: Optional[float] = None
    unit: Optional[str] = None
    formula_type: Optional[str] = None  # curve_radius, stopping_distance, etc.
    source: Optional[str] = None
    
    def evaluate(self) -> float:
        """式を評価"""
        try:
            # 変数を置換
            expr = self.expression
            for var, val in self.variables.items():
                expr = expr.replace(var, str(val))
            
            # 安全な評価
            result = eval(expr, {"__builtins__": {}}, {
                "sqrt": math.sqrt,
                "pow": pow,
                "abs": abs,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi
            })
            
            self.result = float(result)
            return self.result
            
        except Exception as e:
            logger.error(f"Formula evaluation failed: {e}")
            raise


class NumericalExtractor:
    """数値抽出器"""
    
    def __init__(self):
        """数値抽出器の初期化"""
        
        # 数値パターン（単位付き）
        self.numerical_patterns = {
            # 速度
            'speed': [
                (r'(\d+(?:\.\d+)?)\s*km/h', 'km/h'),
                (r'(\d+(?:\.\d+)?)\s*キロメートル毎時', 'km/h'),
                (r'速度\s*[:：]\s*(\d+(?:\.\d+)?)', 'km/h'),
                (r'V\s*=\s*(\d+(?:\.\d+)?)', 'km/h'),
            ],
            
            # 長さ・距離
            'length': [
                (r'(\d+(?:\.\d+)?)\s*m(?![/\w])', 'm'),
                (r'(\d+(?:\.\d+)?)\s*メートル', 'm'),
                (r'(\d+(?:\.\d+)?)\s*km(?![/\w])', 'km'),
                (r'(\d+(?:\.\d+)?)\s*キロメートル', 'km'),
                (r'(\d+(?:\.\d+)?)\s*mm', 'mm'),
                (r'(\d+(?:\.\d+)?)\s*cm', 'cm'),
            ],
            
            # 勾配
            'gradient': [
                (r'(\d+(?:\.\d+)?)\s*%', '%'),
                (r'(\d+(?:\.\d+)?)\s*パーセント', '%'),
                (r'勾配\s*[:：]\s*(\d+(?:\.\d+)?)', '%'),
                (r'i\s*=\s*(\d+(?:\.\d+)?)', '%'),
            ],
            
            # 角度
            'angle': [
                (r'(\d+(?:\.\d+)?)\s*度', '度'),
                (r'(\d+(?:\.\d+)?)\s*°', '°'),
                (r'(\d+(?:\.\d+)?)\s*deg', 'deg'),
            ],
            
            # 面積
            'area': [
                (r'(\d+(?:\.\d+)?)\s*m2', 'm²'),
                (r'(\d+(?:\.\d+)?)\s*m²', 'm²'),
                (r'(\d+(?:\.\d+)?)\s*平方メートル', 'm²'),
            ],
            
            # 荷重
            'load': [
                (r'(\d+(?:\.\d+)?)\s*kN', 'kN'),
                (r'(\d+(?:\.\d+)?)\s*t', 't'),
                (r'(\d+(?:\.\d+)?)\s*トン', 't'),
            ]
        }
        
        # 文脈キーワード
        self.context_keywords = {
            'speed': ['設計速度', '走行速度', '制限速度', '速度'],
            'length': ['幅員', '延長', '距離', '半径', '長さ'],
            'gradient': ['縦断勾配', '横断勾配', '勾配', '傾斜'],
            'angle': ['交角', '偏角', '角度'],
            'area': ['面積', '断面積'],
            'load': ['荷重', '軸重', '輪荷重']
        }
        
    def extract_numerical_values(self, text: str) -> List[NumericalValue]:
        """テキストから数値を抽出"""
        
        numerical_values = []
        
        for value_type, patterns in self.numerical_patterns.items():
            for pattern, unit in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    value_str = match.group(1)
                    
                    try:
                        value = float(value_str)
                        
                        # 文脈を抽出
                        context = self._extract_context(text, match.start(), match.end())
                        
                        # 信頼度を計算
                        confidence = self._calculate_confidence(
                            value_type, context, text
                        )
                        
                        numerical_value = NumericalValue(
                            value=value,
                            unit=unit,
                            original_text=match.group(0),
                            confidence=confidence,
                            context=context,
                            value_type=value_type
                        )
                        
                        numerical_values.append(numerical_value)
                        
                    except ValueError:
                        logger.warning(f"Failed to parse numerical value: {value_str}")
                        
        # 重複を除去（同じ値と単位の組み合わせ）
        unique_values = self._remove_duplicates(numerical_values)
        
        return unique_values
        
    def _extract_context(self, text: str, start: int, end: int, 
                        window: int = 50) -> str:
        """数値の前後の文脈を抽出"""
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # 文の境界で切る
        if context_start > 0:
            first_period = context.find('。')
            if first_period != -1 and first_period < window:
                context = context[first_period + 1:]
                
        if context_end < len(text):
            last_period = context.rfind('。')
            if last_period != -1 and last_period > len(context) - window:
                context = context[:last_period + 1]
                
        return context.strip()
        
    def _calculate_confidence(self, value_type: str, context: str, 
                            full_text: str) -> float:
        """数値の信頼度を計算"""
        
        confidence = 0.5  # 基本信頼度
        
        # 文脈キーワードのマッチング
        if value_type in self.context_keywords:
            keywords = self.context_keywords[value_type]
            for keyword in keywords:
                if keyword in context:
                    confidence += 0.3
                    break
                    
        # 表や図の中の数値は信頼度が高い
        if any(marker in context for marker in ['表', '図', 'Table', 'Figure']):
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _remove_duplicates(self, values: List[NumericalValue]) -> List[NumericalValue]:
        """重複を除去"""
        
        unique_values = []
        seen = set()
        
        for value in values:
            key = (value.value, value.unit, value.value_type)
            if key not in seen:
                seen.add(key)
                unique_values.append(value)
                
        return unique_values


class UnitConverter:
    """単位変換器"""
    
    def __init__(self):
        """単位変換器の初期化"""
        
        # 単位変換テーブル
        self.conversion_table = {
            # 長さ
            ('km', 'm'): 1000.0,
            ('m', 'km'): 0.001,
            ('m', 'cm'): 100.0,
            ('cm', 'm'): 0.01,
            ('m', 'mm'): 1000.0,
            ('mm', 'm'): 0.001,
            
            # 速度
            ('km/h', 'm/s'): 1/3.6,
            ('m/s', 'km/h'): 3.6,
            
            # 角度
            ('°', 'rad'): math.pi / 180,
            ('rad', '°'): 180 / math.pi,
            ('度', '°'): 1.0,
            ('deg', '°'): 1.0,
            
            # 勾配
            ('%', 'ratio'): 0.01,
            ('ratio', '%'): 100.0,
            
            # 荷重
            ('kN', 'N'): 1000.0,
            ('N', 'kN'): 0.001,
            ('t', 'kN'): 9.80665,  # 重力加速度を考慮
            ('kN', 't'): 1/9.80665,
        }
        
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """単位を変換"""
        
        # 同じ単位の場合
        if from_unit == to_unit:
            return value
            
        # 直接変換
        key = (from_unit, to_unit)
        if key in self.conversion_table:
            return value * self.conversion_table[key]
            
        # 間接変換（共通単位を経由）
        common_units = ['m', 'km/h', '°', '%', 'kN']
        
        for common_unit in common_units:
            key1 = (from_unit, common_unit)
            key2 = (common_unit, to_unit)
            
            if key1 in self.conversion_table and key2 in self.conversion_table:
                intermediate = value * self.conversion_table[key1]
                return intermediate * self.conversion_table[key2]
                
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        
    def normalize_unit(self, unit: str) -> str:
        """単位を正規化"""
        
        unit_mapping = {
            'キロメートル': 'km',
            'メートル': 'm',
            'センチメートル': 'cm',
            'ミリメートル': 'mm',
            'キロメートル毎時': 'km/h',
            'パーセント': '%',
            '度': '°',
            'トン': 't'
        }
        
        return unit_mapping.get(unit, unit)


class FormulaParser:
    """計算式パーサー"""
    
    def __init__(self):
        """計算式パーサーの初期化"""
        
        # 道路設計の標準計算式
        self.standard_formulas = {
            # 曲線半径
            'curve_radius': {
                'expression': 'V**2 / (127 * (e + f))',
                'variables': ['V', 'e', 'f'],
                'description': '最小曲線半径の計算',
                'unit': 'm'
            },
            
            # 制動停止距離
            'stopping_distance': {
                'expression': 'V**2 / (254 * f)',
                'variables': ['V', 'f'],
                'description': '制動停止距離の計算',
                'unit': 'm'
            },
            
            # 視距
            'sight_distance': {
                'expression': 'V * t / 3.6 + V**2 / (254 * f)',
                'variables': ['V', 't', 'f'],
                'description': '停止視距の計算',
                'unit': 'm'
            },
            
            # 縦断曲線長
            'vertical_curve_length': {
                'expression': 'K * A',
                'variables': ['K', 'A'],
                'description': '縦断曲線長の計算',
                'unit': 'm'
            }
        }
        
        # 式パターン
        self.formula_patterns = [
            # R = V²/(127(e+f)) 形式
            r'([A-Z])\s*=\s*([^=]+?)(?=[。、\n]|$)',
            # 「半径は...で計算する」形式
            r'(\w+)は\s*([^。]+?)\s*で計算',
            # 「...により求める」形式
            r'(\w+)を\s*([^。]+?)\s*により求める'
        ]
        
    def extract_formulas(self, text: str) -> List[Formula]:
        """テキストから計算式を抽出"""
        
        formulas = []
        
        # パターンマッチング
        for pattern in self.formula_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                formula_text = match.group(0)
                
                # 標準式との照合
                formula = self._match_standard_formula(formula_text)
                
                if formula:
                    formulas.append(formula)
                else:
                    # カスタム式として解析
                    custom_formula = self._parse_custom_formula(formula_text)
                    if custom_formula:
                        formulas.append(custom_formula)
                        
        return formulas
        
    def _match_standard_formula(self, text: str) -> Optional[Formula]:
        """標準計算式とマッチング"""
        
        text_lower = text.lower()
        
        # キーワードでマッチング
        for formula_type, formula_info in self.standard_formulas.items():
            keywords = {
                'curve_radius': ['曲線半径', '最小半径', 'R ='],
                'stopping_distance': ['制動距離', '停止距離', '制動停止'],
                'sight_distance': ['視距', '停止視距'],
                'vertical_curve_length': ['縦断曲線', '縦断曲線長']
            }
            
            if any(kw in text for kw in keywords.get(formula_type, [])):
                # 変数の値を抽出
                variables = {}
                for var in formula_info['variables']:
                    # V = 60 のようなパターンを探す
                    var_pattern = rf'{var}\s*=\s*(\d+(?:\.\d+)?)'
                    var_match = re.search(var_pattern, text)
                    if var_match:
                        variables[var] = float(var_match.group(1))
                        
                if variables:
                    return Formula(
                        expression=formula_info['expression'],
                        variables=variables,
                        unit=formula_info['unit'],
                        formula_type=formula_type,
                        source=text
                    )
                    
        return None
        
    def _parse_custom_formula(self, text: str) -> Optional[Formula]:
        """カスタム計算式を解析"""
        
        # 簡単な式の解析（例: A = B * C）
        simple_pattern = r'([A-Z])\s*=\s*(.+?)(?=[。、\n]|$)'
        match = re.search(simple_pattern, text)
        
        if match:
            var_name = match.group(1)
            expression = match.group(2).strip()
            
            # 変数を抽出
            variables = {}
            var_pattern = r'([A-Z])\b'
            for var_match in re.finditer(var_pattern, expression):
                var = var_match.group(1)
                if var != var_name:
                    variables[var] = None  # 値は後で設定
                    
            if variables:
                return Formula(
                    expression=expression,
                    variables=variables,
                    source=text
                )
                
        return None


class NumericalProcessor:
    """統合数値処理エンジン"""
    
    def __init__(self):
        """数値処理エンジンの初期化"""
        self.extractor = NumericalExtractor()
        self.converter = UnitConverter()
        self.parser = FormulaParser()
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """テキストを処理して数値情報を抽出"""
        
        # 数値を抽出
        numerical_values = self.extractor.extract_numerical_values(text)
        
        # 計算式を抽出
        formulas = self.parser.extract_formulas(text)
        
        # 数値を分類
        categorized_values = self._categorize_values(numerical_values)
        
        # 計算可能な式を評価
        evaluated_formulas = self._evaluate_formulas(formulas, numerical_values)
        
        result = {
            'numerical_values': [v.to_dict() for v in numerical_values],
            'categorized_values': categorized_values,
            'formulas': evaluated_formulas,
            'summary': self._generate_summary(numerical_values, formulas)
        }
        
        return result
        
    def _categorize_values(self, values: List[NumericalValue]) -> Dict[str, List[Dict[str, Any]]]:
        """数値を種類別に分類"""
        
        categories = {}
        
        for value in values:
            value_type = value.value_type or 'other'
            
            if value_type not in categories:
                categories[value_type] = []
                
            categories[value_type].append(value.to_dict())
            
        return categories
        
    def _evaluate_formulas(self, formulas: List[Formula], 
                          values: List[NumericalValue]) -> List[Dict[str, Any]]:
        """計算式を評価"""
        
        evaluated = []
        
        for formula in formulas:
            # 数値から変数の値を推定
            for var in formula.variables:
                if formula.variables[var] is None:
                    # 文脈から値を探す
                    for value in values:
                        if self._match_variable(var, value):
                            formula.variables[var] = value.value
                            break
                            
            # すべての変数が揃っていれば計算
            if all(v is not None for v in formula.variables.values()):
                try:
                    result = formula.evaluate()
                    evaluated.append({
                        'expression': formula.expression,
                        'variables': formula.variables,
                        'result': result,
                        'unit': formula.unit,
                        'type': formula.formula_type,
                        'source': formula.source
                    })
                except Exception as e:
                    logger.error(f"Formula evaluation failed: {e}")
                    
        return evaluated
        
    def _match_variable(self, var_name: str, value: NumericalValue) -> bool:
        """変数名と数値のマッチング"""
        
        # 変数名のマッピング
        var_mapping = {
            'V': 'speed',
            'R': 'length',  # 半径
            'L': 'length',  # 長さ
            'i': 'gradient',
            'e': 'gradient',  # 片勾配
            'f': 'friction',  # 摩擦係数
        }
        
        expected_type = var_mapping.get(var_name)
        
        return expected_type == value.value_type if expected_type else False
        
    def _generate_summary(self, values: List[NumericalValue], 
                         formulas: List[Formula]) -> Dict[str, Any]:
        """処理結果のサマリーを生成"""
        
        summary = {
            'total_values': len(values),
            'total_formulas': len(formulas),
            'value_types': {},
            'confidence_stats': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        # 値の種類別カウント
        for value in values:
            value_type = value.value_type or 'other'
            summary['value_types'][value_type] = summary['value_types'].get(value_type, 0) + 1
            
            # 信頼度の分類
            if value.confidence >= 0.8:
                summary['confidence_stats']['high'] += 1
            elif value.confidence >= 0.5:
                summary['confidence_stats']['medium'] += 1
            else:
                summary['confidence_stats']['low'] += 1
                
        return summary
        
    def compare_values(self, values1: List[NumericalValue], 
                      values2: List[NumericalValue]) -> Dict[str, Any]:
        """数値の比較"""
        
        comparison = {
            'matched': [],
            'changed': [],
            'added': [],
            'removed': []
        }
        
        # 値をキーでインデックス化
        values1_dict = {(v.value_type, v.unit): v for v in values1}
        values2_dict = {(v.value_type, v.unit): v for v in values2}
        
        # 比較
        for key, value1 in values1_dict.items():
            if key in values2_dict:
                value2 = values2_dict[key]
                if abs(value1.value - value2.value) < 0.001:
                    comparison['matched'].append({
                        'type': value1.value_type,
                        'value': value1.value,
                        'unit': value1.unit
                    })
                else:
                    comparison['changed'].append({
                        'type': value1.value_type,
                        'old_value': value1.value,
                        'new_value': value2.value,
                        'unit': value1.unit,
                        'change': value2.value - value1.value,
                        'change_percent': ((value2.value - value1.value) / value1.value * 100)
                    })
            else:
                comparison['removed'].append(value1.to_dict())
                
        # 追加された値
        for key, value2 in values2_dict.items():
            if key not in values1_dict:
                comparison['added'].append(value2.to_dict())
                
        return comparison


# 便利な関数
def extract_numerical_values(text: str) -> List[NumericalValue]:
    """テキストから数値を抽出（便利関数）"""
    extractor = NumericalExtractor()
    return extractor.extract_numerical_values(text)


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """単位を変換（便利関数）"""
    converter = UnitConverter()
    return converter.convert(value, from_unit, to_unit)


def parse_formulas(text: str) -> List[Formula]:
    """計算式を解析（便利関数）"""
    parser = FormulaParser()
    return parser.extract_formulas(text)