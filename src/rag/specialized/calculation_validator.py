"""
計算式検証システム
道路設計の計算式と数値の妥当性を検証
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
import math
from decimal import Decimal, ROUND_HALF_UP

from .numerical_processor import NumericalValue, Formula, NumericalProcessor
from datetime import datetime


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    value_type: str
    value: float
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    recommended_value: Optional[float] = None
    message: str = ""
    severity: str = "info"  # info, warning, error
    reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'is_valid': self.is_valid,
            'value_type': self.value_type,
            'value': self.value,
            'unit': self.unit,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'recommended_value': self.recommended_value,
            'message': self.message,
            'severity': self.severity,
            'reference': self.reference
        }


@dataclass
class CalculationCheck:
    """計算チェック結果"""
    formula_type: str
    input_values: Dict[str, float]
    calculated_value: float
    expected_value: Optional[float] = None
    is_correct: bool = True
    deviation: Optional[float] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'formula_type': self.formula_type,
            'input_values': self.input_values,
            'calculated_value': self.calculated_value,
            'expected_value': self.expected_value,
            'is_correct': self.is_correct,
            'deviation': self.deviation,
            'message': self.message
        }


class DesignStandardValidator:
    """設計基準バリデーター"""
    
    def __init__(self):
        """設計基準バリデーターの初期化"""
        
        # 道路構造令に基づく設計基準値
        self.design_standards = {
            # 設計速度別の基準値
            'design_speed': {
                'ranges': {
                    '第1種': [60, 80, 100, 120],  # km/h
                    '第2種': [50, 60, 80],
                    '第3種': [40, 50, 60, 80],
                    '第4種': [20, 30, 40, 50, 60]
                }
            },
            
            # 最小曲線半径（設計速度別）
            'minimum_curve_radius': {
                120: {'min': 710, 'recommended': 1000},
                100: {'min': 460, 'recommended': 700},
                80: {'min': 280, 'recommended': 450},
                60: {'min': 150, 'recommended': 250},
                50: {'min': 100, 'recommended': 150},
                40: {'min': 60, 'recommended': 100},
                30: {'min': 30, 'recommended': 50},
                20: {'min': 15, 'recommended': 30}
            },
            
            # 最大縦断勾配（設計速度別）
            'maximum_gradient': {
                120: {'standard': 3, 'special': 5},  # %
                100: {'standard': 4, 'special': 6},
                80: {'standard': 5, 'special': 7},
                60: {'standard': 6, 'special': 8},
                50: {'standard': 7, 'special': 9},
                40: {'standard': 8, 'special': 10},
                30: {'standard': 9, 'special': 11},
                20: {'standard': 10, 'special': 12}
            },
            
            # 車道幅員
            'lane_width': {
                'standard': {
                    '第1種': 3.5,  # m
                    '第2種': 3.25,
                    '第3種': 3.0,
                    '第4種': 2.75
                },
                'minimum': 2.5,
                'maximum': 3.75
            },
            
            # 路肩幅員
            'shoulder_width': {
                'left': {
                    '第1種': {'standard': 1.75, 'minimum': 1.25},
                    '第2種': {'standard': 1.25, 'minimum': 0.75},
                    '第3種': {'standard': 0.75, 'minimum': 0.5},
                    '第4種': {'standard': 0.5, 'minimum': 0.5}
                },
                'right': {
                    '第1種': {'standard': 2.5, 'minimum': 1.75},
                    '第2種': {'standard': 1.75, 'minimum': 1.25},
                    '第3種': {'standard': 0.75, 'minimum': 0.5},
                    '第4種': {'standard': 0.5, 'minimum': 0.5}
                }
            },
            
            # 停止視距（設計速度別）
            'stopping_sight_distance': {
                120: 210,  # m
                100: 160,
                80: 110,
                60: 75,
                50: 55,
                40: 40,
                30: 30,
                20: 20
            }
        }
        
        # 摩擦係数の標準値
        self.friction_coefficients = {
            'dry': 0.8,      # 乾燥時
            'wet': 0.35,     # 湿潤時（設計用）
            'snow': 0.2,     # 積雪時
            'ice': 0.1       # 凍結時
        }
        
    def validate_numerical_value(self, value: NumericalValue, 
                               context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """数値の妥当性を検証"""
        
        # 値の種類に応じた検証
        if value.value_type == 'speed':
            return self._validate_speed(value, context)
        elif value.value_type == 'length':
            return self._validate_length(value, context)
        elif value.value_type == 'gradient':
            return self._validate_gradient(value, context)
        elif value.value_type == 'angle':
            return self._validate_angle(value, context)
        else:
            # 一般的な検証
            return self._validate_general(value)
            
    def _validate_speed(self, value: NumericalValue, 
                       context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """速度の検証"""
        
        speed = value.value
        
        # 設計速度の妥当性チェック
        valid_speeds = []
        for road_class, speeds in self.design_standards['design_speed']['ranges'].items():
            valid_speeds.extend(speeds)
            
        valid_speeds = sorted(set(valid_speeds))
        
        if speed in valid_speeds:
            return ValidationResult(
                is_valid=True,
                value_type='speed',
                value=speed,
                unit=value.unit,
                message=f"設計速度 {speed}km/h は標準値です",
                severity='info',
                reference="道路構造令"
            )
        else:
            # 最も近い標準値を探す
            closest = min(valid_speeds, key=lambda x: abs(x - speed))
            
            return ValidationResult(
                is_valid=False,
                value_type='speed',
                value=speed,
                unit=value.unit,
                recommended_value=closest,
                message=f"設計速度 {speed}km/h は標準値ではありません。最も近い標準値は {closest}km/h です",
                severity='warning',
                reference="道路構造令"
            )
            
    def _validate_length(self, value: NumericalValue,
                        context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """長さ・距離の検証"""
        
        # コンテキストから何の長さかを推定
        if value.context:
            context_lower = value.context.lower()
            
            # 曲線半径の場合
            if '半径' in context_lower or 'radius' in context_lower:
                return self._validate_curve_radius(value, context)
                
            # 車道幅員の場合
            elif '車道' in context_lower and '幅' in context_lower:
                return self._validate_lane_width(value, context)
                
            # 視距の場合
            elif '視距' in context_lower:
                return self._validate_sight_distance(value, context)
                
        # 一般的な長さの検証
        return ValidationResult(
            is_valid=True,
            value_type='length',
            value=value.value,
            unit=value.unit,
            message="長さの値です",
            severity='info'
        )
        
    def _validate_curve_radius(self, value: NumericalValue,
                              context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """曲線半径の検証"""
        
        radius = value.value
        design_speed = context.get('design_speed', 60) if context else 60
        
        if design_speed in self.design_standards['minimum_curve_radius']:
            standards = self.design_standards['minimum_curve_radius'][design_speed]
            min_radius = standards['min']
            recommended = standards['recommended']
            
            if radius >= recommended:
                severity = 'info'
                message = f"曲線半径 {radius}m は推奨値以上です"
            elif radius >= min_radius:
                severity = 'warning'
                message = f"曲線半径 {radius}m は最小値以上ですが、推奨値 {recommended}m 未満です"
            else:
                severity = 'error'
                message = f"曲線半径 {radius}m は最小値 {min_radius}m 未満です"
                
            return ValidationResult(
                is_valid=radius >= min_radius,
                value_type='curve_radius',
                value=radius,
                unit=value.unit,
                min_value=min_radius,
                recommended_value=recommended,
                message=message,
                severity=severity,
                reference=f"道路構造令（設計速度{design_speed}km/h）"
            )
            
        return ValidationResult(
            is_valid=True,
            value_type='curve_radius',
            value=radius,
            unit=value.unit,
            message="曲線半径の値です",
            severity='info'
        )
        
    def _validate_gradient(self, value: NumericalValue,
                          context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """勾配の検証"""
        
        gradient = value.value
        design_speed = context.get('design_speed', 60) if context else 60
        
        if design_speed in self.design_standards['maximum_gradient']:
            standards = self.design_standards['maximum_gradient'][design_speed]
            standard_max = standards['standard']
            special_max = standards['special']
            
            if gradient <= standard_max:
                severity = 'info'
                message = f"縦断勾配 {gradient}% は標準値以下です"
            elif gradient <= special_max:
                severity = 'warning'
                message = f"縦断勾配 {gradient}% は特別値以下ですが、標準値 {standard_max}% を超えています"
            else:
                severity = 'error'
                message = f"縦断勾配 {gradient}% は特別値 {special_max}% を超えています"
                
            return ValidationResult(
                is_valid=gradient <= special_max,
                value_type='gradient',
                value=gradient,
                unit=value.unit,
                max_value=special_max,
                recommended_value=standard_max,
                message=message,
                severity=severity,
                reference=f"道路構造令（設計速度{design_speed}km/h）"
            )
            
        return ValidationResult(
            is_valid=True,
            value_type='gradient',
            value=gradient,
            unit=value.unit,
            message="勾配の値です",
            severity='info'
        )
        
    def _validate_lane_width(self, value: NumericalValue,
                           context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """車道幅員の検証"""
        
        width = value.value
        road_class = context.get('road_class', '第3種') if context else '第3種'
        
        standards = self.design_standards['lane_width']
        if road_class in standards['standard']:
            standard_width = standards['standard'][road_class]
            min_width = standards['minimum']
            max_width = standards['maximum']
            
            if min_width <= width <= max_width:
                if abs(width - standard_width) < 0.1:
                    severity = 'info'
                    message = f"車道幅員 {width}m は{road_class}の標準値です"
                else:
                    severity = 'warning'
                    message = f"車道幅員 {width}m は許容範囲内ですが、{road_class}の標準値 {standard_width}m と異なります"
            else:
                severity = 'error'
                message = f"車道幅員 {width}m は許容範囲（{min_width}m～{max_width}m）外です"
                
            return ValidationResult(
                is_valid=min_width <= width <= max_width,
                value_type='lane_width',
                value=width,
                unit=value.unit,
                min_value=min_width,
                max_value=max_width,
                recommended_value=standard_width,
                message=message,
                severity=severity,
                reference=f"道路構造令（{road_class}）"
            )
            
        return ValidationResult(
            is_valid=True,
            value_type='lane_width',
            value=width,
            unit=value.unit,
            message="車道幅員の値です",
            severity='info'
        )
        
    def _validate_sight_distance(self, value: NumericalValue,
                               context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """視距の検証"""
        
        distance = value.value
        design_speed = context.get('design_speed', 60) if context else 60
        
        if design_speed in self.design_standards['stopping_sight_distance']:
            required_distance = self.design_standards['stopping_sight_distance'][design_speed]
            
            if distance >= required_distance:
                severity = 'info'
                message = f"停止視距 {distance}m は必要視距 {required_distance}m 以上です"
            else:
                severity = 'error'
                message = f"停止視距 {distance}m は必要視距 {required_distance}m 未満です"
                
            return ValidationResult(
                is_valid=distance >= required_distance,
                value_type='sight_distance',
                value=distance,
                unit=value.unit,
                min_value=required_distance,
                message=message,
                severity=severity,
                reference=f"道路構造令（設計速度{design_speed}km/h）"
            )
            
        return ValidationResult(
            is_valid=True,
            value_type='sight_distance',
            value=distance,
            unit=value.unit,
            message="視距の値です",
            severity='info'
        )
        
    def _validate_angle(self, value: NumericalValue,
                       context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """角度の検証"""
        
        angle = value.value
        
        # 基本的な範囲チェック
        if 0 <= angle <= 180:
            return ValidationResult(
                is_valid=True,
                value_type='angle',
                value=angle,
                unit=value.unit,
                message=f"角度 {angle}度は有効な範囲内です",
                severity='info'
            )
        else:
            return ValidationResult(
                is_valid=False,
                value_type='angle',
                value=angle,
                unit=value.unit,
                message=f"角度 {angle}度は無効な値です（0～180度の範囲外）",
                severity='error'
            )
            
    def _validate_general(self, value: NumericalValue) -> ValidationResult:
        """一般的な数値の検証"""
        
        # 負の値チェック
        if value.value < 0 and value.value_type not in ['gradient']:
            return ValidationResult(
                is_valid=False,
                value_type=value.value_type or 'general',
                value=value.value,
                unit=value.unit,
                message="負の値は通常無効です",
                severity='warning'
            )
            
        return ValidationResult(
            is_valid=True,
            value_type=value.value_type or 'general',
            value=value.value,
            unit=value.unit,
            message="数値の値です",
            severity='info'
        )


class FormulaValidator:
    """計算式バリデーター"""
    
    def __init__(self):
        """計算式バリデーターの初期化"""
        
        self.standard_validator = DesignStandardValidator()
        
        # 標準計算式
        self.standard_formulas = {
            # 最小曲線半径
            'minimum_curve_radius': {
                'formula': lambda V, e, f: V**2 / (127 * (e + f)),
                'variables': {
                    'V': '設計速度 (km/h)',
                    'e': '片勾配 (比)',
                    'f': '横すべり摩擦係数'
                },
                'unit': 'm'
            },
            
            # 制動停止距離
            'braking_distance': {
                'formula': lambda V, f: V**2 / (254 * f),
                'variables': {
                    'V': '設計速度 (km/h)',
                    'f': '縦すべり摩擦係数'
                },
                'unit': 'm'
            },
            
            # 停止視距
            'stopping_sight_distance': {
                'formula': lambda V, t, f: V * t / 3.6 + V**2 / (254 * f),
                'variables': {
                    'V': '設計速度 (km/h)',
                    't': '反応時間 (s)',
                    'f': '縦すべり摩擦係数'
                },
                'unit': 'm'
            },
            
            # 縦断曲線長
            'vertical_curve_length': {
                'formula': lambda K, A: K * A,
                'variables': {
                    'K': '縦断曲線変化率',
                    'A': '代数差 (%)'
                },
                'unit': 'm'
            }
        }
        
    def validate_calculation(self, formula: Formula, 
                           expected_result: Optional[float] = None) -> CalculationCheck:
        """計算式の検証"""
        
        # 計算式タイプを特定
        formula_type = self._identify_formula_type(formula)
        
        if formula_type and formula_type in self.standard_formulas:
            # 標準式で再計算
            standard_formula = self.standard_formulas[formula_type]
            
            # 変数マッピング
            mapped_vars = self._map_variables(formula.variables, formula_type)
            
            if mapped_vars:
                try:
                    # 標準式で計算
                    calculated = standard_formula['formula'](**mapped_vars)
                    
                    # 元の計算結果と比較
                    if formula.result:
                        deviation = abs(calculated - formula.result) / calculated * 100
                        is_correct = deviation < 5  # 5%以内の誤差は許容
                        
                        message = f"計算結果の偏差: {deviation:.2f}%"
                        if not is_correct:
                            message += " (許容範囲を超えています)"
                    else:
                        is_correct = True
                        message = f"計算結果: {calculated:.2f} {standard_formula['unit']}"
                        
                    return CalculationCheck(
                        formula_type=formula_type,
                        input_values=mapped_vars,
                        calculated_value=calculated,
                        expected_value=expected_result or formula.result,
                        is_correct=is_correct,
                        deviation=deviation if formula.result else None,
                        message=message
                    )
                    
                except Exception as e:
                    logger.error(f"Calculation validation failed: {e}")
                    
        # 標準式でない場合の一般的な検証
        return CalculationCheck(
            formula_type=formula_type or 'custom',
            input_values=formula.variables,
            calculated_value=formula.result or 0,
            expected_value=expected_result,
            is_correct=True,
            message="カスタム計算式"
        )
        
    def _identify_formula_type(self, formula: Formula) -> Optional[str]:
        """計算式のタイプを特定"""
        
        if formula.formula_type:
            return formula.formula_type
            
        # 式のパターンから推定
        expr = formula.expression.lower()
        
        if 'v**2' in expr and '127' in expr:
            return 'minimum_curve_radius'
        elif 'v**2' in expr and '254' in expr and 't' in formula.variables:
            return 'stopping_sight_distance'
        elif 'v**2' in expr and '254' in expr:
            return 'braking_distance'
        elif 'k' in formula.variables and 'a' in formula.variables:
            return 'vertical_curve_length'
            
        return None
        
    def _map_variables(self, variables: Dict[str, float], 
                      formula_type: str) -> Optional[Dict[str, float]]:
        """変数をマッピング"""
        
        if formula_type not in self.standard_formulas:
            return None
            
        standard_vars = self.standard_formulas[formula_type]['variables']
        mapped = {}
        
        for std_var in standard_vars:
            # 変数名の照合（大文字小文字を無視）
            for var, value in variables.items():
                if var.upper() == std_var.upper():
                    mapped[std_var] = value
                    break
                    
        # すべての変数が揃っているかチェック
        if len(mapped) == len(standard_vars):
            return mapped
            
        return None
        
    def check_formula_consistency(self, formulas: List[Formula]) -> List[Dict[str, Any]]:
        """複数の計算式の整合性をチェック"""
        
        consistency_checks = []
        
        # 同じ変数を使用する式を探す
        for i in range(len(formulas)):
            for j in range(i + 1, len(formulas)):
                formula1 = formulas[i]
                formula2 = formulas[j]
                
                # 共通変数を探す
                common_vars = set(formula1.variables.keys()) & set(formula2.variables.keys())
                
                if common_vars:
                    # 値の一致をチェック
                    inconsistencies = []
                    
                    for var in common_vars:
                        val1 = formula1.variables[var]
                        val2 = formula2.variables[var]
                        
                        if val1 and val2 and abs(val1 - val2) > 0.001:
                            inconsistencies.append({
                                'variable': var,
                                'value1': val1,
                                'value2': val2,
                                'difference': abs(val1 - val2)
                            })
                            
                    if inconsistencies:
                        consistency_checks.append({
                            'formula1': formula1.expression,
                            'formula2': formula2.expression,
                            'inconsistencies': inconsistencies,
                            'message': f"{len(inconsistencies)}個の変数値の不一致があります"
                        })
                        
        return consistency_checks


class CalculationValidator:
    """統合計算検証システム"""
    
    def __init__(self):
        """計算検証システムの初期化"""
        self.standard_validator = DesignStandardValidator()
        self.formula_validator = FormulaValidator()
        self.numerical_processor = NumericalProcessor()
        
    def validate_document(self, text: str, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """文書全体の数値・計算を検証"""
        
        # 数値と計算式を抽出
        processed = self.numerical_processor.process_text(text)
        
        # 数値の検証
        value_validations = []
        for value_dict in processed['numerical_values']:
            value = NumericalValue(**value_dict)
            validation = self.standard_validator.validate_numerical_value(value, context)
            value_validations.append(validation.to_dict())
            
        # 計算式の検証
        formula_validations = []
        formulas = []
        
        for formula_dict in processed['formulas']:
            formula = Formula(
                expression=formula_dict['expression'],
                variables=formula_dict['variables'],
                result=formula_dict.get('result'),
                unit=formula_dict.get('unit'),
                formula_type=formula_dict.get('type')
            )
            formulas.append(formula)
            
            check = self.formula_validator.validate_calculation(formula)
            formula_validations.append(check.to_dict())
            
        # 整合性チェック
        consistency_checks = self.formula_validator.check_formula_consistency(formulas)
        
        # サマリー生成
        summary = self._generate_validation_summary(
            value_validations, formula_validations, consistency_checks
        )
        
        return {
            'numerical_validations': value_validations,
            'formula_validations': formula_validations,
            'consistency_checks': consistency_checks,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
        
    def _generate_validation_summary(self, 
                                   value_validations: List[Dict[str, Any]],
                                   formula_validations: List[Dict[str, Any]],
                                   consistency_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """検証結果のサマリーを生成"""
        
        # エラー・警告の集計
        errors = 0
        warnings = 0
        
        for validation in value_validations:
            if validation['severity'] == 'error':
                errors += 1
            elif validation['severity'] == 'warning':
                warnings += 1
                
        for validation in formula_validations:
            if not validation['is_correct']:
                errors += 1
                
        if consistency_checks:
            warnings += len(consistency_checks)
            
        return {
            'total_values': len(value_validations),
            'total_formulas': len(formula_validations),
            'errors': errors,
            'warnings': warnings,
            'is_valid': errors == 0,
            'message': self._generate_summary_message(errors, warnings)
        }
        
    def _generate_summary_message(self, errors: int, warnings: int) -> str:
        """サマリーメッセージを生成"""
        
        if errors == 0 and warnings == 0:
            return "すべての数値と計算式が設計基準に適合しています"
        elif errors == 0:
            return f"{warnings}件の警告があります。内容を確認してください"
        else:
            return f"{errors}件のエラーと{warnings}件の警告があります。修正が必要です"


# 便利な関数
def validate_numerical_values(text: str, 
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """テキスト内の数値を検証（便利関数）"""
    validator = CalculationValidator()
    return validator.validate_document(text, context)


def check_design_standard(value: float, value_type: str, 
                         unit: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """設計基準との適合性をチェック（便利関数）"""
    validator = DesignStandardValidator()
    numerical_value = NumericalValue(
        value=value,
        unit=unit,
        original_text=f"{value}{unit}",
        value_type=value_type
    )
    return validator.validate_numerical_value(numerical_value, context)