"""
特化機能のテストスクリプト
数値処理とバージョン管理の動作確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.specialized import (
    NumericalProcessor, VersionManager, CalculationValidator,
    extract_numerical_values, check_design_standard
)
from datetime import datetime


def test_numerical_processing():
    """数値処理機能のテスト"""
    print("=" * 60)
    print("数値処理機能のテスト")
    print("=" * 60)
    
    # テストテキスト
    test_text = """
    【道路設計基準】
    
    1. 設計条件
    設計速度：60km/h
    道路区分：第3種第2級
    
    2. 幾何構造
    最小曲線半径は、R = V²/(127(e+f)) により計算する。
    ここで、V=60km/h、e=0.06（片勾配）、f=0.15（横すべり摩擦係数）とすると、
    R = 60²/(127×(0.06+0.15)) = 134.7m ≈ 135m となる。
    
    よって、最小曲線半径は150mを採用する。
    
    3. 横断構成
    - 車道幅員：3.25m × 2車線 = 6.5m
    - 路肩幅員：左側0.75m、右側1.75m
    - 横断勾配：2.0%（標準部）
    
    4. 縦断線形
    - 最大縦断勾配：6%（特別な場合8%）
    - 縦断曲線長：L = K × A（Kは変化率、Aは代数差）
    """
    
    # 数値処理エンジンを初期化
    processor = NumericalProcessor()
    
    # テキストを処理
    result = processor.process_text(test_text)
    
    # 結果を表示
    print("\n【抽出された数値】")
    for value in result['numerical_values']:
        print(f"- {value['value']}{value['unit']} ({value['value_type']}) "
              f"信頼度: {value['confidence']:.2f}")
    
    print("\n【カテゴリ別集計】")
    for category, values in result['categorized_values'].items():
        print(f"- {category}: {len(values)}個")
    
    print("\n【抽出された計算式】")
    for formula in result['formulas']:
        print(f"- 式: {formula['expression']}")
        print(f"  変数: {formula['variables']}")
        if 'result' in formula and formula['result']:
            print(f"  計算結果: {formula['result']:.2f}")
    
    print("\n【サマリー】")
    summary = result['summary']
    print(f"- 総数値数: {summary['total_values']}")
    print(f"- 総計算式数: {summary['total_formulas']}")
    print(f"- 信頼度分布: 高={summary['confidence_stats']['high']}, "
          f"中={summary['confidence_stats']['medium']}, "
          f"低={summary['confidence_stats']['low']}")


def test_design_standard_validation():
    """設計基準検証機能のテスト"""
    print("\n" + "=" * 60)
    print("設計基準検証機能のテスト")
    print("=" * 60)
    
    # 検証するデータ
    test_cases = [
        {'value': 60, 'type': 'speed', 'unit': 'km/h', 'context': {'road_class': '第3種'}},
        {'value': 135, 'type': 'curve_radius', 'unit': 'm', 'context': {'design_speed': 60}},
        {'value': 3.25, 'type': 'lane_width', 'unit': 'm', 'context': {'road_class': '第3種'}},
        {'value': 8, 'type': 'gradient', 'unit': '%', 'context': {'design_speed': 60}},
        {'value': 12, 'type': 'gradient', 'unit': '%', 'context': {'design_speed': 60}},
    ]
    
    print("\n【設計基準との適合性チェック】")
    for test in test_cases:
        result = check_design_standard(
            test['value'], test['type'], test['unit'], test['context']
        )
        
        print(f"\n{test['type']} = {test['value']}{test['unit']}")
        print(f"  妥当性: {'○' if result.is_valid else '×'}")
        print(f"  {result.message}")
        if result.reference:
            print(f"  参照: {result.reference}")


def test_version_management():
    """バージョン管理機能のテスト"""
    print("\n" + "=" * 60)
    print("バージョン管理機能のテスト")
    print("=" * 60)
    
    # バージョン管理システムを初期化
    manager = VersionManager(db_path="./test_version.db")
    
    # テスト文書のバージョンを作成
    doc_id = "road_design_standard_001"
    
    # バージョン1を作成
    content_v1 = """
    道路設計基準書
    
    第1章 総則
    1.1 目的
    本基準は、道路の設計に関する技術的基準を定める。
    
    第2章 設計速度
    2.1 設計速度は60km/hを標準とする。
    """
    
    version1 = manager.create_version(
        document_id=doc_id,
        title="道路設計基準書 v1.0",
        content=content_v1,
        metadata={'author': 'システム管理者', 'category': '設計基準'}
    )
    
    print(f"バージョン1作成: {version1.version_id}")
    
    # バージョン2を作成（内容を更新）
    content_v2 = """
    道路設計基準書
    
    第1章 総則
    1.1 目的
    本基準は、道路の設計に関する技術的基準を定める。
    
    第2章 設計速度
    2.1 設計速度は原則として60km/hを標準とする。
    2.2 地形条件により、80km/hまで引き上げることができる。
    
    第3章 幾何構造
    3.1 最小曲線半径は設計速度に応じて決定する。
    """
    
    version2 = manager.create_version(
        document_id=doc_id,
        title="道路設計基準書 v1.1",
        content=content_v2,
        parent_version_id=version1.version_id,
        metadata={'author': 'システム管理者', 'category': '設計基準'}
    )
    
    print(f"バージョン2作成: {version2.version_id}")
    
    # バージョン間の差分を検出
    diff = manager.compare_versions(
        version1.version_id, version2.version_id,
        content_v1, content_v2
    )
    
    print("\n【バージョン間の差分】")
    print(f"追加されたセクション: {len(diff.added_sections)}個")
    for section in diff.added_sections[:3]:
        print(f"  - {section}")
    
    print(f"変更されたセクション: {len(diff.modified_sections)}個")
    print(f"類似度スコア: {diff.similarity_score:.2%}")
    
    # バージョンツリーを表示
    tree = manager.get_version_tree(doc_id)
    print("\n【バージョンツリー】")
    print(f"文書ID: {tree['document_id']}")
    print(f"総バージョン数: {len(tree['versions'])}")
    for version_id, version_info in tree['versions'].items():
        print(f"  - {version_info['version_number']}: {version_info['title']}")


def test_calculation_validation():
    """計算検証機能のテスト"""
    print("\n" + "=" * 60)
    print("計算検証機能のテスト")
    print("=" * 60)
    
    test_text = """
    設計速度V=80km/hの場合の計算例：
    
    1. 最小曲線半径
    R = V²/(127(e+f))
    ここで、e=0.06、f=0.14とすると、
    R = 80²/(127×(0.06+0.14)) = 251.97m
    
    2. 制動停止距離
    D = V²/(254f)
    f=0.30として、
    D = 80²/(254×0.30) = 83.99m
    
    3. 停止視距
    S = V×t/3.6 + V²/(254×f)
    反応時間t=2.5秒、f=0.30として、
    S = 80×2.5/3.6 + 80²/(254×0.30) = 139.54m
    """
    
    # 計算検証器を初期化
    validator = CalculationValidator()
    
    # 文書を検証
    validation_result = validator.validate_document(
        test_text,
        context={'design_speed': 80, 'road_class': '第1種'}
    )
    
    print("\n【数値検証結果】")
    for val in validation_result['numerical_validations'][:5]:
        severity_mark = '○' if val['severity'] == 'info' else ('△' if val['severity'] == 'warning' else '×')
        print(f"{severity_mark} {val['value']}{val['unit']} - {val['message']}")
    
    print("\n【計算式検証結果】")
    for calc in validation_result['formula_validations']:
        check_mark = '○' if calc['is_correct'] else '×'
        print(f"{check_mark} {calc['formula_type']}: 計算値={calc['calculated_value']:.2f}")
        if calc['deviation'] is not None:
            print(f"   偏差: {calc['deviation']:.2f}%")
    
    print("\n【検証サマリー】")
    summary = validation_result['summary']
    print(f"総数値数: {summary['total_values']}")
    print(f"総計算式数: {summary['total_formulas']}")
    print(f"エラー数: {summary['errors']}")
    print(f"警告数: {summary['warnings']}")
    print(f"検証結果: {summary['message']}")


def main():
    """メイン実行関数"""
    print("土木道路設計特化型RAGシステム - 特化機能テスト")
    print("=" * 60)
    
    try:
        # 各機能をテスト
        test_numerical_processing()
        test_design_standard_validation()
        test_version_management()
        test_calculation_validation()
        
        print("\n" + "=" * 60)
        print("すべてのテストが完了しました")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()