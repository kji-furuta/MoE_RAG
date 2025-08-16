"""
特化機能の簡易テスト（依存関係最小化版）
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_numerical_extraction():
    """数値抽出の簡易テスト"""
    print("=" * 50)
    print("数値抽出テスト")
    print("=" * 50)
    
    from src.rag.specialized.numerical_processor import NumericalExtractor
    
    extractor = NumericalExtractor()
    
    test_text = """
    設計速度は60km/hとし、最小曲線半径は150m、
    縦断勾配は最大5%、車道幅員は3.25mを標準とする。
    """
    
    values = extractor.extract_numerical_values(test_text)
    
    print(f"抽出された数値数: {len(values)}")
    for value in values:
        print(f"- {value.value}{value.unit} ({value.value_type}) 信頼度: {value.confidence:.2f}")
    
    return len(values) > 0


def test_unit_conversion():
    """単位変換の簡易テスト"""
    print("\n" + "=" * 50)
    print("単位変換テスト")
    print("=" * 50)
    
    from src.rag.specialized.numerical_processor import UnitConverter
    
    converter = UnitConverter()
    
    test_cases = [
        (1.5, 'km', 'm', 1500.0),
        (72, 'km/h', 'm/s', 20.0),
        (180, '°', 'rad', 3.14159),
    ]
    
    all_passed = True
    
    for value, from_unit, to_unit, expected in test_cases:
        try:
            result = converter.convert(value, from_unit, to_unit)
            passed = abs(result - expected) < 0.01
            print(f"{value}{from_unit} → {to_unit}: {result:.2f} {'✓' if passed else '✗'}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{value}{from_unit} → {to_unit}: エラー - {e}")
            all_passed = False
    
    return all_passed


def test_design_validation():
    """設計基準検証の簡易テスト"""
    print("\n" + "=" * 50)
    print("設計基準検証テスト")
    print("=" * 50)
    
    from src.rag.specialized.calculation_validator import DesignStandardValidator
    from src.rag.specialized.numerical_processor import NumericalValue
    
    validator = DesignStandardValidator()
    
    # 設計速度のテスト
    speed_value = NumericalValue(
        value=60.0,
        unit='km/h',
        original_text='60km/h',
        value_type='speed'
    )
    
    result = validator.validate_numerical_value(speed_value)
    print(f"設計速度60km/h: {'✓' if result.is_valid else '✗'} - {result.message}")
    
    # 曲線半径のテスト
    radius_value = NumericalValue(
        value=135.0,
        unit='m',
        original_text='135m',
        value_type='length',
        context='曲線半径は135mとする'
    )
    
    result = validator._validate_curve_radius(radius_value, {'design_speed': 60})
    print(f"曲線半径135m: {'✓' if result.is_valid else '✗'} - {result.message}")
    
    return True


def test_version_manager():
    """バージョン管理の簡易テスト"""
    print("\n" + "=" * 50)
    print("バージョン管理テスト")
    print("=" * 50)
    
    from src.rag.specialized.version_manager import VersionManager
    import os
    
    # テスト用データベースファイル
    test_db = "./test_version.db"
    
    try:
        manager = VersionManager(db_path=test_db)
        
        # バージョン作成
        doc_id = "test_doc_001"
        content1 = "バージョン1のコンテンツ"
        
        version1 = manager.create_version(
            document_id=doc_id,
            title="テスト文書 v1.0",
            content=content1
        )
        
        print(f"バージョン1作成: {version1.version_id}")
        
        # バージョン2作成
        content2 = "バージョン2のコンテンツ（更新版）"
        
        version2 = manager.create_version(
            document_id=doc_id,
            title="テスト文書 v1.1", 
            content=content2,
            parent_version_id=version1.version_id
        )
        
        print(f"バージョン2作成: {version2.version_id}")
        
        # バージョン一覧を取得
        versions = manager.db.get_document_versions(doc_id)
        print(f"総バージョン数: {len(versions)}")
        
        return True
        
    except Exception as e:
        print(f"エラー: {e}")
        return False
    finally:
        # テストファイルを削除
        if os.path.exists(test_db):
            os.remove(test_db)


def main():
    """メイン実行関数"""
    print("土木道路設計特化型RAGシステム - 特化機能簡易テスト")
    
    test_results = []
    
    try:
        # 各テストを実行
        test_results.append(test_numerical_extraction())
        test_results.append(test_unit_conversion())
        test_results.append(test_design_validation())
        test_results.append(test_version_manager())
        
        # 結果集計
        passed = sum(test_results)
        total = len(test_results)
        
        print("\n" + "=" * 50)
        print("テスト結果")
        print("=" * 50)
        print(f"成功: {passed}/{total}")
        
        if passed == total:
            print("すべてのテストが成功しました ✓")
        else:
            print(f"{total - passed}個のテストが失敗しました ✗")
            
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()