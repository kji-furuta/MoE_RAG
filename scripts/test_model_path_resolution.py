"""
モデルパス解決機能のテストスクリプト
設定の不整合問題の解決確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_path_resolver():
    """モデルパス解決器のテスト"""
    print("=" * 60)
    print("モデルパス解決器テスト")
    print("=" * 60)
    
    from src.rag.config.model_path_resolver import ModelPathResolver
    
    resolver = ModelPathResolver("./outputs")
    
    # 利用可能なモデル一覧を取得
    models = resolver.list_available_models()
    
    print(f"\n利用可能なモデル数: {len(models)}")
    for i, model in enumerate(models[:5], 1):  # 上位5個まで表示
        print(f"{i}. {model['name']}")
        print(f"   タイプ: {model['model_type']}")
        print(f"   サイズ: {model['size_mb']}MB")
        print(f"   作成日時: {model['created_time_str']}")
        print(f"   パス: {model['path_str']}")
        print()
    
    # 最新モデルを検索
    latest_model = resolver.find_latest_model()
    
    if latest_model:
        print(f"最新モデル: {latest_model}")
        
        # モデル情報を取得
        model_info = resolver.get_model_info(latest_model)
        print(f"モデル詳細:")
        print(f"  存在: {model_info['exists']}")
        print(f"  タイプ: {model_info['model_type']}")
        print(f"  サイズ: {model_info['size'] / (1024*1024):.1f}MB")
        print(f"  ファイル数: {len(model_info['files'])}")
        
        # 上位5ファイルを表示
        for file_info in model_info['files'][:5]:
            size_mb = file_info['size'] / (1024 * 1024)
            print(f"    - {file_info['name']} ({size_mb:.1f}MB)")
    else:
        print("最新モデルが見つかりませんでした")
    
    return latest_model is not None


def test_model_path_validation():
    """モデルパス検証のテスト"""
    print("\n" + "=" * 60)
    print("モデルパス検証テスト")
    print("=" * 60)
    
    from src.rag.config.model_path_resolver import ModelPathResolver
    
    resolver = ModelPathResolver("./outputs")
    
    # テストケース
    test_paths = [
        "./outputs/latest",  # 存在しないパス
        "./outputs/lora_20250725_061715",  # 実際に存在するパス（予想）
        "./nonexistent_path",  # 存在しないパス
        "./outputs",  # ディレクトリだがモデルではない
    ]
    
    all_passed = True
    
    for path in test_paths:
        result = resolver.validate_model_path(path)
        
        status = "✓" if result['is_valid'] else "✗"
        print(f"{status} {path}")
        print(f"  存在: {result['exists']}")
        print(f"  ディレクトリ: {result['is_directory']}")
        print(f"  モデルファイル有: {result['has_model_files']}")
        print(f"  モデルタイプ: {result['model_type']}")
        
        if result['issues']:
            print(f"  問題: {', '.join(result['issues'])}")
        print()
        
        # 実際に存在するパスで検証が失敗した場合は問題
        if Path(path).exists() and not result['is_valid']:
            all_passed = False
    
    return all_passed


def test_config_loading_with_resolution():
    """設定読み込み（モデルパス解決付き）のテスト"""
    print("\n" + "=" * 60)
    print("設定読み込み（モデルパス解決付き）テスト")
    print("=" * 60)
    
    from src.rag.config.rag_config import load_config
    
    try:
        # モデルパス解決を有効にして設定を読み込み
        config = load_config(resolve_model_paths=True)
        
        print("設定読み込み成功 ✓")
        print(f"システム名: {config.system_name}")
        print(f"バージョン: {config.version}")
        print(f"言語: {config.language}")
        
        print(f"\nLLM設定:")
        print(f"  ファインチューニング済み使用: {config.llm.use_finetuned}")
        print(f"  ファインチューニング済みパス: {config.llm.finetuned_model_path}")
        print(f"  ベースモデル: {config.llm.base_model}")
        print(f"  デバイスマップ: {config.llm.device_map}")
        
        # パスの存在確認
        if config.llm.use_finetuned:
            path_exists = Path(config.llm.finetuned_model_path).exists()
            print(f"  ファインチューニング済みパス存在: {'✓' if path_exists else '✗'}")
        
        print(f"\n埋め込み設定:")
        print(f"  モデル名: {config.embedding.model_name}")
        print(f"  デバイス: {config.embedding.device}")
        print(f"  バッチサイズ: {config.embedding.batch_size}")
        
        print(f"\nベクトルストア設定:")
        print(f"  タイプ: {config.vector_store.type}")
        print(f"  パス: {config.vector_store.path}")
        print(f"  コレクション名: {config.vector_store.collection_name}")
        
        return True
        
    except Exception as e:
        print(f"設定読み込み失敗 ✗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_behavior():
    """フォールバック動作のテスト"""
    print("\n" + "=" * 60)
    print("フォールバック動作テスト")
    print("=" * 60)
    
    from src.rag.config.rag_config import load_config
    
    # 存在しない設定ファイルを指定
    try:
        config = load_config("nonexistent_config.yaml")
        
        print("デフォルト設定読み込み成功 ✓")
        print(f"システム名: {config.system_name}")
        print(f"LLMベースモデル: {config.llm.base_model}")
        
        # デフォルト値が正しく設定されているかチェック
        expected_defaults = {
            'system_name': '土木道路設計特化型RAGシステム',
            'version': '1.0.0',
            'language': 'ja'
        }
        
        all_correct = True
        for key, expected in expected_defaults.items():
            actual = getattr(config, key)
            if actual != expected:
                print(f"  デフォルト値不正: {key} = {actual} (期待値: {expected}) ✗")
                all_correct = False
            else:
                print(f"  {key}: {actual} ✓")
        
        return all_correct
        
    except Exception as e:
        print(f"フォールバック失敗 ✗: {e}")
        return False


def test_symlink_creation():
    """シンボリックリンク作成のテスト"""
    print("\n" + "=" * 60)
    print("シンボリックリンク作成テスト")
    print("=" * 60)
    
    from src.rag.config.model_path_resolver import ModelPathResolver
    
    resolver = ModelPathResolver("./outputs")
    
    # 最新モデルのシンボリックリンクを作成
    latest_link = resolver.create_latest_symlink()
    
    if latest_link:
        print(f"シンボリックリンク作成成功: {latest_link} ✓")
        
        # リンクの存在確認
        link_path = Path(latest_link)
        if link_path.exists():
            print(f"リンク先: {link_path.resolve()} ✓")
            return True
        else:
            print(f"リンクが無効です ✗")
            return False
    else:
        print("シンボリックリンク作成失敗 ✗")
        return False


def main():
    """メイン実行関数"""
    print("モデルパス解決機能テスト")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 各テストを実行
        test_results.append(("モデルパス解決器", test_model_path_resolver()))
        test_results.append(("モデルパス検証", test_model_path_validation()))
        test_results.append(("設定読み込み", test_config_loading_with_resolution()))
        test_results.append(("フォールバック動作", test_fallback_behavior()))
        test_results.append(("シンボリックリンク作成", test_symlink_creation()))
        
        # 結果集計
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print("\n" + "=" * 60)
        print("テスト結果")
        print("=" * 60)
        
        for test_name, result in test_results:
            status = "✓" if result else "✗"
            print(f"{status} {test_name}")
        
        print(f"\n成功: {passed}/{total}")
        
        if passed == total:
            print("すべてのテストが成功しました ✓")
            print("\n🎉 設定の不整合問題が解決されました！")
        else:
            print(f"{total - passed}個のテストが失敗しました ✗")
            
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()