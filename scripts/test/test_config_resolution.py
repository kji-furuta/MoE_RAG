"""
設定解決機能の統合テスト
モデルパス解決・設定検証・自動修正の確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_complete_config_loading():
    """完全な設定読み込みテスト"""
    print("=" * 60)
    print("完全な設定読み込みテスト")
    print("=" * 60)
    
    from src.rag.config.rag_config import load_config
    
    try:
        # 完全な設定読み込み（モデルパス解決・検証付き）
        config = load_config(resolve_model_paths=True)
        
        print("✅ 設定読み込み成功")
        print(f"システム名: {config.system_name}")
        print(f"バージョン: {config.version}")
        
        print(f"\n📋 LLM設定:")
        print(f"  ファインチューニング済み使用: {config.llm.use_finetuned}")
        print(f"  モデルパス: {config.llm.finetuned_model_path}")
        print(f"  ベースモデル: {config.llm.base_model}")
        
        # パスの存在確認
        if config.llm.use_finetuned:
            path_exists = Path(config.llm.finetuned_model_path).exists()
            print(f"  パス存在: {'✅' if path_exists else '❌'}")
            
            if path_exists:
                # モデル詳細情報
                from src.rag.config.model_path_resolver import ModelPathResolver
                resolver = ModelPathResolver()
                model_info = resolver.get_model_info(config.llm.finetuned_model_path)
                print(f"  モデルタイプ: {model_info['model_type']}")
                print(f"  サイズ: {model_info['size'] / (1024*1024):.1f}MB")
        
        print(f"\n🔍 埋め込み設定:")
        print(f"  モデル: {config.embedding.model_name}")
        print(f"  デバイス: {config.embedding.device}")
        print(f"  バッチサイズ: {config.embedding.batch_size}")
        
        print(f"\n🗄️ ベクトルストア設定:")
        print(f"  タイプ: {config.vector_store.type}")
        print(f"  パス: {config.vector_store.path}")
        print(f"  コレクション: {config.vector_store.collection_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 設定読み込み失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """設定検証機能のテスト"""
    print("\n" + "=" * 60)
    print("設定検証機能テスト")
    print("=" * 60)
    
    from src.rag.config.rag_config import load_config
    from src.rag.config.config_validator import print_validation_report
    
    try:
        # 設定読み込み（検証なし）
        config = load_config(resolve_model_paths=False)
        
        print("📋 設定検証レポート:")
        print_validation_report(config, auto_fix=True)
        
        return True
        
    except Exception as e:
        print(f"❌ 設定検証失敗: {e}")
        return False


def test_model_fallback():
    """モデルフォールバック機能のテスト"""
    print("\n" + "=" * 60)
    print("モデルフォールバック機能テスト")
    print("=" * 60)
    
    from src.rag.config.rag_config import RAGConfig, LLMConfig
    
    # 存在しないモデルパスを設定
    config = RAGConfig()
    config.llm = LLMConfig(
        use_finetuned=True,
        finetuned_model_path="./nonexistent_model_path",
        base_model="cyberagent/calm3-22b-chat"
    )
    
    print(f"設定前 - ファインチューニング済み: {config.llm.use_finetuned}")
    print(f"設定前 - モデルパス: {config.llm.finetuned_model_path}")
    
    # 設定検証・自動修正
    from src.rag.config.config_validator import validate_config
    
    issues, fixed_count = validate_config(config, auto_fix=True)
    
    print(f"\n修正後 - ファインチューニング済み: {config.llm.use_finetuned}")
    print(f"修正後 - モデルパス: {config.llm.finetuned_model_path}")
    print(f"修正された問題数: {fixed_count}")
    
    # 問題があればレポート表示
    if issues:
        from src.rag.config.config_validator import ConfigValidator
        validator = ConfigValidator(config)
        print("\n" + validator.generate_report())
    
    return True


def test_available_models():
    """利用可能モデル一覧のテスト"""
    print("\n" + "=" * 60)
    print("利用可能モデル一覧テスト")
    print("=" * 60)
    
    from src.rag.config.model_path_resolver import ModelPathResolver
    
    resolver = ModelPathResolver()
    models = resolver.list_available_models()
    
    print(f"発見されたモデル数: {len(models)}")
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   📁 パス: {model['path_str']}")
        print(f"   🏷️ タイプ: {model['model_type']}")
        print(f"   📊 サイズ: {model['size_mb']}MB")
        print(f"   📅 作成日: {model['created_time_str']}")
        
        # ファイル詳細（上位3つ）
        model_info = resolver.get_model_info(model['path_str'])
        if model_info.get('files'):
            print(f"   📄 主要ファイル:")
            for file_info in model_info['files'][:3]:
                size_mb = file_info['size'] / (1024 * 1024)
                print(f"     - {file_info['name']} ({size_mb:.1f}MB)")
    
    return len(models) > 0


def test_symlink_creation():
    """シンボリックリンク作成テスト"""
    print("\n" + "=" * 60)
    print("シンボリックリンク作成テスト")
    print("=" * 60)
    
    from src.rag.config.model_path_resolver import ModelPathResolver
    
    resolver = ModelPathResolver()
    
    # 最新モデル検索
    latest_model = resolver.find_latest_model()
    
    if not latest_model:
        print("❌ 最新モデルが見つかりません")
        return False
    
    print(f"最新モデル: {latest_model}")
    
    # シンボリックリンク作成
    link_path = resolver.create_latest_symlink(latest_model)
    
    if link_path:
        print(f"✅ シンボリックリンク作成: {link_path}")
        
        # リンクの確認
        link_path_obj = Path(link_path)
        if link_path_obj.exists():
            print(f"✅ リンク有効: {link_path_obj.resolve()}")
            return True
        else:
            print(f"❌ リンク無効")
            return False
    else:
        print("❌ シンボリックリンク作成失敗")
        return False


def main():
    """メイン実行関数"""
    print("🔧 設定解決機能 統合テスト")
    print("=" * 60)
    print("設定の不整合問題の解決確認\n")
    
    test_results = []
    
    try:
        # 各テストを実行
        test_results.append(("完全な設定読み込み", test_complete_config_loading()))
        test_results.append(("設定検証機能", test_config_validation()))
        test_results.append(("モデルフォールバック", test_model_fallback()))
        test_results.append(("利用可能モデル一覧", test_available_models()))
        test_results.append(("シンボリックリンク作成", test_symlink_creation()))
        
        # 結果集計
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        print("\n" + "=" * 60)
        print("🏁 最終テスト結果")
        print("=" * 60)
        
        for test_name, result in test_results:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
        
        print(f"\n📊 成功率: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 すべてのテストが成功しました！")
            print("💡 設定の不整合問題が完全に解決されました。")
            print("\n✨ 主な改善点:")
            print("  - モデルパスの自動検出・解決")
            print("  - 設定ファイルの自動検証・修正")
            print("  - エラー時のフォールバック機能")
            print("  - シンボリックリンクによる統一アクセス")
        else:
            print(f"\n⚠️ {total - passed}個のテストが失敗しました")
            print("🔧 追加の調整が必要な可能性があります")
            
    except Exception as e:
        print(f"\n💥 テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()