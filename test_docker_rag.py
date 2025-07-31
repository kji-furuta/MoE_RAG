#!/usr/bin/env python3
"""
Docker RAG統合テスト
Docker環境でのRAGシステム依存関係とWebインターフェース統合をテスト
"""

import sys
import importlib
from pathlib import Path


def test_rag_dependencies():
    """RAG依存関係のインポートテスト"""
    print("=== Docker RAG依存関係テスト ===")
    
    # 必須RAG依存関係のリスト
    rag_dependencies = [
        ("loguru", "ログ管理"),
        ("qdrant_client", "ベクトルデータベース"),
        ("sentence_transformers", "埋め込みモデル"),
        ("fitz", "PDF処理 (PyMuPDF)"),
        ("pdfplumber", "PDF解析"),
        ("spacy", "自然言語処理"),
        ("yaml", "設定ファイル読み込み"),
        ("rich", "コンソール出力"),
        ("tqdm", "プログレスバー"),
        ("sklearn", "機械学習"),
        ("nltk", "自然言語処理"),
        ("llama_index", "LlamaIndex"),
        ("langchain", "LangChain"),
        ("easyocr", "OCR"),
    ]
    
    print("📦 RAG依存関係インポートチェック:")
    
    success_count = 0
    total_count = len(rag_dependencies)
    
    for module_name, description in rag_dependencies:
        try:
            # モジュールのインポートテスト
            importlib.import_module(module_name)
            print(f"  ✅ {module_name} - {description}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module_name} - {description} (エラー: {e})")
        except Exception as e:
            print(f"  ⚠️ {module_name} - {description} (警告: {e})")
    
    success_rate = (success_count / total_count) * 100
    print(f"\n📊 依存関係成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    return success_count == total_count


def test_spacy_japanese_model():
    """spaCy日本語モデルのテスト"""
    print("\n=== spaCy日本語モデルテスト ===")
    
    try:
        import spacy
        print("✅ spacy基本ライブラリインポート成功")
        
        # 日本語モデルの読み込みテスト
        try:
            nlp = spacy.load("ja_core_news_lg")
            print("✅ ja_core_news_lg モデル読み込み成功")
            
            # 簡単な解析テスト
            doc = nlp("道路設計の技術基準について説明します。")
            tokens = [token.text for token in doc]
            print(f"✅ トークン化テスト成功: {tokens[:5]}...")
            
            return True
            
        except OSError as e:
            print(f"❌ 日本語モデル読み込み失敗: {e}")
            print("💡 解決策: python -m spacy download ja_core_news_lg")
            return False
            
    except ImportError as e:
        print(f"❌ spacy インポート失敗: {e}")
        return False


def test_pytorch_integration():
    """PyTorchとRAGライブラリの連携テスト"""
    print("\n=== PyTorch-RAG統合テスト ===")
    
    try:
        import torch
        print(f"✅ PyTorch バージョン: {torch.__version__}")
        print(f"✅ CUDA利用可能: {torch.cuda.is_available()}")
        
        # sentence-transformersとPyTorchの連携テスト
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ sentence_transformers インポート成功")
            
            # GPU利用可能性チェック
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"✅ 推奨デバイス: {device}")
            
            return True
            
        except ImportError as e:
            print(f"❌ sentence_transformers インポート失敗: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch インポート失敗: {e}")
        return False


def test_rag_config_loading():
    """RAG設定ファイル読み込みテスト"""
    print("\n=== RAG設定読み込みテスト ===")
    
    config_path = Path("/workspace/src/rag/config/rag_config.yaml")
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return False
    
    try:
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ YAML設定ファイル読み込み成功")
        
        # 重要な設定項目の確認
        required_sections = ["system", "llm", "embedding", "vector_store"]
        
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section} セクション存在")
            else:
                print(f"  ❌ {section} セクション不足")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return False


def test_unified_webapp_imports():
    """統合Webアプリのインポートテスト"""
    print("\n=== 統合Webアプリインポートテスト ===")
    
    try:
        # FastAPIとRAG統合のテスト
        from fastapi import FastAPI
        print("✅ FastAPI インポート成功")
        
        #統合アプリケーションのインポートテスト
        sys.path.insert(0, "/workspace")
        
        try:
            # main_unified.pyが正常にインポートできるかテスト
            import app.main_unified
            print("✅ main_unified.py インポート成功")
            
            # RAG関連のクラスが定義されているかチェック
            if hasattr(app.main_unified, 'RAGApplication'):
                print("✅ RAGApplication クラス定義確認")
            else:
                print("❌ RAGApplication クラス未定義")
                return False
            
            if hasattr(app.main_unified, 'rag_app'):
                print("✅ rag_app インスタンス確認")
            else:
                print("❌ rag_app インスタンス未定義")
                return False
                
            return True
            
        except ImportError as e:
            print(f"❌ main_unified.py インポート失敗: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ FastAPI インポート失敗: {e}")
        return False


def test_directory_structure():
    """ディレクトリ構造テスト"""
    print("\n=== ディレクトリ構造テスト ===")
    
    required_dirs = [
        "/workspace/src/rag",
        "/workspace/temp_uploads",
        "/workspace/qdrant_data", 
        "/workspace/outputs/rag_index",
        "/workspace/data/rag_documents",
        "/workspace/config",
        "/workspace/app",
        "/workspace/logs"
    ]
    
    print("📁 必要ディレクトリ存在確認:")
    
    success_count = 0
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
            success_count += 1
        else:
            print(f"  ❌ {dir_path} - 存在しません")
    
    success_rate = (success_count / len(required_dirs)) * 100
    print(f"\n📊 ディレクトリ成功率: {success_count}/{len(required_dirs)} ({success_rate:.1f}%)")
    
    return success_count == len(required_dirs)


def main():
    """メインテスト実行"""
    print("🐳 Docker RAG統合テスト")
    print("=" * 60)
    print("Docker環境でのRAGシステム統合状況を確認します\n")
    
    tests = [
        ("RAG依存関係テスト", test_rag_dependencies),
        ("spaCy日本語モデルテスト", test_spacy_japanese_model),
        ("PyTorch-RAG統合テスト", test_pytorch_integration),
        ("RAG設定読み込みテスト", test_rag_config_loading),
        ("統合Webアプリインポートテスト", test_unified_webapp_imports),
        ("ディレクトリ構造テスト", test_directory_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}でエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("🏁 Docker RAGテスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n📊 総合成功率: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 Docker RAG統合が完全に成功！")
        print("💡 統合Webインターフェースがポート8050で利用可能")
        print("🚀 RAG機能とファインチューニング機能が統合環境で動作")
    else:
        print(f"\n⚠️ {total - passed}個のテストが失敗")
        print("🔧 Docker環境の修正が必要です")
        
        # 失敗時の推奨アクション
        if passed < total * 0.5:  # 50%未満の成功率
            print("\n🛠️ 推奨アクション:")
            print("1. docker build --no-cache でイメージを再ビルド")
            print("2. requirements_rag.txt の依存関係を確認")
            print("3. python -m spacy download ja_core_news_lg を実行")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)