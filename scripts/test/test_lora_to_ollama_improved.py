#!/usr/bin/env python3
"""
改善版LoRA to Ollamaのテストスクリプト
"""

import sys
import os
import json
import time
from pathlib import Path

# パスを追加
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/scripts')

from apply_lora_to_gguf_improved import ImprovedLoRAToOllamaConverter

def test_improved_converter():
    """改善版コンバーターのテスト"""
    print("=" * 60)
    print("改善版LoRA to Ollamaコンバーターのテスト")
    print("=" * 60)
    
    # テスト1: 一時ディレクトリの使用確認
    print("\n1. 一時ディレクトリモードのテスト")
    print("-" * 40)
    
    # ローカル環境用のパスを使用
    workspace_dir = "/home/kjifu/MoE_RAG"
    if not Path(workspace_dir).exists():
        workspace_dir = "/workspace"
        
    converter = ImprovedLoRAToOllamaConverter(
        workspace_dir=workspace_dir,
        use_temp_dir=True
    )
    
    try:
        # llama.cppのセットアップ
        llama_cpp_dir = converter.setup_llama_cpp()
        print(f"✅ llama.cppセットアップ完了: {llama_cpp_dir}")
        
        # 一時ディレクトリの確認
        if str(llama_cpp_dir).startswith("/tmp"):
            print("✅ 一時ディレクトリを使用しています")
        else:
            print("⚠️ 一時ディレクトリを使用していません")
            
        # クリーンアップテスト
        converter.cleanup()
        if converter.temp_dir and not converter.temp_dir.exists():
            print("✅ クリーンアップ成功")
        else:
            print("⚠️ クリーンアップの確認が必要")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        
    # テスト2: LoRAアダプター検索
    print("\n2. LoRAアダプター検索テスト")
    print("-" * 40)
    
    converter2 = ImprovedLoRAToOllamaConverter(workspace_dir=workspace_dir)
    
    try:
        lora_dir, metadata = converter2.find_lora_adapter()
        print(f"✅ LoRAアダプター発見: {lora_dir}")
        print(f"   メタデータ: {json.dumps(metadata, indent=2)[:200]}...")
    except FileNotFoundError:
        print("⚠️ LoRAアダプターが見つかりません（これは正常な場合があります）")
    except Exception as e:
        print(f"❌ エラー: {e}")
        
    # テスト3: ベースモデル検索
    print("\n3. ベースモデル検索テスト")
    print("-" * 40)
    
    test_metadata = {
        "base_model_name_or_path": f"{workspace_dir}/models/deepseek-base"
    }
    
    base_model_path = converter2.find_base_model_path(test_metadata)
    if base_model_path:
        print(f"✅ ベースモデル発見: {base_model_path}")
    else:
        print("⚠️ ベースモデルが見つかりません（これは正常な場合があります）")
        
    # クリーンアップ
    converter2.cleanup()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
    
    # 実際の変換テスト（オプション）
    if len(sys.argv) > 1 and sys.argv[1] == "--run-full":
        print("\n完全な変換テストを実行中...")
        converter3 = ImprovedLoRAToOllamaConverter(
            workspace_dir=workspace_dir,
            use_temp_dir=True
        )
        
        result = converter3.run(
            output_model_name="test-model",
            skip_ollama=True  # Ollama登録をスキップ
        )
        
        print(f"\n変換結果:")
        print(f"成功: {result['success']}")
        print(f"メッセージ: {result['message']}")
        print(f"完了ステップ: {result['steps_completed']}")
        
        # ファイルが作成されたか確認
        if result['model_path']:
            model_path = Path(result['model_path'])
            if model_path.exists():
                size_gb = model_path.stat().st_size / 1024 / 1024 / 1024
                print(f"✅ モデルファイル作成成功: {size_gb:.2f} GB")
            else:
                print("❌ モデルファイルが見つかりません")

if __name__ == "__main__":
    test_improved_converter()