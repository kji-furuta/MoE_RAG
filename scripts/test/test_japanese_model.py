#!/usr/bin/env python3
"""
日本語モデルのテストスクリプト
"""

import sys
import torch
from src.models.japanese_model import JapaneseModel
from src.utils.gpu_utils import get_gpu_memory_info
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """モデルのロードテスト"""
    print("\n=== モデルロードテスト ===")
    
    # GPU情報を表示
    gpu_info = get_gpu_memory_info()
    if gpu_info["available"]:
        print(f"GPU利用可能: {gpu_info['device_count']}台")
        for device in gpu_info["devices"]:
            print(f"  - {device['name']}: {device['free_memory_gb']:.1f}GB free")
    else:
        print("GPU利用不可 - CPUモードで実行します")
    
    # 小さいモデルでテスト（メモリ節約のため）
    model_name = "stabilityai/japanese-stablelm-3b-4e1t-instruct"
    print(f"\nテストモデル: {model_name}")
    
    # モデル初期化
    model = JapaneseModel(
        model_name=model_name,
        load_in_8bit=True,  # メモリ節約のため8bit量子化
        use_flash_attention=True,
        gradient_checkpointing=True
    )
    
    # フォールバック付きでロード
    success = model.load_with_fallback()
    
    if success:
        print("✓ モデルのロードに成功しました")
        
        # モデル情報を表示
        info = model.get_model_info()
        print(f"\nモデル情報:")
        print(f"  - 総パラメータ数: {info['total_parameters']:,}")
        print(f"  - デバイス: {info['device']}")
        print(f"  - データ型: {info['dtype']}")
        print(f"  - 量子化: 8bit={info['quantization']['8bit']}, 4bit={info['quantization']['4bit']}")
    else:
        print("✗ モデルのロードに失敗しました")
        return False
    
    return model


def test_text_generation(model):
    """テキスト生成テスト"""
    print("\n=== テキスト生成テスト ===")
    
    test_cases = [
        {
            "instruction": "次の文章を要約してください。",
            "input_text": "人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。",
        },
        {
            "instruction": "次の質問に答えてください: 東京の人口は何人ですか？",
            "input_text": None,
        },
        {
            "instruction": "以下の文章を英語に翻訳してください。",
            "input_text": "こんにちは、今日はいい天気ですね。",
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nテスト {i}:")
        print(f"指示: {test_case['instruction']}")
        if test_case['input_text']:
            print(f"入力: {test_case['input_text']}")
        
        try:
            response = model.generate_japanese(
                instruction=test_case['instruction'],
                input_text=test_case['input_text'],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            print(f"応答: {response}")
        except Exception as e:
            print(f"エラー: {e}")


def test_supported_models():
    """サポートモデル一覧の表示"""
    print("\n=== サポートされているモデル一覧 ===")
    
    models = JapaneseModel.list_supported_models()
    
    for model_id, info in models.items():
        print(f"\n{info['display_name']}:")
        print(f"  - モデルID: {model_id}")
        print(f"  - 最小GPU必要メモリ: {info['min_gpu_memory_gb']}GB")
        print(f"  - 推奨データ型: {info['recommended_dtype']}")


def main():
    """メイン関数"""
    print("日本語モデル実装のテスト")
    print("=" * 50)
    
    # サポートモデル一覧を表示
    test_supported_models()
    
    # モデルをロード
    model = test_model_loading()
    
    if model:
        # テキスト生成テスト
        test_text_generation(model)
    
    print("\n\nテスト完了")


if __name__ == "__main__":
    main()