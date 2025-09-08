#!/usr/bin/env python3
"""
GPT-NeoX-20B専用: LoRAアダプターをGGUF形式のベースモデルに完全マージする実装
GGUFファイル操作を含む完全な実装
"""

import os
import sys
import json
import struct
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, BinaryIO
import logging
import numpy as np

# 必要なライブラリをインポート
try:
    import torch
    import safetensors.torch
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドでインストールしてください:")
    print("pip install torch safetensors gguf numpy")
    sys.exit(1)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPTNeoXLoRAMerger:
    """GPT-NeoX-20BのLoRAアダプタをGGUFモデルに完全マージ"""
    
    # GPT-NeoX特有のレイヤーマッピング
    NEOX_TO_GGUF_MAPPING = {
        # Attention layers
        "attention.query_key_value": ["attn_q", "attn_k", "attn_v"],
        "attention.dense": ["attn_output"],
        # MLP layers  
        "mlp.dense_h_to_4h": ["ffn_up"],
        "mlp.dense_4h_to_h": ["ffn_down"]
    }
    
    # GGUFのテンソル名パターン
    GGUF_TENSOR_PATTERNS = {
        "attn_q": "blk.{layer}.attn_q.weight",
        "attn_k": "blk.{layer}.attn_k.weight",
        "attn_v": "blk.{layer}.attn_v.weight",
        "attn_output": "blk.{layer}.attn_output.weight",
        "ffn_up": "blk.{layer}.ffn_up.weight",
        "ffn_down": "blk.{layer}.ffn_down.weight"
    }
    
    def __init__(self, lora_adapter_path: str, base_model_path: str, output_path: str,
                 lora_scale: float = 1.0):
        """
        Args:
            lora_adapter_path: LoRAアダプタのディレクトリパス
            base_model_path: GGUFベースモデルのパス
            output_path: 出力GGUFモデルのパス
            lora_scale: LoRAスケーリング係数（デフォルト: 1.0）
        """
        self.lora_adapter_path = Path(lora_adapter_path)
        self.base_model_path = Path(base_model_path)
        self.output_path = Path(output_path)
        self.lora_scale = lora_scale
        
        # LoRAアダプタを読み込み
        self.adapter_config = self._load_adapter_config()
        self.lora_weights = self._load_lora_weights()
        
        # LoRAパラメータを取得
        self.lora_r = self.adapter_config.get("r", 16)
        self.lora_alpha = self.adapter_config.get("lora_alpha", 32)
        self.scaling = self.lora_alpha / self.lora_r * self.lora_scale
        
        logger.info(f"LoRA設定: r={self.lora_r}, alpha={self.lora_alpha}, scaling={self.scaling:.4f}")
    
    def _load_adapter_config(self) -> Dict[str, Any]:
        """LoRAアダプタの設定を読み込み"""
        config_path = self.lora_adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"adapter_config.jsonが見つかりません: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # GPT-NeoXモデルかチェック
        base_model = config.get("base_model_name_or_path", "")
        if "gpt-neox" not in base_model.lower():
            logger.warning(f"警告: ベースモデルがGPT-NeoXではない可能性があります: {base_model}")
            
        return config
    
    def _load_lora_weights(self) -> Dict[str, torch.Tensor]:
        """LoRAウェイトを読み込み"""
        weights_path = self.lora_adapter_path / "adapter_model.safetensors"
        if weights_path.exists():
            logger.info(f"Safetensorsファイルを読み込み中: {weights_path}")
            weights = safetensors.torch.load_file(str(weights_path))
        else:
            weights_path = self.lora_adapter_path / "adapter_model.bin"
            if weights_path.exists():
                logger.info(f"PyTorchファイルを読み込み中: {weights_path}")
                weights = torch.load(str(weights_path), map_location="cpu")
            else:
                raise FileNotFoundError("LoRAウェイトファイルが見つかりません")
        
        # float32に変換
        for key in weights:
            weights[key] = weights[key].float()
        
        return weights
    
    def _split_qkv_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        query_key_valueの統合ウェイトをQ, K, Vに3分割
        """
        # 最初の次元を3等分
        split_size = weight.shape[0] // 3
        q_weight = weight[:split_size]
        k_weight = weight[split_size:split_size*2]
        v_weight = weight[split_size*2:]
        
        return q_weight, k_weight, v_weight
    
    def _parse_lora_key(self, lora_key: str) -> Optional[Dict[str, Any]]:
        """
        LoRAキーを解析してレイヤー情報を抽出
        例: "base_model.model.gpt_neox.layers.0.attention.query_key_value.lora_A.weight"
        """
        parts = lora_key.split(".")
        
        # レイヤー番号を探す
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    continue
        
        if layer_idx is None:
            return None
        
        # モジュール名を特定
        module_name = None
        for neox_name in self.NEOX_TO_GGUF_MAPPING.keys():
            if neox_name.replace(".", "_") in lora_key or neox_name in lora_key:
                module_name = neox_name
                break
        
        if module_name is None:
            return None
        
        # LoRA A/Bを判定
        is_lora_a = "lora_A" in lora_key or "lora_a" in lora_key
        is_lora_b = "lora_B" in lora_key or "lora_b" in lora_key
        
        if not is_lora_a and not is_lora_b:
            return None
        
        return {
            "layer_idx": layer_idx,
            "module_name": module_name,
            "is_lora_a": is_lora_a,
            "is_lora_b": is_lora_b
        }
    
    def _compute_lora_delta(self, lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
        """
        LoRAのデルタウェイトを計算: ΔW = B @ A * scaling
        """
        # LoRA: ΔW = B @ A * (alpha / r)
        delta = torch.matmul(lora_b, lora_a) * self.scaling
        return delta
    
    def merge_to_gguf(self):
        """GGUFファイルにLoRAウェイトをマージ"""
        logger.info("=" * 60)
        logger.info("GPT-NeoX-20B LoRA → GGUF 完全マージ処理")
        logger.info("=" * 60)
        
        # GGUFファイルを読み込み
        logger.info(f"ベースモデルを読み込み中: {self.base_model_path}")
        reader = GGUFReader(str(self.base_model_path))
        
        # メタデータとテンソルを取得
        metadata = {}
        tensors = {}
        
        # メタデータを読み込み
        for key, field in reader.fields.items():
            if hasattr(field, 'value'):
                metadata[key] = field.value
            elif hasattr(field, 'data'):
                metadata[key] = field.data
            else:
                metadata[key] = field
        
        # テンソルを読み込み
        logger.info("ベースモデルのテンソルを読み込み中...")
        for tensor in reader.tensors:
            tensor_name = tensor.name
            
            # テンソルのデータタイプを判定
            tensor_type = tensor.tensor_type
            
            # データタイプに応じてnumpy配列を作成
            if tensor_type == GGMLQuantizationType.F32:
                dtype = np.float32
                element_size = 4
            elif tensor_type == GGMLQuantizationType.F16:
                dtype = np.float16
                element_size = 2
            elif tensor_type in [GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K_M, 
                                GGMLQuantizationType.Q4_K_S]:
                # 量子化されたデータはそのまま保持
                tensors[tensor_name] = {
                    "data": tensor.data,
                    "shape": tensor.shape,
                    "dtype": tensor_type,
                    "quantized": True
                }
                continue
            else:
                # その他の量子化タイプもそのまま保持
                tensors[tensor_name] = {
                    "data": tensor.data,
                    "shape": tensor.shape,
                    "dtype": tensor_type,
                    "quantized": True
                }
                continue
            
            # float32/float16の場合は変換
            expected_size = np.prod(tensor.shape) * element_size
            actual_size = len(tensor.data)
            
            if actual_size == expected_size:
                tensor_array = np.frombuffer(tensor.data, dtype=dtype).reshape(tensor.shape)
            else:
                # サイズが合わない場合はそのまま保持
                tensor_array = tensor.data
            
            tensors[tensor_name] = {
                "data": tensor_array,
                "shape": tensor.shape,
                "dtype": tensor_type,
                "quantized": False
            }
        
        logger.info(f"読み込み完了: {len(tensors)}個のテンソル")
        
        # LoRAデルタを計算してマージ
        logger.info("LoRAウェイトをマージ中...")
        lora_pairs = {}  # (layer, module) -> {lora_a, lora_b}
        
        # LoRA A/Bのペアを収集
        for lora_key, lora_weight in self.lora_weights.items():
            parsed = self._parse_lora_key(lora_key)
            if not parsed:
                continue
            
            key = (parsed["layer_idx"], parsed["module_name"])
            if key not in lora_pairs:
                lora_pairs[key] = {}
            
            if parsed["is_lora_a"]:
                lora_pairs[key]["lora_a"] = lora_weight
            elif parsed["is_lora_b"]:
                lora_pairs[key]["lora_b"] = lora_weight
        
        # 各LoRAペアをマージ
        merged_count = 0
        for (layer_idx, module_name), pair in lora_pairs.items():
            if "lora_a" not in pair or "lora_b" not in pair:
                logger.warning(f"不完全なLoRAペア: layer={layer_idx}, module={module_name}")
                continue
            
            lora_a = pair["lora_a"]
            lora_b = pair["lora_b"]
            
            # デルタを計算
            delta = self._compute_lora_delta(lora_a, lora_b)
            
            # query_key_valueの場合は分割
            if module_name == "attention.query_key_value":
                q_delta, k_delta, v_delta = self._split_qkv_weights(delta)
                deltas = [q_delta, k_delta, v_delta]
                gguf_names = ["attn_q", "attn_k", "attn_v"]
            else:
                deltas = [delta]
                gguf_names = self.NEOX_TO_GGUF_MAPPING[module_name]
            
            # GGUFテンソルにマージ
            for delta_tensor, gguf_name in zip(deltas, gguf_names):
                tensor_name = self.GGUF_TENSOR_PATTERNS[gguf_name].format(layer=layer_idx)
                
                if tensor_name in tensors:
                    tensor_info = tensors[tensor_name]
                    
                    # 量子化されたテンソルはスキップ（警告を出す）
                    if tensor_info.get("quantized", False):
                        logger.warning(f"量子化されたテンソルはマージできません: {tensor_name}")
                        logger.warning(f"  完全なマージにはfloat32モデルが必要です")
                        continue
                    
                    # 元のテンソルにデルタを加算
                    original = tensor_info["data"]
                    
                    # numpy配列でない場合はスキップ
                    if not isinstance(original, np.ndarray):
                        logger.warning(f"テンソルデータが配列ではありません: {tensor_name}")
                        continue
                    
                    # 形状を確認
                    delta_np = delta_tensor.numpy()
                    if original.shape != delta_np.shape:
                        logger.warning(f"形状不一致: {tensor_name}")
                        logger.warning(f"  Original: {original.shape}, Delta: {delta_np.shape}")
                        
                        # 転置を試みる
                        if original.shape == delta_np.T.shape:
                            delta_np = delta_np.T
                            logger.info(f"  転置して形状を一致させました")
                        else:
                            continue
                    
                    # マージ
                    tensors[tensor_name]["data"] = original + delta_np
                    merged_count += 1
                    logger.debug(f"マージ完了: {tensor_name}")
                else:
                    logger.warning(f"テンソルが見つかりません: {tensor_name}")
        
        logger.info(f"マージ完了: {merged_count}個のテンソル")
        
        # 新しいGGUFファイルを作成
        logger.info(f"マージ済みモデルを保存中: {self.output_path}")
        
        # GGUFWriterを使用して保存
        writer = GGUFWriter(str(self.output_path), "gptneox")
        
        # メタデータを書き込み
        for key, value in metadata.items():
            if key not in ["general.file_type", "general.quantization_version"]:
                # 型に応じて適切なメソッドを使用
                if isinstance(value, str):
                    writer.add_string(key, value)
                elif isinstance(value, int):
                    writer.add_uint32(key, value)
                elif isinstance(value, float):
                    writer.add_float32(key, value)
                elif isinstance(value, bool):
                    writer.add_bool(key, value)
                elif isinstance(value, list):
                    if value and isinstance(value[0], str):
                        writer.add_array(key, value)
        
        # テンソルを書き込み
        for tensor_name, tensor_info in tensors.items():
            tensor_data = tensor_info["data"]
            
            # float32に変換
            if tensor_data.dtype != np.float32:
                tensor_data = tensor_data.astype(np.float32)
            
            # テンソルを追加
            writer.add_tensor(tensor_name, tensor_data)
        
        # ヘッダーを書き込み
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        
        logger.info(f"✅ マージ完了: {self.output_path}")
        
        # ファイルサイズを確認
        file_size = self.output_path.stat().st_size / (1024 ** 3)  # GB
        logger.info(f"出力ファイルサイズ: {file_size:.2f} GB")
        
        return True


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPT-NeoX-20B用LoRAアダプタをGGUFモデルに完全マージ"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="LoRAアダプタのディレクトリパス"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="GGUFベースモデルのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力GGUFモデルのパス"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRAスケーリング係数（デフォルト: 1.0）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを表示"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # マージ実行
        merger = GPTNeoXLoRAMerger(
            lora_adapter_path=args.lora_path,
            base_model_path=args.base_model,
            output_path=args.output,
            lora_scale=args.lora_scale
        )
        
        success = merger.merge_to_gguf()
        
        if success:
            logger.info("\n🎉 GPT-NeoX-20B LoRAマージが完了しました！")
            
            # Ollamaへの登録方法を表示
            logger.info("\nOllamaへの登録方法:")
            logger.info(f"1. Modelfileを作成:")
            logger.info(f"   FROM {args.output}")
            logger.info(f"   SYSTEM \"あなたは日本の土木設計の専門家です。\"")
            logger.info(f"")
            logger.info(f"2. モデルを作成:")
            logger.info(f"   ollama create gpt-neox-20b-merged -f Modelfile")
        else:
            logger.error("マージに失敗しました")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()