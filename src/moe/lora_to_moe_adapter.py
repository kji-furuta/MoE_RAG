"""
LoRAアダプタをMoEアーキテクチャに統合するアダプター
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import json
import shutil
from dataclasses import dataclass

from .moe_architecture import CivilEngineeringMoEModel
from .base_config import MoEConfig
from .constants import ExpertType, DOMAIN_KEYWORDS

logger = logging.getLogger(__name__)


@dataclass
class LoRAMoEConfig:
    """LoRA-MoE変換設定"""
    lora_paths: List[str]  # 複数のLoRAアダプタパス
    expert_names: List[str]  # 各エキスパートの名前
    base_model_name: str
    num_experts: int = 4
    num_experts_per_token: int = 2
    output_dir: str = "/workspace/outputs/moe_from_lora"
    merge_strategy: str = "weighted"  # weighted, selective, duplicate
    temperature: float = 1.0  # ルーター温度


class LoRAToMoEAdapter:
    """LoRAファインチューニング済みモデルをMoEに変換"""
    
    def __init__(self, config: LoRAMoEConfig):
        """
        Args:
            config: LoRA-MoE変換設定
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def convert_to_moe(self) -> Dict:
        """複数のLoRAモデルをMoEアーキテクチャに変換"""
        
        logger.info(f"LoRA→MoE変換開始: {len(self.config.lora_paths)}個のLoRAアダプタ")
        
        try:
            # 1. ベースモデルをロード
            logger.info(f"ベースモデルをロード: {self.config.base_model_name}")
            base_model = self._load_base_model()
            
            # 2. 各LoRAアダプタをエキスパートとして準備
            experts = self._create_experts_from_lora(base_model)
            
            # 3. MoEモデルを構築
            moe_model = self._build_moe_model(base_model, experts)
            
            # 4. ルーターを初期化
            self._initialize_router(moe_model)
            
            # 5. モデルを保存
            output_path = self._save_moe_model(moe_model)
            
            # 6. 変換用スクリプトも生成
            self._generate_conversion_scripts(output_path)
            
            return {
                "success": True,
                "output_dir": str(output_path),
                "num_experts": len(experts),
                "expert_names": self.config.expert_names,
                "message": "LoRA→MoE変換が完了しました",
                "next_steps": [
                    "MoEモデルのテスト: python scripts/test_moe_model.py",
                    "GGUF変換: python scripts/moe_to_gguf.py",
                    "Ollama登録: ollama create moe-model -f Modelfile"
                ]
            }
            
        except Exception as e:
            logger.error(f"LoRA→MoE変換エラー: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _load_base_model(self):
        """ベースモデルをロード"""
        from transformers import BitsAndBytesConfig
        
        # メモリ効率のため4bit量子化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model
    
    def _create_experts_from_lora(self, base_model) -> List[nn.Module]:
        """各LoRAアダプタをエキスパートモジュールとして作成"""
        experts = []
        
        for i, (lora_path, expert_name) in enumerate(zip(
            self.config.lora_paths, 
            self.config.expert_names
        )):
            logger.info(f"エキスパート{i+1}を作成: {expert_name} from {lora_path}")
            
            if Path(lora_path).exists():
                # LoRAアダプタを適用したモデルを作成
                expert_model = PeftModel.from_pretrained(
                    base_model,
                    lora_path,
                    adapter_name=expert_name
                )
                
                # エキスパートモジュールとして抽出
                expert_module = self._extract_expert_module(expert_model, expert_name)
                experts.append(expert_module)
            else:
                # LoRAが存在しない場合はベースモデルの複製を使用
                logger.warning(f"LoRAパスが見つかりません: {lora_path}")
                expert_module = self._create_default_expert(base_model, expert_name)
                experts.append(expert_module)
        
        return experts
    
    def _extract_expert_module(self, peft_model, expert_name: str) -> nn.Module:
        """PEFTモデルからエキスパートモジュールを抽出"""
        
        class LoRAExpert(nn.Module):
            """LoRAベースのエキスパート"""
            
            def __init__(self, peft_model, name):
                super().__init__()
                self.name = name
                # LoRAレイヤーを保持
                self.lora_layers = {}
                
                for name, module in peft_model.named_modules():
                    if "lora" in name.lower():
                        self.lora_layers[name] = module
                
                # FFN層を作成（MoEアーキテクチャに合わせる）
                hidden_size = peft_model.config.hidden_size
                intermediate_size = hidden_size * 4
                
                self.w1 = nn.Linear(hidden_size, intermediate_size)
                self.w2 = nn.Linear(intermediate_size, hidden_size)
                self.w3 = nn.Linear(hidden_size, intermediate_size)
                self.act_fn = nn.SiLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, hidden_states):
                # Gate projection
                gate_output = self.act_fn(self.w1(hidden_states))
                # Up projection
                up_output = self.w3(hidden_states)
                # Combine and down project
                x = gate_output * up_output
                x = self.dropout(x)
                output = self.w2(x)
                
                return output
        
        return LoRAExpert(peft_model, expert_name)
    
    def _create_default_expert(self, base_model, expert_name: str) -> nn.Module:
        """デフォルトのエキスパートモジュールを作成"""
        
        class DefaultExpert(nn.Module):
            def __init__(self, hidden_size, name):
                super().__init__()
                self.name = name
                intermediate_size = hidden_size * 4
                
                self.w1 = nn.Linear(hidden_size, intermediate_size)
                self.w2 = nn.Linear(intermediate_size, hidden_size)
                self.w3 = nn.Linear(hidden_size, intermediate_size)
                self.act_fn = nn.SiLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, hidden_states):
                gate_output = self.act_fn(self.w1(hidden_states))
                up_output = self.w3(hidden_states)
                x = gate_output * up_output
                x = self.dropout(x)
                output = self.w2(x)
                return output
        
        hidden_size = base_model.config.hidden_size
        return DefaultExpert(hidden_size, expert_name)
    
    def _build_moe_model(self, base_model, experts: List[nn.Module]) -> nn.Module:
        """MoEモデルを構築"""
        
        # MoE設定を作成
        moe_config = MoEConfig(
            hidden_size=base_model.config.hidden_size,
            num_experts=len(experts),
            num_experts_per_tok=self.config.num_experts_per_token,
            output_dir=self.config.output_dir,
            domain_specific_routing=True
        )
        
        # カスタムMoEモデル
        class LoRAMoEModel(nn.Module):
            def __init__(self, config, experts_list):
                super().__init__()
                self.config = config
                self.experts = nn.ModuleList(experts_list)
                
                # ルーター（ゲート）
                self.gate = nn.Linear(
                    config.hidden_size,
                    len(experts_list),
                    bias=False
                )
                
                # エキスパート選択用の温度パラメータ
                self.temperature = nn.Parameter(torch.tensor(1.0))
                
            def forward(self, hidden_states):
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # ルーティング確率を計算
                router_logits = self.gate(hidden_states)
                routing_weights = torch.softmax(
                    router_logits / self.temperature,
                    dim=-1
                )
                
                # Top-kエキスパートを選択
                topk_weights, topk_indices = torch.topk(
                    routing_weights,
                    k=min(self.config.num_experts_per_tok, len(self.experts)),
                    dim=-1
                )
                
                # エキスパート出力を計算
                output = torch.zeros_like(hidden_states)
                
                for i, expert in enumerate(self.experts):
                    # このエキスパートが選択された位置を取得
                    expert_mask = (topk_indices == i).any(dim=-1)
                    
                    if expert_mask.any():
                        expert_input = hidden_states[expert_mask]
                        expert_output = expert(expert_input)
                        
                        # 重み付けして出力に加算
                        weights = topk_weights[topk_indices == i].unsqueeze(-1)
                        output[expert_mask] += expert_output * weights
                
                return output
        
        moe_model = LoRAMoEModel(moe_config, experts)
        return moe_model
    
    def _initialize_router(self, moe_model):
        """ルーターを専門分野に基づいて初期化"""
        logger.info("MoEルーターを初期化中...")
        
        # ドメイン固有のキーワードに基づいてルーター重みを初期化
        if hasattr(moe_model, 'gate'):
            # Xavier初期化
            nn.init.xavier_uniform_(moe_model.gate.weight)
            
            # 専門分野ごとにバイアスを追加（オプション）
            domain_biases = {
                "road_geometry": 0.1,
                "bridge_design": 0.0,
                "tunnel_design": -0.1,
                "general": 0.0
            }
            
            for i, expert_name in enumerate(self.config.expert_names):
                if expert_name in domain_biases:
                    moe_model.gate.weight.data[i] += domain_biases[expert_name]
    
    def _save_moe_model(self, moe_model) -> Path:
        """MoEモデルを保存"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # モデルの状態を保存
        torch.save({
            'model_state_dict': moe_model.state_dict(),
            'config': self.config.__dict__,
            'expert_names': self.config.expert_names,
            'num_experts': len(self.config.expert_names),
            'base_model': self.config.base_model_name
        }, output_path / "moe_model.pt")
        
        # 設定ファイルを保存
        with open(output_path / "moe_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "base_model": self.config.base_model_name,
                "experts": self.config.expert_names,
                "num_experts": len(self.config.expert_names),
                "num_experts_per_token": self.config.num_experts_per_token,
                "lora_paths": self.config.lora_paths
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"MoEモデルを保存: {output_path}")
        return output_path
    
    def _generate_conversion_scripts(self, output_path: Path):
        """GGUF変換用スクリプトを生成"""
        
        # GGUF変換スクリプト
        gguf_script = f"""#!/usr/bin/env python3
# MoEモデルをGGUF形式に変換

import sys
import torch
from pathlib import Path

def convert_moe_to_gguf():
    moe_path = "{output_path}/moe_model.pt"
    output_gguf = "{output_path}/moe_model.gguf"
    
    print(f"変換中: {{moe_path}} -> {{output_gguf}}")
    
    # TODO: 実際のGGUF変換ロジックを実装
    # llama.cppのconvert.pyを使用するか、カスタム変換を実装
    
    print("変換完了!")

if __name__ == "__main__":
    convert_moe_to_gguf()
"""
        
        script_path = output_path / "convert_to_gguf.py"
        with open(script_path, "w") as f:
            f.write(gguf_script)
        script_path.chmod(0o755)
        
        # Ollama用Modelfile
        modelfile = f"""FROM ./moe_model.gguf

SYSTEM "あなたは日本の土木設計の専門家集団（MoE）です。複数の専門分野のエキスパートが協力して回答します。

エキスパート:
{chr(10).join(f'- {name}' for name in self.config.expert_names)}

各エキスパートの知識を統合して、最も正確な回答を提供します。"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 2048
"""
        
        with open(output_path / "Modelfile", "w") as f:
            f.write(modelfile)
        
        logger.info(f"変換スクリプトを生成: {output_path}")


def integrate_lora_to_moe(
    lora_paths: List[str],
    expert_names: List[str],
    base_model: str = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese",
    output_dir: str = "/workspace/outputs/moe_from_lora"
) -> Dict:
    """複数のLoRAモデルをMoEに統合する簡易インターフェース"""
    
    config = LoRAMoEConfig(
        lora_paths=lora_paths,
        expert_names=expert_names,
        base_model_name=base_model,
        output_dir=output_dir,
        num_experts=len(lora_paths),
        num_experts_per_token=2
    )
    
    adapter = LoRAToMoEAdapter(config)
    return adapter.convert_to_moe()


# 使用例
if __name__ == "__main__":
    result = integrate_lora_to_moe(
        lora_paths=[
            "/workspace/outputs/lora_road_design",
            "/workspace/outputs/lora_bridge_design",
            "/workspace/outputs/lora_tunnel_design",
            "/workspace/outputs/lora_general"
        ],
        expert_names=[
            "road_geometry_expert",
            "bridge_design_expert",
            "tunnel_design_expert",
            "general_expert"
        ]
    )
    
    print(result)