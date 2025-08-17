"""
MoEモデルサービング
MoEモデルのロード、推論、エキスパート選択を提供
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys
import os

# MoEモジュールのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe.moe_architecture import CivilEngineeringMoEModel, MoEConfig, ExpertType
from moe.moe_architecture import DomainKeywordDetector

import logging
logger = logging.getLogger(__name__)


@dataclass
class MoEInferenceResult:
    """MoE推論結果"""
    text: str
    expert_scores: Dict[str, float]
    selected_experts: List[str]
    confidence: float
    hidden_states: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'expert_scores': self.expert_scores,
            'selected_experts': self.selected_experts,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }


class MoEModelServer:
    """MoEモデルサーバー"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[MoEConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        load_pretrained: bool = False
    ):
        """
        Args:
            model_path: 保存済みモデルのパス
            config: MoE設定
            device: 実行デバイス
            load_pretrained: 事前学習済みモデルをロードするか
        """
        self.device = device
        self.config = config or self._get_default_config()
        
        # モデルのロード
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
            logger.info(f"Loaded MoE model from {model_path}")
        else:
            self.model = CivilEngineeringMoEModel(self.config, base_model=None)
            logger.info("Created new MoE model")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # ドメインキーワード検出器
        self.keyword_detector = DomainKeywordDetector(self.config)
        
        # エキスパートタイプのマッピング
        self.expert_names = {
            0: "構造設計",
            1: "道路設計", 
            2: "地盤工学",
            3: "水理・排水",
            4: "材料工学",
            5: "施工管理",
            6: "法規・基準",
            7: "環境・維持管理"
        }
        
        logger.info(f"MoE Model Server initialized on {device}")
    
    def _get_default_config(self) -> MoEConfig:
        """デフォルト設定を取得"""
        return MoEConfig(
            hidden_size=512,
            num_experts=8,
            num_experts_per_tok=2,
            expert_capacity_factor=1.25,
            domain_specific_routing=True,
            aux_loss_coef=0.01
        )
    
    def _load_model(self, model_path: str) -> CivilEngineeringMoEModel:
        """保存済みモデルをロード"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CivilEngineeringMoEModel(self.config, base_model=None)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @torch.no_grad()
    def infer(
        self,
        text: str,
        max_length: int = 512,
        return_hidden_states: bool = False,
        temperature: float = 1.0
    ) -> MoEInferenceResult:
        """
        テキストに対してMoE推論を実行
        
        Args:
            text: 入力テキスト
            max_length: 最大トークン長
            return_hidden_states: 隠れ状態を返すか
            temperature: サンプリング温度
            
        Returns:
            推論結果
        """
        # トークン化（簡易実装）
        input_ids = self._tokenize(text, max_length)
        attention_mask = torch.ones_like(input_ids)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # MoEモデルの推論
        outputs = self.model(input_ids, attention_mask)
        
        # エキスパート選択情報の取得
        router_info = outputs.get('router_info', [])
        expert_scores, selected_experts = self._extract_expert_info(router_info)
        
        # 信頼度スコアの計算
        confidence = self._calculate_confidence(expert_scores)
        
        # テキスト生成（簡易実装）
        generated_text = self._generate_text(outputs['hidden_states'], temperature)
        
        return MoEInferenceResult(
            text=generated_text,
            expert_scores=expert_scores,
            selected_experts=selected_experts,
            confidence=confidence,
            hidden_states=outputs['hidden_states'] if return_hidden_states else None,
            metadata={
                'aux_loss': outputs.get('aux_loss', 0).item() if torch.is_tensor(outputs.get('aux_loss')) else 0,
                'num_tokens': input_ids.shape[1]
            }
        )
    
    def get_expert_for_query(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """
        クエリに最適なエキスパートを選択
        
        Args:
            query: クエリテキスト
            
        Returns:
            (選択されたエキスパート名, エキスパートスコア)
        """
        # ドメインキーワード検出
        input_ids = self._tokenize(query, 256)
        domain_keywords = self.keyword_detector(input_ids.to(self.device))
        
        # エキスパートスコアの計算
        expert_scores = {}
        for expert_type in ExpertType:
            score = 0.0
            if expert_type.value in domain_keywords:
                score = 1.0
            expert_scores[self.expert_names[list(ExpertType).index(expert_type)]] = score
        
        # Top-Kエキスパートの選択
        sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, score in sorted_experts[:self.config.num_experts_per_tok]]
        
        return selected, expert_scores
    
    def _tokenize(self, text: str, max_length: int) -> torch.Tensor:
        """簡易トークン化"""
        # 実際の実装では適切なトークナイザーを使用
        vocab_size = 30000  # 安全な範囲
        # ダミー実装
        tokens = torch.randint(100, min(vocab_size, 20000), (1, min(len(text.split()), max_length)))
        return tokens
    
    def _extract_expert_info(
        self,
        router_info: List[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, float], List[str]]:
        """ルーター情報からエキスパート情報を抽出"""
        expert_scores = {name: 0.0 for name in self.expert_names.values()}
        expert_counts = {i: 0 for i in range(self.config.num_experts)}
        
        if not router_info:
            return expert_scores, []
        
        # 各レイヤーのルーティング情報を集計
        for info in router_info:
            if 'expert_indices' in info:
                indices = info['expert_indices'].cpu().numpy().flatten()
                for idx in indices:
                    if 0 <= idx < self.config.num_experts:
                        expert_counts[idx] += 1
        
        # スコアの正規化
        total_count = sum(expert_counts.values())
        if total_count > 0:
            for idx, count in expert_counts.items():
                expert_name = self.expert_names.get(idx, f"Expert_{idx}")
                expert_scores[expert_name] = count / total_count
        
        # 選択されたエキスパート
        selected_experts = [
            self.expert_names[idx]
            for idx, count in expert_counts.items()
            if count > 0
        ]
        
        return expert_scores, selected_experts
    
    def _calculate_confidence(self, expert_scores: Dict[str, float]) -> float:
        """信頼度スコアを計算"""
        if not expert_scores:
            return 0.0
        
        # エントロピーベースの信頼度計算
        scores = np.array(list(expert_scores.values()))
        scores = scores[scores > 0]
        
        if len(scores) == 0:
            return 0.0
        
        # 正規化
        scores = scores / scores.sum()
        
        # エントロピー
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        max_entropy = np.log(len(scores))
        
        # 信頼度（低エントロピー = 高信頼度）
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return float(confidence)
    
    def _generate_text(self, hidden_states: torch.Tensor, temperature: float) -> str:
        """隠れ状態からテキストを生成（簡易実装）"""
        # 実際の実装では適切なデコーダーを使用
        # ここではダミーテキストを返す
        return "MoEモデルによる専門的な回答がここに生成されます。"
    
    def save_model(self, save_path: str):
        """モデルを保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            'num_experts': self.config.num_experts,
            'num_experts_per_tok': self.config.num_experts_per_tok,
            'hidden_size': self.config.hidden_size,
            'total_parameters': param_count,
            'device': str(self.device),
            'expert_types': list(self.expert_names.values())
        }


# 使用例
if __name__ == "__main__":
    # MoEサーバーの初期化
    server = MoEModelServer()
    
    # クエリに対する推論
    query = "道路の設計速度と曲線半径の関係について教えてください"
    result = server.infer(query)
    
    print(f"Query: {query}")
    print(f"Response: {result.text}")
    print(f"Selected Experts: {result.selected_experts}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Expert Scores: {result.expert_scores}")