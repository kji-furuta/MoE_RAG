"""
エキスパートルーター
クエリに基づいて最適なMoEエキスパートを選択するルーティング機能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """ルーティング決定結果"""
    primary_experts: List[str]
    secondary_experts: List[str]
    expert_scores: Dict[str, float]
    routing_strategy: str
    confidence: float
    keywords_detected: Dict[str, List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_experts': self.primary_experts,
            'secondary_experts': self.secondary_experts,
            'expert_scores': self.expert_scores,
            'routing_strategy': self.routing_strategy,
            'confidence': self.confidence,
            'keywords_detected': self.keywords_detected
        }


class ExpertRouter:
    """エキスパートルーター"""
    
    def __init__(self, num_experts_per_tok: int = 2):
        """
        Args:
            num_experts_per_tok: トークンごとに選択するエキスパート数
        """
        self.num_experts_per_tok = num_experts_per_tok
        
        # エキスパートドメインの定義
        self.expert_domains = {
            "構造設計": {
                "keywords": ["構造", "荷重", "応力", "変形", "耐震", "設計", "梁", "柱", "基礎", "フレーム"],
                "patterns": [r"構造.*設計", r"耐震.*性能", r"荷重.*計算"],
                "weight": 1.0
            },
            "道路設計": {
                "keywords": ["道路", "曲線", "勾配", "視距", "設計速度", "交差点", "線形", "平面", "縦断", "横断"],
                "patterns": [r"道路.*設計", r"設計速度.*\d+km", r"曲線半径"],
                "weight": 1.0
            },
            "地盤工学": {
                "keywords": ["地盤", "土質", "支持力", "沈下", "液状化", "地質", "N値", "土圧", "斜面", "擁壁"],
                "patterns": [r"地盤.*調査", r"支持力.*計算", r"液状化.*判定"],
                "weight": 1.0
            },
            "水理・排水": {
                "keywords": ["水理", "排水", "流量", "管渠", "雨水", "下水", "流速", "水位", "浸水", "治水"],
                "patterns": [r"排水.*計画", r"流量.*計算", r"雨水.*処理"],
                "weight": 1.0
            },
            "材料工学": {
                "keywords": ["材料", "コンクリート", "鋼材", "強度", "耐久性", "配合", "セメント", "骨材", "鉄筋"],
                "patterns": [r"コンクリート.*強度", r"材料.*試験", r"配合.*設計"],
                "weight": 1.0
            },
            "施工管理": {
                "keywords": ["施工", "工程", "品質", "安全", "管理", "検査", "工法", "仮設", "重機", "労務"],
                "patterns": [r"施工.*計画", r"品質.*管理", r"安全.*対策"],
                "weight": 1.0
            },
            "法規・基準": {
                "keywords": ["法規", "基準", "規格", "仕様", "規定", "標準", "指針", "要領", "法令", "通達"],
                "patterns": [r".*基準", r".*規格", r".*仕様書"],
                "weight": 1.0
            },
            "環境・維持管理": {
                "keywords": ["環境", "維持", "管理", "点検", "補修", "劣化", "診断", "更新", "LCC", "アセット"],
                "patterns": [r"維持.*管理", r"環境.*影響", r"劣化.*診断"],
                "weight": 1.0
            }
        }
        
        # 複合ドメインの定義（複数エキスパートが必要なケース）
        self.composite_domains = {
            "橋梁設計": ["構造設計", "材料工学", "地盤工学"],
            "トンネル工事": ["地盤工学", "施工管理", "材料工学"],
            "河川改修": ["水理・排水", "環境・維持管理", "地盤工学"],
            "道路新設": ["道路設計", "地盤工学", "環境・維持管理"]
        }
        
        logger.info(f"Expert Router initialized with {len(self.expert_domains)} experts")
    
    def route(self, query: str) -> RoutingDecision:
        """
        クエリに基づいてエキスパートをルーティング
        
        Args:
            query: 入力クエリ
            
        Returns:
            ルーティング決定結果
        """
        # 1. キーワード検出
        keywords_detected = self._detect_keywords(query)
        
        # 2. パターンマッチング
        pattern_scores = self._pattern_matching(query)
        
        # 3. 複合ドメイン検出
        composite_match = self._detect_composite_domain(query)
        
        # 4. スコア統合
        expert_scores = self._integrate_scores(
            keywords_detected, pattern_scores, composite_match
        )
        
        # 5. エキスパート選択
        primary_experts, secondary_experts = self._select_experts(expert_scores)
        
        # 6. ルーティング戦略の決定
        routing_strategy = self._determine_strategy(
            primary_experts, secondary_experts, composite_match
        )
        
        # 7. 信頼度計算
        confidence = self._calculate_confidence(expert_scores, primary_experts)
        
        return RoutingDecision(
            primary_experts=primary_experts,
            secondary_experts=secondary_experts,
            expert_scores=expert_scores,
            routing_strategy=routing_strategy,
            confidence=confidence,
            keywords_detected=keywords_detected
        )
    
    def _detect_keywords(self, query: str) -> Dict[str, List[str]]:
        """キーワード検出"""
        query_lower = query.lower()
        detected = defaultdict(list)
        
        for expert, config in self.expert_domains.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    detected[expert].append(keyword)
        
        return dict(detected)
    
    def _pattern_matching(self, query: str) -> Dict[str, float]:
        """パターンマッチングによるスコアリング"""
        scores = {}
        
        for expert, config in self.expert_domains.items():
            score = 0.0
            for pattern in config["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
            
            if score > 0:
                scores[expert] = score / len(config["patterns"])
        
        return scores
    
    def _detect_composite_domain(self, query: str) -> Optional[str]:
        """複合ドメインの検出"""
        query_lower = query.lower()
        
        composite_keywords = {
            "橋梁設計": ["橋", "橋梁", "橋脚", "橋台"],
            "トンネル工事": ["トンネル", "坑口", "覆工"],
            "河川改修": ["河川", "堤防", "護岸", "水門"],
            "道路新設": ["道路新設", "新規路線", "バイパス"]
        }
        
        for domain, keywords in composite_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        
        return None
    
    def _integrate_scores(
        self,
        keywords_detected: Dict[str, List[str]],
        pattern_scores: Dict[str, float],
        composite_match: Optional[str]
    ) -> Dict[str, float]:
        """スコアの統合"""
        integrated_scores = defaultdict(float)
        
        # キーワードスコア
        for expert, keywords in keywords_detected.items():
            integrated_scores[expert] += len(keywords) * 0.3
        
        # パターンスコア
        for expert, score in pattern_scores.items():
            integrated_scores[expert] += score * 0.5
        
        # 複合ドメインボーナス
        if composite_match and composite_match in self.composite_domains:
            for expert in self.composite_domains[composite_match]:
                integrated_scores[expert] += 0.2
        
        # 正規化
        max_score = max(integrated_scores.values()) if integrated_scores else 1.0
        if max_score > 0:
            for expert in integrated_scores:
                integrated_scores[expert] /= max_score
        
        return dict(integrated_scores)
    
    def _select_experts(
        self,
        expert_scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """エキスパートの選択"""
        if not expert_scores:
            # デフォルトエキスパート
            return ["道路設計", "構造設計"], []
        
        # スコアでソート
        sorted_experts = sorted(
            expert_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # プライマリエキスパート（Top-K）
        primary_experts = [
            expert for expert, score in sorted_experts[:self.num_experts_per_tok]
            if score > 0.1  # 閾値
        ]
        
        # セカンダリエキスパート（次点）
        secondary_experts = [
            expert for expert, score in sorted_experts[self.num_experts_per_tok:self.num_experts_per_tok + 2]
            if score > 0.05  # より低い閾値
        ]
        
        # 最低1つのエキスパートを保証
        if not primary_experts:
            primary_experts = [sorted_experts[0][0]] if sorted_experts else ["道路設計"]
        
        return primary_experts, secondary_experts
    
    def _determine_strategy(
        self,
        primary_experts: List[str],
        secondary_experts: List[str],
        composite_match: Optional[str]
    ) -> str:
        """ルーティング戦略の決定"""
        if composite_match:
            return "composite_domain"
        elif len(primary_experts) > 1:
            return "multi_expert"
        elif secondary_experts:
            return "primary_with_backup"
        else:
            return "single_expert"
    
    def _calculate_confidence(
        self,
        expert_scores: Dict[str, float],
        primary_experts: List[str]
    ) -> float:
        """信頼度の計算"""
        if not primary_experts or not expert_scores:
            return 0.0
        
        # プライマリエキスパートの平均スコア
        primary_scores = [
            expert_scores.get(expert, 0.0) for expert in primary_experts
        ]
        avg_score = np.mean(primary_scores) if primary_scores else 0.0
        
        # スコアの分散（低い方が良い = 明確な選択）
        all_scores = list(expert_scores.values())
        if len(all_scores) > 1:
            variance = np.var(all_scores)
            clarity_bonus = 0.2 * (1 - min(variance, 1.0))
        else:
            clarity_bonus = 0.0
        
        confidence = min(avg_score + clarity_bonus, 1.0)
        
        return float(confidence)
    
    def get_expert_info(self, expert_name: str) -> Dict[str, Any]:
        """特定のエキスパートの情報を取得"""
        if expert_name not in self.expert_domains:
            return {}
        
        config = self.expert_domains[expert_name]
        return {
            'name': expert_name,
            'keywords': config['keywords'],
            'patterns': [p for p in config['patterns']],
            'weight': config['weight']
        }
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """クエリの複雑度を分析"""
        # 文の長さ
        query_length = len(query)
        
        # 専門用語の数
        technical_terms = 0
        for config in self.expert_domains.values():
            for keyword in config['keywords']:
                if keyword in query.lower():
                    technical_terms += 1
        
        # 数値の存在
        has_numbers = bool(re.search(r'\d+', query))
        
        # 疑問詞の種類
        question_types = []
        question_words = {
            'what': ['何', 'なに', 'どの', 'どんな'],
            'how': ['どう', 'どのように', 'いかに'],
            'why': ['なぜ', 'どうして'],
            'when': ['いつ', 'いつまで'],
            'where': ['どこ', 'どこで']
        }
        
        for q_type, words in question_words.items():
            if any(word in query for word in words):
                question_types.append(q_type)
        
        # 複雑度スコア
        complexity_score = (
            min(query_length / 100, 1.0) * 0.3 +
            min(technical_terms / 5, 1.0) * 0.4 +
            (0.2 if has_numbers else 0) +
            len(question_types) * 0.1
        )
        
        return {
            'query_length': query_length,
            'technical_terms': technical_terms,
            'has_numbers': has_numbers,
            'question_types': question_types,
            'complexity_score': min(complexity_score, 1.0)
        }


# 使用例
if __name__ == "__main__":
    router = ExpertRouter()
    
    # テストクエリ
    queries = [
        "設計速度80km/hの道路における最小曲線半径は？",
        "橋梁の耐震設計における地震動の考慮方法",
        "コンクリートの配合設計と品質管理について",
        "トンネル掘削時の地盤調査方法"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        decision = router.route(query)
        print(f"Primary Experts: {decision.primary_experts}")
        print(f"Strategy: {decision.routing_strategy}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Keywords: {decision.keywords_detected}")
        
        complexity = router.analyze_query_complexity(query)
        print(f"Complexity Score: {complexity['complexity_score']:.2f}")