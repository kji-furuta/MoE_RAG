"""
Civil Engineering Domain Data Preparation
土木・建設分野の専門データ準備とエキスパート別分類
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import re
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DomainData:
    """ドメインデータクラス"""
    expert_domain: str
    question: str
    answer: str
    keywords: List[str]
    difficulty: str  # beginner, intermediate, advanced
    source: str  # JIS, 道路構造令, etc.
    metadata: Optional[Dict] = None


class CivilEngineeringDataPreparator:
    """土木・建設データ準備クラス"""
    
    def __init__(self, output_dir: str = "./data/civil_engineering"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # エキスパートドメインとキーワード定義
        self.domain_keywords = {
            "structural_design": {
                "keywords": [
                    "構造計算", "梁", "柱", "基礎", "スラブ", "耐震設計", "応力",
                    "モーメント", "せん断力", "たわみ", "座屈", "疲労", "振動",
                    "固有周期", "減衰", "地震力", "風荷重", "積載荷重", "死荷重"
                ],
                "standards": ["建築基準法", "道路橋示方書", "コンクリート標準示方書"],
                "sample_questions": [
                    "RC梁の曲げモーメントに対する設計方法を説明してください。",
                    "建築物の耐震設計における保有水平耐力計算とは何ですか？",
                    "プレストレストコンクリートの設計原理について教えてください。"
                ]
            },
            "road_design": {
                "keywords": [
                    "道路構造令", "設計速度", "平面線形", "縦断線形", "横断勾配",
                    "曲線半径", "緩和曲線", "視距", "交差点", "ランプ", "インターチェンジ",
                    "路肩", "中央分離帯", "歩道", "自転車道", "舗装構成"
                ],
                "standards": ["道路構造令", "道路設計基準", "舗装設計施工指針"],
                "sample_questions": [
                    "設計速度80km/hの道路における最小曲線半径を求めてください。",
                    "道路の横断勾配の標準値と特例値について説明してください。",
                    "交差点の設計で考慮すべき視距について教えてください。"
                ]
            },
            "geotechnical": {
                "keywords": [
                    "土質力学", "地盤調査", "N値", "支持力", "沈下", "圧密",
                    "液状化", "斜面安定", "土圧", "擁壁", "基礎", "杭基礎",
                    "地盤改良", "軟弱地盤", "盛土", "切土", "土留め"
                ],
                "standards": ["地盤工学会基準", "建築基礎構造設計指針", "道路土工指針"],
                "sample_questions": [
                    "標準貫入試験のN値から地盤の支持力を推定する方法を説明してください。",
                    "液状化の判定方法と対策工法について教えてください。",
                    "擁壁に作用する土圧の計算方法を説明してください。"
                ]
            },
            "hydraulics": {
                "keywords": [
                    "水理学", "流量計算", "管渠", "開水路", "マニング公式",
                    "ベルヌーイの定理", "損失水頭", "ポンプ", "配管", "雨水流出",
                    "洪水", "排水計画", "調整池", "浸透施設", "下水道"
                ],
                "standards": ["下水道施設計画・設計指針", "河川砂防技術基準"],
                "sample_questions": [
                    "マニング公式を用いた開水路の流量計算方法を説明してください。",
                    "雨水流出抑制施設の設計手法について教えてください。",
                    "管路の損失水頭の計算方法を説明してください。"
                ]
            },
            "materials": {
                "keywords": [
                    "コンクリート", "セメント", "骨材", "混和材", "配合設計",
                    "強度", "スランプ", "空気量", "鉄筋", "鋼材", "アスファルト",
                    "品質管理", "試験方法", "耐久性", "ひび割れ", "中性化"
                ],
                "standards": ["JIS規格", "コンクリート標準示方書", "鋼構造設計規準"],
                "sample_questions": [
                    "コンクリートの配合設計の手順を説明してください。",
                    "鉄筋の種類と特性について教えてください。",
                    "アスファルト舗装の品質管理項目を説明してください。"
                ]
            },
            "construction_management": {
                "keywords": [
                    "工程管理", "品質管理", "安全管理", "原価管理", "施工計画",
                    "仮設計画", "クリティカルパス", "PERT", "ガントチャート",
                    "労働安全衛生", "リスクアセスメント", "検査", "竣工"
                ],
                "standards": ["労働安全衛生法", "建設業法", "品確法"],
                "sample_questions": [
                    "クリティカルパス法による工程管理について説明してください。",
                    "建設現場の安全管理体制について教えてください。",
                    "品質管理における検査の種類と方法を説明してください。"
                ]
            },
            "regulations": {
                "keywords": [
                    "建築基準法", "都市計画法", "道路法", "河川法", "建設業法",
                    "JIS規格", "ISO", "建築確認", "開発許可", "道路占用許可",
                    "建設リサイクル法", "技術基準", "仕様書"
                ],
                "standards": ["各種法令", "技術基準", "標準仕様書"],
                "sample_questions": [
                    "建築確認申請の手続きと必要書類について説明してください。",
                    "道路構造令の主要な規定について教えてください。",
                    "建設リサイクル法の対象工事と手続きを説明してください。"
                ]
            },
            "environmental": {
                "keywords": [
                    "環境影響評価", "騒音", "振動", "大気汚染", "水質汚濁",
                    "廃棄物処理", "リサイクル", "省エネ", "CO2削減", "維持管理",
                    "点検", "補修", "長寿命化", "アセットマネジメント"
                ],
                "standards": ["環境影響評価法", "騒音規制法", "振動規制法", "廃棄物処理法"],
                "sample_questions": [
                    "建設工事における環境影響評価の手順を説明してください。",
                    "騒音・振動の測定方法と規制値について教えてください。",
                    "インフラの長寿命化計画の策定方法を説明してください。"
                ]
            }
        }
    
    def generate_training_data(self, num_samples_per_domain: int = 100) -> None:
        """トレーニングデータの生成"""
        logger.info(f"Generating {num_samples_per_domain} samples per domain...")
        
        for domain, domain_info in self.domain_keywords.items():
            domain_data = []
            
            # 基本的なQ&Aパターンの生成
            for i in range(num_samples_per_domain):
                data = self._generate_domain_sample(domain, domain_info, i)
                domain_data.append(asdict(data))
            
            # ファイルに保存
            self._save_domain_data(domain, domain_data, "train")
            
            logger.info(f"Generated {len(domain_data)} samples for {domain}")
    
    def _generate_domain_sample(self, domain: str, domain_info: Dict, index: int) -> DomainData:
        """ドメイン別サンプル生成"""
        # テンプレートベースの質問生成
        templates = [
            "{keyword}について説明してください。",
            "{keyword}の設計手法を教えてください。",
            "{keyword}における{standard}の規定は何ですか？",
            "{keyword}の計算方法を示してください。",
            "{keyword}と{related_keyword}の関係を説明してください。"
        ]
        
        # ランダムにキーワードと標準を選択
        np.random.seed(index)  # 再現性のため
        keyword = np.random.choice(domain_info["keywords"])
        standard = np.random.choice(domain_info.get("standards", ["一般基準"]))
        related_keyword = np.random.choice(
            [k for k in domain_info["keywords"] if k != keyword]
        )
        
        # テンプレートの選択と質問生成
        template = np.random.choice(templates)
        question = template.format(
            keyword=keyword,
            standard=standard,
            related_keyword=related_keyword
        )
        
        # サンプル質問も時々使用
        if index % 10 == 0 and "sample_questions" in domain_info:
            question = np.random.choice(domain_info["sample_questions"])
        
        # 回答の生成
        answer = self._generate_answer(domain, keyword, question)
        
        # 難易度の判定
        difficulty = self._determine_difficulty(question, answer)
        
        return DomainData(
            expert_domain=domain,
            question=question,
            answer=answer,
            keywords=[keyword],
            difficulty=difficulty,
            source=standard,
            metadata={"generated": True, "index": index}
        )
    
    def _generate_answer(self, domain: str, keyword: str, question: str) -> str:
        """回答生成"""
        # 実際の実装では、専門知識ベースやLLMを使用して詳細な回答を生成
        answer_templates = {
            "structural_design": (
                "{keyword}は、構造設計において重要な要素です。"
                "適切な設計により、安全性と経済性を両立させることができます。"
                "具体的には、荷重条件を考慮し、許容応力度設計法または限界状態設計法により設計を行います。"
            ),
            "road_design": (
                "{keyword}は、道路設計の基本要素であり、道路構造令に基づいて適切に設定する必要があります。"
                "安全で円滑な交通を確保するため、設計速度、交通量、地形条件等を総合的に考慮します。"
            ),
            "geotechnical": (
                "{keyword}は、地盤工学における重要な概念で、構造物の安定性に直接影響します。"
                "地盤調査結果に基づき、適切な基礎形式の選定と設計を行う必要があります。"
            ),
            "hydraulics": (
                "{keyword}は、水理設計において考慮すべき重要なパラメータです。"
                "流量、流速、水位等を適切に計算し、施設の規模を決定します。"
            ),
            "materials": (
                "{keyword}の品質は、構造物の耐久性に大きく影響するため、適切な管理が必要です。"
                "JIS規格等の基準に基づき、品質管理と検査を実施します。"
            ),
            "construction_management": (
                "{keyword}は、プロジェクトの成功に不可欠な管理要素です。"
                "計画・実行・評価のPDCAサイクルにより、継続的な改善を図ります。"
            ),
            "regulations": (
                "{keyword}に関する規定を遵守することは、法的要件を満たすために重要です。"
                "関連法令を適切に理解し、必要な手続きを確実に実施する必要があります。"
            ),
            "environmental": (
                "{keyword}への配慮は、持続可能な建設のために不可欠です。"
                "環境負荷の低減と地域社会との調和を図りながら事業を推進します。"
            )
        }
        
        base_answer = answer_templates.get(domain, "{keyword}について説明します。")
        return base_answer.format(keyword=keyword)
    
    def _determine_difficulty(self, question: str, answer: str) -> str:
        """難易度判定"""
        # 簡易的な難易度判定
        if "計算" in question or "設計" in question or "解析" in question:
            return "advanced"
        elif "説明" in question or "について" in question:
            return "intermediate"
        else:
            return "beginner"
    
    def _save_domain_data(self, domain: str, data: List[Dict], split: str) -> None:
        """ドメインデータの保存"""
        output_path = self.output_dir / split / f"{domain}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} samples to {output_path}")
    
    def create_validation_data(self, ratio: float = 0.1) -> None:
        """検証データの作成"""
        logger.info(f"Creating validation data with {ratio:.0%} split...")
        
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        
        if not train_dir.exists():
            logger.warning("Train directory does not exist, skipping validation data creation")
            return
        
        for domain_file in train_dir.glob("*.jsonl"):
            domain = domain_file.stem
            
            # データ読み込み
            with open(domain_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            # シャッフルと分割
            np.random.seed(42)  # 再現性のため
            np.random.shuffle(data)
            val_size = int(len(data) * ratio)
            
            val_data = data[:val_size]
            train_data = data[val_size:]
            
            # 保存
            self._save_domain_data(domain, val_data, "val")
            self._save_domain_data(domain, train_data, "train")
            
            logger.info(f"Split {domain}: {len(train_data)} train, {len(val_data)} val")
    
    def create_test_scenarios(self) -> None:
        """テストシナリオの作成"""
        test_scenarios = [
            {
                "scenario": "橋梁設計プロジェクト",
                "questions": [
                    "橋梁の設計荷重をどのように決定しますか？",
                    "橋脚の基礎設計における留意点は？",
                    "耐震設計における照査項目を教えてください。"
                ],
                "expected_experts": ["structural_design", "geotechnical", "regulations"]
            },
            {
                "scenario": "道路建設プロジェクト",
                "questions": [
                    "設計速度60km/hの道路の幾何構造設計について",
                    "路床の支持力確保のための対策は？",
                    "雨水排水施設の設計手順を説明してください。"
                ],
                "expected_experts": ["road_design", "geotechnical", "hydraulics"]
            },
            {
                "scenario": "建築物建設プロジェクト",
                "questions": [
                    "RC造建物の構造計算の流れは？",
                    "基礎の沈下対策について教えてください。",
                    "建築確認申請の必要書類は？"
                ],
                "expected_experts": ["structural_design", "geotechnical", "regulations"]
            }
        ]
        
        # テストシナリオの保存
        test_path = self.output_dir / "test_scenarios.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_scenarios, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created {len(test_scenarios)} test scenarios")
    
    def analyze_data_distribution(self) -> Dict:
        """データ分布の分析"""
        stats = []
        
        for split in ["train", "val"]:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
            
            for domain_file in split_dir.glob("*.jsonl"):
                domain = domain_file.stem
                
                with open(domain_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
                
                stats.append({
                    "split": split,
                    "domain": domain,
                    "count": len(data),
                    "avg_question_length": np.mean([len(d["question"]) for d in data]) if data else 0,
                    "avg_answer_length": np.mean([len(d["answer"]) for d in data]) if data else 0
                })
        
        # サマリーの表示
        print("\nData Distribution Summary:")
        print("=" * 60)
        for stat in stats:
            print(f"{stat['split']:5} | {stat['domain']:25} | Count: {stat['count']:4} | "
                  f"Q_len: {stat['avg_question_length']:.1f} | A_len: {stat['avg_answer_length']:.1f}")
        print("=" * 60)
        
        return {"statistics": stats}


def main():
    """メイン実行関数"""
    preparator = CivilEngineeringDataPreparator()
    
    # トレーニングデータの生成
    preparator.generate_training_data(num_samples_per_domain=100)
    
    # 検証データの作成
    preparator.create_validation_data(ratio=0.1)
    
    # テストシナリオの作成
    preparator.create_test_scenarios()
    
    # データ分布の分析
    stats = preparator.analyze_data_distribution()
    
    logger.info("Data preparation completed successfully!")
    
    return stats


if __name__ == "__main__":
    main()
