"""
MoE System Constants and Configuration
MoEシステムの定数と設定の一元管理
"""

from enum import Enum
from typing import Dict, Final

# Model Constants
DEFAULT_VOCAB_SIZE: Final[int] = 32000
SAFE_VOCAB_LIMIT: Final[int] = 30000
DEFAULT_HIDDEN_SIZE: Final[int] = 4096
DEFAULT_NUM_EXPERTS: Final[int] = 8
DEFAULT_EXPERTS_PER_TOKEN: Final[int] = 2
DEFAULT_MLP_RATIO: Final[int] = 4
DEFAULT_NUM_LAYERS: Final[int] = 12

# Training Constants
DEFAULT_LEARNING_RATE: Final[float] = 2e-5
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01
DEFAULT_AUX_LOSS_COEF: Final[float] = 0.01
DEFAULT_ROUTER_LR_MULTIPLIER: Final[float] = 0.1
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_CAPACITY_FACTOR: Final[float] = 1.25
DEFAULT_ROUTER_JITTER_NOISE: Final[float] = 0.1

# Path Constants
DEFAULT_OUTPUT_DIR: Final[str] = "./outputs/moe_civil_engineering"
DEFAULT_CHECKPOINT_DIR: Final[str] = "./checkpoints/moe_civil_engineering"
DEFAULT_DATASET_PATH: Final[str] = "./data/civil_engineering"

class ExpertType(Enum):
    """土木・建設分野の専門エキスパート定義"""
    STRUCTURAL_DESIGN = "structural_design"  # 構造設計
    ROAD_DESIGN = "road_design"  # 道路設計
    GEOTECHNICAL = "geotechnical"  # 地盤工学
    HYDRAULICS = "hydraulics"  # 水理・排水
    MATERIALS = "materials"  # 材料工学
    CONSTRUCTION_MGMT = "construction_management"  # 施工管理
    REGULATIONS = "regulations"  # 法規・基準
    ENVIRONMENTAL = "environmental"  # 環境・維持管理

# Expert Display Names (Japanese)
EXPERT_DISPLAY_NAMES: Final[Dict[int, str]] = {
    0: "構造設計",
    1: "道路設計",
    2: "地盤工学",
    3: "水理・排水",
    4: "材料工学",
    5: "施工管理",
    6: "法規・基準",
    7: "環境・維持管理"
}

# Expert Specialization Weights
DEFAULT_EXPERT_SPECIALIZATION: Final[Dict[str, float]] = {
    ExpertType.STRUCTURAL_DESIGN.value: 1.0,
    ExpertType.ROAD_DESIGN.value: 0.95,
    ExpertType.GEOTECHNICAL.value: 0.9,
    ExpertType.HYDRAULICS.value: 0.85,
    ExpertType.MATERIALS.value: 0.9,
    ExpertType.CONSTRUCTION_MGMT.value: 0.8,
    ExpertType.REGULATIONS.value: 1.0,
    ExpertType.ENVIRONMENTAL.value: 0.75,
}

# Domain Keywords for Each Expert
DOMAIN_KEYWORDS: Final[Dict[str, list]] = {
    ExpertType.STRUCTURAL_DESIGN.value: [
        "構造", "梁", "柱", "基礎", "耐震", "応力", "モーメント", "せん断"
    ],
    ExpertType.ROAD_DESIGN.value: [
        "道路", "舗装", "線形", "勾配", "カーブ", "交差点", "設計速度"
    ],
    ExpertType.GEOTECHNICAL.value: [
        "地盤", "土質", "支持力", "沈下", "液状化", "斜面", "擁壁"
    ],
    ExpertType.HYDRAULICS.value: [
        "排水", "流量", "管渠", "ポンプ", "貯留", "浸透", "洪水"
    ],
    ExpertType.MATERIALS.value: [
        "コンクリート", "鋼材", "アスファルト", "強度", "配合", "試験"
    ],
    ExpertType.CONSTRUCTION_MGMT.value: [
        "工程", "安全", "品質", "コスト", "施工", "管理", "工期"
    ],
    ExpertType.REGULATIONS.value: [
        "基準", "法規", "規格", "JIS", "道路構造令", "建築基準法"
    ],
    ExpertType.ENVIRONMENTAL.value: [
        "環境", "騒音", "振動", "廃棄物", "リサイクル", "維持", "点検"
    ]
}