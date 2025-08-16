#!/usr/bin/env python3
# scripts/create_sample_continual_datasets.py
"""
継続学習用のサンプルデータセットを作成
"""

import json
from pathlib import Path

def create_sample_datasets():
    """サンプルデータセットの作成"""
    
    # データディレクトリの作成
    data_dir = Path("data/continual")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 橋梁関連データセット
    bridges_data = [
        {
            "instruction": "橋梁の設計で最も重要な要素は何ですか？",
            "output": "橋梁設計において最も重要な要素は、構造的安全性と耐久性です。これには、適切な材料選定、荷重計算、地震や風に対する耐性、そして長期的な維持管理計画が含まれます。"
        },
        {
            "instruction": "吊り橋と斜張橋の違いを説明してください。",
            "output": "吊り橋は主ケーブルから垂直に吊り下げられたハンガーロープで橋桁を支える構造です。一方、斜張橋は主塔から直接斜めにケーブルを張って橋桁を支えます。吊り橋は長大スパンに適し、斜張橋は中規模スパンで経済的です。"
        },
        {
            "instruction": "橋梁の耐震設計について教えてください。",
            "output": "橋梁の耐震設計では、地震時の水平力と鉛直力を考慮し、橋脚の靭性設計、免震・制震装置の設置、落橋防止システムの導入などを行います。レベル1地震動とレベル2地震動に対する性能を確保することが重要です。"
        }
    ]
    
    # 2. トンネル関連データセット
    tunnels_data = [
        {
            "instruction": "トンネル掘削工法にはどのような種類がありますか？",
            "output": "主なトンネル掘削工法には、NATM（新オーストリアトンネル工法）、TBM（トンネルボーリングマシン）工法、開削工法、沈埋工法があります。地質条件や施工環境に応じて適切な工法を選択します。"
        },
        {
            "instruction": "トンネルの換気システムについて説明してください。",
            "output": "トンネル換気システムは、車両の排気ガスを除去し、新鮮な空気を供給するために不可欠です。縦流式、横流式、半横流式などの方式があり、トンネルの長さや交通量に応じて設計されます。"
        },
        {
            "instruction": "山岳トンネルの支保工について教えてください。",
            "output": "山岳トンネルの支保工には、吹付けコンクリート、ロックボルト、鋼製支保工などがあります。これらを組み合わせて地山の自立性を活かしながら安全に掘削を進めます。地山等級に応じた支保パターンを選定することが重要です。"
        }
    ]
    
    # 3. 交差点関連データセット
    intersections_data = [
        {
            "instruction": "ラウンドアバウトの利点と欠点を説明してください。",
            "output": "ラウンドアバウトの利点は、信号待ちがなく交通流が円滑、重大事故が減少、維持管理費が低いことです。欠点は、大型車の通行が困難、歩行者横断が複雑、設置に広い用地が必要なことです。"
        },
        {
            "instruction": "交差点の信号制御方式について教えてください。",
            "output": "交差点の信号制御には、定周期制御、感応制御、系統制御などがあります。定周期は一定の周期で動作し、感応制御は交通量を検知して制御、系統制御は複数の交差点を連動させて交通流を最適化します。"
        },
        {
            "instruction": "交差点設計における視距の確保について説明してください。",
            "output": "交差点設計では、運転者が他の車両や歩行者を十分に視認できる視距の確保が重要です。停止視距、交差点視距を確保し、植栽や構造物による視界の遮蔽を避ける必要があります。"
        }
    ]
    
    # データセットの保存
    datasets = {
        "bridges_dataset.jsonl": bridges_data,
        "tunnels_dataset.jsonl": tunnels_data,
        "intersections_dataset.jsonl": intersections_data
    }
    
    for filename, data in datasets.items():
        filepath = data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"Created: {filepath}")
    
    # テストデータセットも作成（各1件）
    for filename, data in datasets.items():
        test_filename = filename.replace('.jsonl', '_test.jsonl')
        test_filepath = data_dir / test_filename
        with open(test_filepath, 'w', encoding='utf-8') as f:
            json.dump(data[0], f, ensure_ascii=False)
            f.write('\n')
        print(f"Created: {test_filepath}")

if __name__ == "__main__":
    create_sample_datasets()
    print("\nサンプルデータセットの作成が完了しました。")
    print("継続学習を実行するには以下のコマンドを使用してください：")
    print("bash scripts/run_continual_learning.sh")
