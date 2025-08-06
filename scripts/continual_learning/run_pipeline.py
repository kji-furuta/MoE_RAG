#!/usr/bin/env python3
# scripts/continual_learning/run_pipeline.py
"""
継続学習パイプラインの自動実行スクリプト
"""

import argparse
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
import torch

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline
from src.evaluation.continual_metrics import ContinualLearningEvaluator

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/continual_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """設定ファイルの読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def validate_config(config: dict) -> bool:
    """設定の検証"""
    required_keys = ['continual_learning', 'continual_learning.base_model', 
                    'continual_learning.tasks']
    
    for key in required_keys:
        keys = key.split('.')
        current = config
        for k in keys:
            if k not in current:
                logger.error(f"必須キーが見つかりません: {key}")
                return False
            current = current[k]
    
    return True

def run_continual_learning_pipeline(args):
    """継続学習パイプラインの実行"""
    
    # 設定の読み込み
    logger.info(f"設定ファイルを読み込み中: {args.task_config}")
    config = load_config(args.task_config)
    
    if not validate_config(config):
        logger.error("設定ファイルの検証に失敗しました")
        return 1
    
    cl_config = config['continual_learning']
    
    # パイプラインの初期化
    logger.info("継続学習パイプラインを初期化中...")
    pipeline = ContinualLearningPipeline(
        base_model_path=args.base_model or cl_config['base_model']['path']
    )
    
    # ベースモデルのロード
    logger.info("ベースモデルをロード中...")
    if cl_config['base_model']['type'] == 'full_finetuned':
        model = pipeline.load_finetuned_model(pipeline.base_model_path)
    else:
        from src.models.base_model import BaseModel
        model = BaseModel.from_pretrained(pipeline.base_model_path)
    
    # EWC設定
    ewc_config = cl_config.get('ewc_config', {})
    ewc_lambda = args.ewc_lambda or ewc_config.get('lambda', 5000)
    
    # 評価器の初期化
    evaluator = ContinualLearningEvaluator()
    
    # 各タスクの実行
    for task_idx, task in enumerate(cl_config['tasks']):
        logger.info(f"\n{'='*50}")
        logger.info(f"タスク {task_idx + 1}/{len(cl_config['tasks'])}: {task['name']}")
        logger.info(f"{'='*50}")
        
        # データセットの確認
        dataset_path = Path(task['dataset'])
        if not dataset_path.exists():
            logger.error(f"データセットが見つかりません: {dataset_path}")
            continue
        
        # タスクの実行
        try:
            model = pipeline.run_continual_task(
                model=model,
                task_name=task['name'],
                train_dataset_path=str(dataset_path),
                epochs=task.get('epochs', 2),
                use_previous_fisher=task_idx > 0,  # 最初のタスク以外はFisherを使用
                fisher_importance=ewc_lambda
            )
            
            # 前のタスクの評価
            if args.monitor_forgetting and task_idx > 0:
                logger.info("破滅的忘却の評価中...")
                forgetting_results = evaluator.evaluate_forgetting(
                    model, pipeline.task_history
                )
                
                # レポート生成
                if forgetting_results:
                    report_path = evaluator.generate_report(forgetting_results)
                    logger.info(f"評価レポートを保存: {report_path}")
            
        except Exception as e:
            logger.error(f"タスク実行エラー: {str(e)}", exc_info=True)
            if not args.continue_on_error:
                return 1
    
    # 最終評価
    if len(pipeline.task_history) > 1:
        logger.info("\n最終評価を実行中...")
        final_results = evaluator.evaluate_forgetting(model, pipeline.task_history)
        
        # 最終レポート生成
        final_report_path = evaluator.generate_report(
            final_results,
            save_path=Path(args.log_dir) / "final_forgetting_analysis.png"
        )
        logger.info(f"最終評価レポート: {final_report_path}")
        
        # 学習曲線のプロット
        curves_path = evaluator.plot_learning_curves(
            pipeline.task_history,
            save_path=Path(args.log_dir) / "learning_curves.png"
        )
        logger.info(f"学習曲線: {curves_path}")
    
    # サマリーの出力
    logger.info("\n" + "="*50)
    logger.info("継続学習パイプライン完了")
    logger.info(f"実行タスク数: {len(pipeline.task_history)}")
    logger.info(f"最終モデル: {pipeline.task_history[-1]['model_path'] if pipeline.task_history else 'なし'}")
    logger.info("="*50)
    
    # メモリクリーンアップ
    if args.cleanup_memory:
        del model
        torch.cuda.empty_cache()
        logger.info("メモリをクリーンアップしました")
    
    return 0

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="継続学習パイプラインの自動実行"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        help="ベースモデルのパス（設定ファイルの値を上書き）"
    )
    
    parser.add_argument(
        "--task_config",
        type=str,
        required=True,
        help="タスク設定ファイルのパス"
    )
    
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        help="EWCの重要度パラメータ（設定ファイルの値を上書き）"
    )
    
    parser.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="すべてのチェックポイントを保存"
    )
    
    parser.add_argument(
        "--use_memory_efficient_fisher",
        action="store_true",
        help="メモリ効率的なFisher行列計算を使用"
    )
    
    parser.add_argument(
        "--monitor_forgetting",
        action="store_true",
        help="各タスク後に破滅的忘却を監視"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/continual_learning",
        help="ログディレクトリ"
    )
    
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="エラー発生時も継続"
    )
    
    parser.add_argument(
        "--cleanup_memory",
        action="store_true",
        help="実行後にメモリをクリーンアップ"
    )
    
    args = parser.parse_args()
    
    # ログディレクトリの作成
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # パイプラインの実行
    try:
        exit_code = run_continual_learning_pipeline(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
