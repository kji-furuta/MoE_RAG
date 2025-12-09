#!/usr/bin/env python3
"""
RARdata.json更新後の継続学習スクリプト

このスクリプトは、RARdata.jsonにデータを追加または変更した後、
継続学習を実行して既存の知識を保持しながら新しい知識を学習します。

使用方法:
    # 基本的な使用
    python scripts/update_rardata_model.py

    # データファイルを指定
    python scripts/update_rardata_model.py --data-path /workspace/RARdata_v2.json

    # 前回のモデルを指定
    python scripts/update_rardata_model.py --base-model outputs/continual_rardata_training_*/checkpoint-final

    # Fisher行列をリセット（全体再学習）
    python scripts/update_rardata_model.py --reset-fisher

特徴:
    ✅ EWC (Elastic Weight Consolidation) による破滅的忘却の防止
    ✅ 前回の学習内容を保持しながら新しい知識を追加
    ✅ 自動的に前回の学習済みモデルを検出
    ✅ データ品質チェック機能
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.continual_learning_pipeline import ContinualLearningPipeline

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_model(base_dir: str = "outputs") -> Optional[str]:
    """
    最新の学習済みモデルを自動検出

    Args:
        base_dir: モデルディレクトリのベースパス

    Returns:
        最新モデルのパス、見つからない場合はNone
    """
    base_path = Path(base_dir)

    # continual_rardata_* パターンのディレクトリを検索
    pattern = "continual_rardata_*"
    model_dirs = sorted(base_path.glob(pattern), reverse=True)

    for model_dir in model_dirs:
        checkpoint_dir = model_dir / "checkpoint-final"
        if checkpoint_dir.exists():
            logger.info(f"最新モデルを検出: {checkpoint_dir}")
            return str(checkpoint_dir)

    logger.warning("学習済みモデルが見つかりません")
    return None


def validate_rar_data(file_path: str) -> Tuple[bool, List[str], Dict[str, any]]:
    """
    RARデータの品質チェック

    Args:
        file_path: データファイルのパス

    Returns:
        (is_valid, error_messages, statistics)
    """
    errors = []
    stats = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"ファイル読み込みエラー: {e}"], {}

    stats['total_entries'] = len(data)

    # ID重複チェック
    ids = [entry.get('id') for entry in data]
    id_counts = {}
    for id_val in ids:
        id_counts[id_val] = id_counts.get(id_val, 0) + 1

    duplicates = [id_val for id_val, count in id_counts.items() if count > 1]
    if duplicates:
        errors.append(f"ID重複: {duplicates}")

    stats['id_range'] = f"{ids[0]} ～ {ids[-1]}" if ids else "N/A"

    # 必須フィールドチェック
    required_fields = ['id', 'instruction', 'documents', 'output']

    for i, entry in enumerate(data):
        for field in required_fields:
            if field not in entry:
                errors.append(f"エントリ {i} (ID: {entry.get('id', 'N/A')}): {field} が欠落")

        # output内の必須フィールド
        if 'output' in entry:
            output = entry['output']
            if 'chain_of_thought' not in output:
                errors.append(f"エントリ {i}: chain_of_thought が欠落")
            if 'final_answer' not in output:
                errors.append(f"エントリ {i}: final_answer が欠落")
            if 'citations' not in output:
                errors.append(f"エントリ {i}: citations が欠落")

    # Oracle文書の比率
    oracle_count = 0
    total_docs = 0

    for entry in data:
        docs = entry.get('documents', [])
        total_docs += len(docs)
        oracle_count += sum(1 for doc in docs if doc.get('is_oracle', False))

    if total_docs > 0:
        oracle_ratio = oracle_count / total_docs
        stats['oracle_ratio'] = f"{oracle_ratio:.1%}"

        if oracle_ratio < 0.5 or oracle_ratio > 0.8:
            errors.append(f"Oracle比率が推奨範囲外: {oracle_ratio:.1%} (推奨: 50-80%)")
    else:
        stats['oracle_ratio'] = "N/A"

    # Citation整合性
    for i, entry in enumerate(data):
        citations = entry.get('output', {}).get('citations', [])
        docs = entry.get('documents', [])
        doc_sources = {doc.get('source') for doc in docs}

        for citation in citations:
            if citation.get('source') not in doc_sources:
                errors.append(f"エントリ {i}: Citation元が documents に存在しない")

    if len(data) == 0:
        errors.append("データが空です")

    return len(errors) == 0, errors, stats


def main():
    """メイン実行関数"""

    parser = argparse.ArgumentParser(
        description="RARdata.json更新後の継続学習",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用（最新モデルを自動検出）
  python scripts/update_rardata_model.py

  # データファイルを指定
  python scripts/update_rardata_model.py --data-path /workspace/RARdata_v2.json

  # ベースモデルを明示的に指定
  python scripts/update_rardata_model.py --base-model outputs/continual_rardata_training_20251208_141953/checkpoint-final

  # Fisher行列をリセットして全体再学習
  python scripts/update_rardata_model.py --reset-fisher
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='/workspace/RARdata.json',
        help='学習データファイルのパス (デフォルト: /workspace/RARdata.json)'
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        help='ベースモデルのパス（指定しない場合は最新モデルを自動検出）'
    )

    parser.add_argument(
        '--task-name',
        type=str,
        default=None,
        help='タスク名（指定しない場合は自動生成）'
    )

    parser.add_argument(
        '--ewc-lambda',
        type=float,
        default=5000.0,
        help='EWC正則化の強度 (デフォルト: 5000)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='学習エポック数 (デフォルト: 3)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='バッチサイズ (デフォルト: 1)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='学習率 (デフォルト: 2e-5)'
    )

    parser.add_argument(
        '--reset-fisher',
        action='store_true',
        help='Fisher行列をリセット（全体再学習）'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='データ品質チェックをスキップ'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RARdata.json 更新後の継続学習")
    logger.info("=" * 80)

    # データ品質チェック
    if not args.skip_validation:
        logger.info("データ品質チェック中...")
        is_valid, errors, stats = validate_rar_data(args.data_path)

        logger.info(f"総エントリ数: {stats.get('total_entries', 0)}")
        logger.info(f"ID範囲: {stats.get('id_range', 'N/A')}")
        logger.info(f"Oracle比率: {stats.get('oracle_ratio', 'N/A')}")

        if not is_valid:
            logger.error("❌ データ品質チェック失敗:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        else:
            logger.info("✅ データ品質チェック合格")

    # ベースモデルの決定
    if args.base_model is None:
        base_model_path = find_latest_model()
        if base_model_path is None:
            logger.error("学習済みモデルが見つかりません。--base-model で明示的に指定してください。")
            sys.exit(1)
    else:
        base_model_path = args.base_model

    # タスク名の生成
    if args.task_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_name = f"rardata_update_{timestamp}"
    else:
        task_name = args.task_name

    # パラメータ表示
    logger.info("")
    logger.info("学習設定:")
    logger.info(f"  タスク名: {task_name}")
    logger.info(f"  データパス: {args.data_path}")
    logger.info(f"  ベースモデル: {base_model_path}")
    logger.info(f"  EWC Lambda: {args.ewc_lambda}")
    logger.info(f"  エポック数: {args.epochs}")
    logger.info(f"  バッチサイズ: {args.batch_size}")
    logger.info(f"  学習率: {args.learning_rate}")
    logger.info(f"  Fisher行列使用: {not args.reset_fisher}")
    logger.info("")

    # 継続学習パイプライン初期化
    logger.info("継続学習パイプラインを初期化中...")
    pipeline = ContinualLearningPipeline(
        base_model_path=None,
        use_efficient_fisher=True
    )

    # モデルロード
    logger.info(f"モデルをロード中: {base_model_path}")
    model, tokenizer = pipeline.load_finetuned_model(base_model_path)
    logger.info("モデルのロードが完了しました")

    # 継続学習タスクを開始
    logger.info("継続学習タスクを開始...")
    try:
        trained_model = pipeline.run_continual_task(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            train_dataset_path=args.data_path,
            epochs=args.epochs,
            use_previous_fisher=not args.reset_fisher,
            fisher_importance=args.ewc_lambda,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        logger.info("=" * 80)
        logger.info("✅ 継続学習が完了しました！")
        logger.info(f"モデル保存先: outputs/continual_{task_name}/checkpoint-final")
        logger.info("=" * 80)
        logger.info("")
        logger.info("次のステップ:")
        logger.info("1. RAGシステムでモデルを使用:")
        logger.info(f"   curl -X POST \"http://localhost:8050/rag/query\" \\")
        logger.info(f"     -H \"Content-Type: application/json\" \\")
        logger.info(f"     -d '{{\"query\": \"テストクエリ\", \"use_reasoning_model\": true}}'")
        logger.info("")
        logger.info("2. Web UIで確認:")
        logger.info("   http://localhost:8050/rag")

    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
