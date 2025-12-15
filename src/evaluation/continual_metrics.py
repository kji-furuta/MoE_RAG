# src/evaluation/continual_metrics.py
import torch
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ContinualLearningEvaluator:
    """継続学習の評価クラス"""
    
    def __init__(self, output_dir: str = "outputs/ewc_data/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_forgetting(self, model, task_history: List[Dict]) -> Dict:
        """破滅的忘却の評価
        
        Args:
            model: 評価対象のモデル
            task_history: タスク履歴のリスト
            
        Returns:
            各タスクの忘却スコアを含む辞書
        """
        results = {}
        
        for task_idx, task in enumerate(task_history):
            logger.info(f"タスク {task['task_name']} の評価中...")
            
            # テストデータセットの読み込み
            test_dataset_path = task.get('test_dataset', None)
            if not test_dataset_path:
                # テストデータがない場合は、訓練データの一部を使用
                test_dataset_path = task['dataset'].replace('.jsonl', '_test.jsonl')
                if not Path(test_dataset_path).exists():
                    logger.warning(f"テストデータが見つかりません: {test_dataset_path}")
                    continue
            
            # パフォーマンスの評価
            current_metrics = self._evaluate_task_performance(
                model, test_dataset_path
            )
            
            # 元のパフォーマンスとの比較
            original_metrics = task.get('original_metrics', {})
            forgetting_score = self._compute_forgetting_score(
                current_metrics, original_metrics
            )
            
            results[task['task_name']] = {
                'task_index': task_idx,
                'current_performance': current_metrics,
                'original_performance': original_metrics,
                'forgetting_score': forgetting_score,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def _evaluate_task_performance(self, model, dataset_path: str) -> Dict:
        """タスクのパフォーマンスを評価

        Args:
            model: 評価対象のモデル
            dataset_path: データセットのパス

        Returns:
            評価メトリクス
        """
        try:
            # データセットの読み込み (JSON配列形式とJSONL形式の両方に対応)
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                # ファイルの最初の文字を確認
                first_char = f.read(1)
                f.seek(0)  # ファイルポインタを先頭に戻す

                if first_char == '[':
                    # JSON配列形式 (例: [{"key": "value"}, {...}])
                    try:
                        data = json.load(f)
                        logger.info(f"Loaded {len(data)} samples from JSON array format")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON array: {e}")
                        return {'error': f'Invalid JSON array format: {str(e)}'}
                else:
                    # JSONL形式 (各行が独立したJSONオブジェクト)
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 空行をスキップ
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue
                    logger.info(f"Loaded {len(data)} samples from JSONL format")

            if not data:
                return {'error': 'No data found'}
            
            # サンプル評価（最大100件）
            eval_samples = data[:100]
            total_loss = 0
            correct_predictions = 0
            
            model.eval()
            with torch.no_grad():
                for sample in eval_samples:
                    # 入力テキストと期待される出力を取得
                    if 'instruction' in sample and 'output' in sample:
                        input_text = sample['instruction']
                        expected_output = sample['output']
                    elif 'text' in sample:
                        # テキストを分割して評価
                        parts = sample['text'].split('\n\n')
                        if len(parts) >= 2:
                            input_text = parts[0]
                            expected_output = parts[1]
                        else:
                            continue
                    else:
                        continue
                    
                    # モデルの予測（簡易評価）
                    # 実際の実装では、より詳細な評価が必要
                    loss_value = np.random.uniform(0.5, 2.0)  # プレースホルダー
                    total_loss += loss_value
                    
                    # 正解率の計算（簡易版）
                    if np.random.random() > 0.3:  # プレースホルダー
                        correct_predictions += 1
            
            num_samples = len(eval_samples)
            metrics = {
                'average_loss': total_loss / num_samples if num_samples > 0 else 0,
                'accuracy': correct_predictions / num_samples if num_samples > 0 else 0,
                'perplexity': np.exp(total_loss / num_samples) if num_samples > 0 else 0,
                'num_samples': num_samples
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"評価エラー: {str(e)}")
            return {'error': str(e)}
    
    def _compute_forgetting_score(
        self,
        current_metrics: Dict,
        original_metrics: Dict
    ) -> float:
        """忘却スコアの計算
        
        Args:
            current_metrics: 現在のメトリクス
            original_metrics: 元のメトリクス
            
        Returns:
            忘却スコア（0-1の範囲、0が忘却なし）
        """
        if not original_metrics or 'accuracy' not in original_metrics:
            return 0.0
        
        if 'error' in current_metrics or 'error' in original_metrics:
            return 1.0
        
        # 精度の低下を忘却スコアとする
        original_acc = original_metrics.get('accuracy', 0)
        current_acc = current_metrics.get('accuracy', 0)
        
        if original_acc == 0:
            return 0.0
        
        # 忘却スコア = max(0, (元の精度 - 現在の精度) / 元の精度)
        forgetting = max(0, (original_acc - current_acc) / original_acc)
        
        return min(1.0, forgetting)
    
    def generate_report(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> Path:
        """評価レポートの生成

        Args:
            results: 評価結果
            save_path: 保存パス（省略時は自動生成）

        Returns:
            レポート画像のパス
        """
        if not save_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"forgetting_analysis_{timestamp}.png"
        else:
            save_path = Path(save_path)

        # resultsが空の場合の対策
        if not results:
            logger.warning("評価結果が空です。デフォルトのレポートを生成します。")
            # 空のレポートを生成
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'No evaluation data available',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=20)
            ax.axis('off')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            return save_path

        # 図のサイズ設定
        plt.figure(figsize=(12, 8))

        # データの準備
        tasks = []
        forgetting_scores = []
        accuracies_original = []
        accuracies_current = []

        for task_name, data in results.items():
            tasks.append(task_name)
            # forgetting_scoreが存在しない場合のデフォルト値
            forgetting_scores.append(data.get('forgetting_score', 0.0))
            
            orig_metrics = data.get('original_performance', {})
            curr_metrics = data.get('current_performance', {})

            accuracies_original.append(orig_metrics.get('accuracy', 0))
            accuracies_current.append(curr_metrics.get('accuracy', 0))

        # 空のリストの場合の対策
        if not tasks:
            logger.warning("タスクリストが空です。デフォルトのレポートを生成します。")
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'No tasks to evaluate',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=20)
            ax.axis('off')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            return save_path
        
        # サブプロット作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 忘却スコアのバーグラフ
        ax1 = axes[0, 0]
        bars = ax1.bar(tasks, forgetting_scores, color='coral')
        ax1.set_xlabel('タスク', fontsize=12)
        ax1.set_ylabel('忘却スコア', fontsize=12)
        ax1.set_title('タスクごとの破滅的忘却スコア', fontsize=14)
        ax1.set_ylim(0, 1.1)
        
        # バーの上に値を表示
        for bar, score in zip(bars, forgetting_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. 精度の比較
        ax2 = axes[0, 1]
        x = np.arange(len(tasks))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, accuracies_original, width,
                       label='元の精度', color='skyblue')
        bars2 = ax2.bar(x + width/2, accuracies_current, width,
                       label='現在の精度', color='lightgreen')
        
        ax2.set_xlabel('タスク', fontsize=12)
        ax2.set_ylabel('精度', fontsize=12)
        ax2.set_title('タスクごとの精度比較', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        # 3. 忘却スコアの推移
        ax3 = axes[1, 0]
        task_indices = list(range(len(tasks)))
        ax3.plot(task_indices, forgetting_scores, 'o-', linewidth=2,
                markersize=8, color='darkred')
        ax3.set_xlabel('タスクインデックス', fontsize=12)
        ax3.set_ylabel('忘却スコア', fontsize=12)
        ax3.set_title('忘却スコアの推移', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. サマリー統計
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 統計情報の計算（空配列対策）
        if len(forgetting_scores) > 0:
            avg_forgetting = np.mean(forgetting_scores)
            max_forgetting = np.max(forgetting_scores)
            min_forgetting = np.min(forgetting_scores)
        else:
            avg_forgetting = 0.0
            max_forgetting = 0.0
            min_forgetting = 0.0
        
        summary_text = f"""
        評価サマリー
        
        平均忘却スコア: {avg_forgetting:.3f}
        最大忘却スコア: {max_forgetting:.3f}
        最小忘却スコア: {min_forgetting:.3f}
        
        評価タスク数: {len(tasks)}
        評価日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12,
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.5))
        
        # 全体のタイトル
        fig.suptitle('継続学習 - 破滅的忘却分析レポート', fontsize=16)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"評価レポートを保存しました: {save_path}")
        
        # JSON形式でも結果を保存
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return save_path
    
    def plot_learning_curves(
        self,
        task_history: List[Dict],
        save_path: Optional[str] = None
    ) -> Path:
        """学習曲線のプロット
        
        Args:
            task_history: タスク履歴
            save_path: 保存パス
            
        Returns:
            画像のパス
        """
        if not save_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"learning_curves_{timestamp}.png"
        else:
            save_path = Path(save_path)
        
        plt.figure(figsize=(10, 6))
        
        # 各タスクの学習曲線をプロット（プレースホルダー）
        for idx, task in enumerate(task_history):
            epochs = list(range(1, task.get('epochs', 3) + 1))
            # 実際の実装では、学習中のロスデータを使用
            losses = [2.0 - 0.3 * e + 0.1 * np.random.randn() for e in epochs]
            
            plt.plot(epochs, losses, 'o-', label=f"{task['task_name']}")
        
        plt.xlabel('エポック', fontsize=12)
        plt.ylabel('損失', fontsize=12)
        plt.title('タスクごとの学習曲線', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
