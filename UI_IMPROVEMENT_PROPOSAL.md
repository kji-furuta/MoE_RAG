# MoE_RAG システム UI改善提案書

## 📋 概要
MoE_RAGシステムに統合管理インターフェースを追加し、モデル、データ、ログの一元管理を実現します。

## 🎯 改善目標

1. **モデル管理**: 生成モデルの確認、削除、比較機能
2. **データ管理**: 学習データ、RAGドキュメントの管理
3. **ログ管理**: 学習ログ、システムログの閲覧・削除
4. **ダッシュボード**: システム全体の状態可視化
5. **メンテナンス**: クリーンアップ、バックアップ機能

## 🏗️ 提案するUI構造

### 1. 新規追加ページ

#### A. 統合管理ダッシュボード (`/admin`)
```html
<!-- メインダッシュボード -->
<div class="admin-dashboard">
  <!-- システムステータスカード -->
  <div class="status-cards">
    <div class="card">
      <h3>モデル数</h3>
      <p class="metric">15</p>
      <span class="sub">LoRA: 12, Full: 3</span>
    </div>
    <div class="card">
      <h3>ストレージ使用量</h3>
      <p class="metric">234 GB</p>
      <div class="progress-bar"></div>
    </div>
    <div class="card">
      <h3>アクティブタスク</h3>
      <p class="metric">2</p>
      <span class="sub">学習: 1, RAG: 1</span>
    </div>
    <div class="card">
      <h3>システム状態</h3>
      <p class="metric status-ok">正常</p>
      <span class="sub">GPU: 45%, MEM: 60%</span>
    </div>
  </div>

  <!-- クイックアクション -->
  <div class="quick-actions">
    <button class="btn-action" onclick="cleanupOldModels()">
      <i class="fas fa-broom"></i> 古いモデルの削除
    </button>
    <button class="btn-action" onclick="optimizeStorage()">
      <i class="fas fa-compress"></i> ストレージ最適化
    </button>
    <button class="btn-action" onclick="exportLogs()">
      <i class="fas fa-download"></i> ログエクスポート
    </button>
    <button class="btn-action" onclick="systemBackup()">
      <i class="fas fa-save"></i> バックアップ作成
    </button>
  </div>

  <!-- 最近のアクティビティ -->
  <div class="recent-activity">
    <h3>最近のアクティビティ</h3>
    <div class="activity-list">
      <!-- 動的に生成 -->
    </div>
  </div>
</div>
```

#### B. モデル管理ページ (`/admin/models`)
```javascript
// モデル管理インターフェース
const ModelManager = {
  // モデルリスト表示
  displayModels() {
    return {
      columns: [
        { key: 'name', label: 'モデル名', sortable: true },
        { key: 'type', label: 'タイプ', filter: ['LoRA', 'Full', 'Continual'] },
        { key: 'size', label: 'サイズ', sortable: true },
        { key: 'created', label: '作成日時', sortable: true },
        { key: 'performance', label: '性能指標' },
        { key: 'actions', label: 'アクション' }
      ],
      actions: [
        { icon: 'eye', action: 'view', tooltip: '詳細表示' },
        { icon: 'download', action: 'export', tooltip: 'エクスポート' },
        { icon: 'chart', action: 'compare', tooltip: '比較' },
        { icon: 'trash', action: 'delete', tooltip: '削除', confirm: true }
      ]
    };
  },

  // モデル詳細表示
  viewModelDetails(modelId) {
    return {
      info: {
        base_model: 'cyberagent/DeepSeek-R1-Distill-Qwen-32B',
        training_data: 'civil_engineering_v2.json',
        parameters: { rank: 32, alpha: 64, dropout: 0.1 },
        metrics: { loss: 0.23, perplexity: 1.45 }
      },
      visualizations: {
        loss_curve: 'chart_data',
        attention_heatmap: 'heatmap_data'
      }
    };
  }
};
```

#### C. データ管理ページ (`/admin/data`)
```javascript
// データ管理インターフェース
const DataManager = {
  sections: {
    training_data: {
      title: '学習データセット',
      actions: ['upload', 'view', 'validate', 'delete'],
      stats: {
        total_samples: 5000,
        categories: ['道路設計', '橋梁', 'トンネル'],
        quality_score: 0.92
      }
    },
    rag_documents: {
      title: 'RAGドキュメント',
      actions: ['upload', 'index', 'search', 'delete'],
      stats: {
        total_docs: 150,
        indexed_chunks: 12500,
        vector_dim: 1024
      }
    },
    embeddings: {
      title: '埋め込みデータ',
      actions: ['regenerate', 'optimize', 'export', 'delete'],
      stats: {
        total_vectors: 12500,
        storage_size: '4.5 GB',
        index_type: 'HNSW'
      }
    }
  }
};
```

#### D. ログビューワー (`/admin/logs`)
```javascript
// ログ管理インターフェース
const LogViewer = {
  // ログフィルター
  filters: {
    level: ['ERROR', 'WARNING', 'INFO', 'DEBUG'],
    source: ['training', 'rag', 'api', 'system'],
    timeRange: ['1h', '24h', '7d', '30d', 'custom']
  },

  // リアルタイムログ表示
  streamLogs(filter) {
    const ws = new WebSocket('ws://localhost:8050/admin/logs/stream');
    ws.onmessage = (event) => {
      const log = JSON.parse(event.data);
      this.appendLog(log);
    };
  },

  // ログ分析
  analyzeLogs() {
    return {
      error_frequency: 'chart_data',
      performance_metrics: 'metrics_data',
      resource_usage: 'usage_data'
    };
  }
};
```

### 2. 統合ナビゲーション改善

```html
<!-- 改善されたナビゲーションバー -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">
      <img src="/static/logo_teikoku.png" alt="Logo" height="40">
      MoE_RAG System
    </a>

    <!-- メインナビゲーション -->
    <div class="navbar-nav me-auto">
      <a class="nav-link" href="/finetune">
        <i class="fas fa-brain"></i> ファインチューニング
      </a>
      <a class="nav-link" href="/continual">
        <i class="fas fa-graduation-cap"></i> 継続学習
      </a>
      <a class="nav-link" href="/rag">
        <i class="fas fa-search"></i> RAG検索
      </a>
      <div class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" href="#" data-bs-toggle="dropdown">
          <i class="fas fa-cog"></i> 管理
        </a>
        <ul class="dropdown-menu">
          <li><a class="dropdown-item" href="/admin">
            <i class="fas fa-tachometer-alt"></i> ダッシュボード
          </a></li>
          <li><a class="dropdown-item" href="/admin/models">
            <i class="fas fa-cube"></i> モデル管理
          </a></li>
          <li><a class="dropdown-item" href="/admin/data">
            <i class="fas fa-database"></i> データ管理
          </a></li>
          <li><a class="dropdown-item" href="/admin/logs">
            <i class="fas fa-file-alt"></i> ログビューワー
          </a></li>
          <li><hr class="dropdown-divider"></li>
          <li><a class="dropdown-item" href="/admin/settings">
            <i class="fas fa-sliders-h"></i> システム設定
          </a></li>
        </ul>
      </div>
    </div>

    <!-- ステータスインジケーター -->
    <div class="navbar-nav">
      <span class="navbar-text me-3">
        <i class="fas fa-server"></i> GPU: <span id="gpu-usage">45%</span>
      </span>
      <span class="navbar-text me-3">
        <i class="fas fa-memory"></i> MEM: <span id="mem-usage">60%</span>
      </span>
      <span class="navbar-text">
        <span class="badge bg-success">システム正常</span>
      </span>
    </div>
  </div>
</nav>
```

### 3. API エンドポイント追加

```python
# app/routers/admin.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict
import shutil
import os
from pathlib import Path

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """ダッシュボード統計情報を取得"""
    return {
        "models": {
            "total": count_models(),
            "lora": count_lora_models(),
            "full": count_full_models(),
            "continual": count_continual_models()
        },
        "storage": {
            "used": get_storage_usage(),
            "total": get_total_storage(),
            "percentage": calculate_storage_percentage()
        },
        "system": {
            "gpu_usage": get_gpu_usage(),
            "memory_usage": get_memory_usage(),
            "active_tasks": get_active_tasks()
        }
    }

@router.get("/models")
async def list_models(
    model_type: Optional[str] = Query(None),
    sort_by: str = Query("created", regex="^(name|size|created|performance)$"),
    order: str = Query("desc", regex="^(asc|desc)$")
):
    """モデル一覧を取得"""
    models = scan_all_models()
    if model_type:
        models = filter_by_type(models, model_type)
    return sort_models(models, sort_by, order)

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """モデルを削除"""
    model_path = get_model_path(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # バックアップを作成
    backup_path = create_backup(model_path)

    try:
        shutil.rmtree(model_path)
        return {"status": "deleted", "backup": str(backup_path)}
    except Exception as e:
        restore_from_backup(backup_path, model_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/compare")
async def compare_models(model_ids: List[str]):
    """複数モデルを比較"""
    comparison = {}
    for model_id in model_ids:
        model_info = get_model_info(model_id)
        metrics = get_model_metrics(model_id)
        comparison[model_id] = {
            "info": model_info,
            "metrics": metrics,
            "performance": calculate_performance_score(metrics)
        }
    return comparison

@router.get("/data/summary")
async def get_data_summary():
    """データサマリーを取得"""
    return {
        "training_data": analyze_training_data(),
        "rag_documents": analyze_rag_documents(),
        "embeddings": analyze_embeddings(),
        "storage_distribution": calculate_storage_distribution()
    }

@router.delete("/data/cleanup")
async def cleanup_old_data(days: int = Query(30, ge=7)):
    """古いデータをクリーンアップ"""
    deleted = {
        "models": cleanup_old_models(days),
        "logs": cleanup_old_logs(days),
        "temp_files": cleanup_temp_files()
    }
    return {
        "deleted": deleted,
        "freed_space": calculate_freed_space(deleted)
    }

@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0)
):
    """ログを取得"""
    logs = read_system_logs()
    if level:
        logs = filter_by_level(logs, level)
    if source:
        logs = filter_by_source(logs, source)
    return paginate_logs(logs, limit, offset)

@router.websocket("/logs/stream")
async def stream_logs(websocket: WebSocket):
    """ログをリアルタイムストリーミング"""
    await websocket.accept()
    try:
        async for log in tail_logs():
            await websocket.send_json(log)
    except Exception as e:
        await websocket.close()

@router.post("/backup")
async def create_system_backup():
    """システムバックアップを作成"""
    backup_path = create_full_backup()
    return {
        "backup_path": str(backup_path),
        "size": get_file_size(backup_path),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/optimize")
async def optimize_system():
    """システムを最適化"""
    optimization_results = {
        "models_compressed": compress_models(),
        "indices_rebuilt": rebuild_indices(),
        "cache_cleared": clear_caches(),
        "logs_rotated": rotate_logs()
    }
    return optimization_results
```

### 4. フロントエンド実装

```javascript
// app/static/admin/admin.js
class AdminDashboard {
    constructor() {
        this.refreshInterval = 5000; // 5秒ごとに更新
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadDashboardStats();
        this.initCharts();
        this.setupAutoRefresh();
        this.setupEventListeners();
    }

    async loadDashboardStats() {
        try {
            const response = await fetch('/admin/dashboard/stats');
            const stats = await response.json();
            this.updateDashboard(stats);
        } catch (error) {
            console.error('Failed to load dashboard stats:', error);
            this.showError('ダッシュボードの読み込みに失敗しました');
        }
    }

    updateDashboard(stats) {
        // モデル数の更新
        document.getElementById('model-count').textContent = stats.models.total;
        document.getElementById('model-details').textContent =
            `LoRA: ${stats.models.lora}, Full: ${stats.models.full}`;

        // ストレージ使用量の更新
        const storagePercent = stats.storage.percentage;
        document.getElementById('storage-usage').textContent =
            `${stats.storage.used} / ${stats.storage.total} GB`;
        document.getElementById('storage-bar').style.width = `${storagePercent}%`;

        // システムステータスの更新
        document.getElementById('gpu-usage').textContent = `${stats.system.gpu_usage}%`;
        document.getElementById('memory-usage').textContent = `${stats.system.memory_usage}%`;

        this.updateSystemStatus(stats.system);
    }

    initCharts() {
        // GPU使用率チャート
        this.charts.gpu = new Chart(document.getElementById('gpu-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU使用率',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        // メモリ使用率チャート
        this.charts.memory = new Chart(document.getElementById('memory-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'メモリ使用率',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    async deleteModel(modelId) {
        if (!confirm(`モデル ${modelId} を削除してもよろしいですか？`)) {
            return;
        }

        try {
            const response = await fetch(`/admin/models/${modelId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showSuccess('モデルが削除されました');
                await this.refreshModelList();
            } else {
                throw new Error('削除に失敗しました');
            }
        } catch (error) {
            this.showError(`エラー: ${error.message}`);
        }
    }

    async compareModels(modelIds) {
        try {
            const response = await fetch('/admin/models/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_ids: modelIds })
            });

            const comparison = await response.json();
            this.showComparisonModal(comparison);
        } catch (error) {
            this.showError('モデル比較に失敗しました');
        }
    }

    showComparisonModal(comparison) {
        const modal = new bootstrap.Modal(document.getElementById('comparison-modal'));
        const content = this.generateComparisonTable(comparison);
        document.getElementById('comparison-content').innerHTML = content;
        modal.show();
    }

    async cleanupOldData() {
        const days = prompt('何日以前のデータを削除しますか？', '30');
        if (!days) return;

        try {
            const response = await fetch(`/admin/data/cleanup?days=${days}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            this.showSuccess(`${result.freed_space} GB のストレージが解放されました`);
            await this.loadDashboardStats();
        } catch (error) {
            this.showError('クリーンアップに失敗しました');
        }
    }

    setupAutoRefresh() {
        setInterval(() => {
            this.loadDashboardStats();
            this.updateCharts();
        }, this.refreshInterval);
    }

    showSuccess(message) {
        Toastify({
            text: message,
            backgroundColor: "linear-gradient(to right, #00b09b, #96c93d)",
            duration: 3000
        }).showToast();
    }

    showError(message) {
        Toastify({
            text: message,
            backgroundColor: "linear-gradient(to right, #ff5f6d, #ffc371)",
            duration: 5000
        }).showToast();
    }
}

// ページロード時に初期化
document.addEventListener('DOMContentLoaded', () => {
    window.adminDashboard = new AdminDashboard();
});
```

### 5. スタイル定義

```css
/* app/static/admin/admin.css */
.admin-dashboard {
    padding: 20px;
    background: #f8f9fa;
    min-height: 100vh;
}

.status-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.status-cards .card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}

.status-cards .card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
}

.status-cards .metric {
    font-size: 2.5rem;
    font-weight: bold;
    color: #333;
    margin: 10px 0;
}

.status-cards .sub {
    color: #6c757d;
    font-size: 0.9rem;
}

.quick-actions {
    display: flex;
    gap: 15px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.btn-action {
    padding: 12px 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    gap: 8px;
}

.btn-action:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.model-table {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.model-table th {
    background: #343a40;
    color: white;
    padding: 15px;
    text-align: left;
}

.model-table td {
    padding: 15px;
    border-bottom: 1px solid #dee2e6;
}

.model-actions {
    display: flex;
    gap: 10px;
}

.model-actions button {
    padding: 5px 10px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
}

.btn-view { background: #17a2b8; color: white; }
.btn-export { background: #28a745; color: white; }
.btn-compare { background: #ffc107; color: #333; }
.btn-delete { background: #dc3545; color: white; }

.progress-bar {
    width: 100%;
    height: 8px;
    background: #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-bar::after {
    content: '';
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 4px;
    transition: width 0.3s;
}

.status-ok { color: #28a745; }
.status-warning { color: #ffc107; }
.status-error { color: #dc3545; }

/* ログビューワー */
.log-viewer {
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Courier New', monospace;
    padding: 20px;
    border-radius: 10px;
    height: 600px;
    overflow-y: auto;
}

.log-entry {
    padding: 5px 10px;
    border-left: 3px solid transparent;
    margin-bottom: 2px;
}

.log-error { border-left-color: #f44336; background: rgba(244, 67, 54, 0.1); }
.log-warning { border-left-color: #ff9800; background: rgba(255, 152, 0, 0.1); }
.log-info { border-left-color: #2196f3; background: rgba(33, 150, 243, 0.1); }
.log-debug { border-left-color: #9e9e9e; background: rgba(158, 158, 158, 0.1); }

/* レスポンシブ対応 */
@media (max-width: 768px) {
    .status-cards {
        grid-template-columns: 1fr;
    }

    .quick-actions {
        flex-direction: column;
    }

    .btn-action {
        width: 100%;
        justify-content: center;
    }
}
```

## 🚀 実装計画

### フェーズ1: 基盤構築（1週間）
1. 管理用APIエンドポイントの実装
2. データベーススキーマの拡張
3. 認証・権限管理の実装

### フェーズ2: UI実装（1週間）
1. 管理ダッシュボードの実装
2. モデル管理インターフェースの実装
3. データ管理インターフェースの実装

### フェーズ3: 高度な機能（1週間）
1. リアルタイムログストリーミング
2. モデル比較・分析機能
3. 自動最適化機能

### フェーズ4: テスト・最適化（3日）
1. 統合テスト
2. パフォーマンス最適化
3. ドキュメント作成

## 📈 期待される効果

1. **運用効率**: 管理作業時間を50%削減
2. **可視性向上**: システム状態の即時把握
3. **保守性向上**: 問題の早期発見と対処
4. **ストレージ最適化**: 不要データの自動削除で30%容量削減
5. **ユーザビリティ**: 直感的な操作で学習コスト削減

## 🔒 セキュリティ考慮事項

1. **アクセス制御**: 管理機能への認証必須化
2. **操作ログ**: 全ての管理操作を記録
3. **バックアップ**: 削除前の自動バックアップ
4. **確認ダイアログ**: 破壊的操作の二重確認

## 📊 成功指標

- モデル管理操作の完了時間: 現在の1/3以下
- システムエラーの検出時間: 5分以内
- ストレージ使用率: 20%以上の削減
- ユーザー満足度: 90%以上

---

この提案により、MoE_RAGシステムの管理性と保守性が大幅に向上し、
効率的なMLOps環境を実現できます。