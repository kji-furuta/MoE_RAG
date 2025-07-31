// AI Fine-tuning Toolkit JavaScript

// グローバル変数
let currentTrainingTask = null;
let systemInfo = {};
let availableModels = [];
let savedModels = [];

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupNavigation();
    loadSystemInfo();
    loadModels().then(() => {
        // 初期読み込み時にベースモデル選択肢も設定
        loadBaseModelsForTraining();
    }).catch(error => {
        console.error('Failed to load models:', error);
        // エラーが発生してもベースモデル選択肢は設定
        loadBaseModelsForTraining();
    });
});

// アプリケーション初期化
function initializeApp() {
    console.log('AI Fine-tuning Toolkit Web Interface starting...');
    
    // ページ読み込み時にダッシュボードを表示
    showSection('dashboard');
    
    // 定期的にシステム情報を更新
    setInterval(loadSystemInfo, 30000); // 30秒毎
    
    // トレーニングタスクのチェック
    if (currentTrainingTask) {
        checkTrainingStatus();
        setInterval(checkTrainingStatus, 5000); // 5秒毎
    }
}

// ナビゲーション設定
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // アクティブクラスの管理
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // セクション表示
            const target = this.getAttribute('href').substring(1);
            showSection(target);
        });
    });
}

// セクション表示
function showSection(sectionId) {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.style.display = 'block';
        targetSection.classList.add('fade-in');
    }
    
    // セクション固有の初期化
    if (sectionId === 'models') {
        loadModels();
    } else if (sectionId === 'inference') {
        // モデル情報を再読み込みしてから選択肢を更新
        loadModels().then(() => {
            loadSavedModelsForInference();
        });
    } else if (sectionId === 'training') {
        // トレーニングセクションでベースモデル選択肢を更新
        loadModels().then(() => {
            loadBaseModelsForTraining();
        });
    }
}

// システム情報取得
async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system-info');
        const data = await response.json();
        systemInfo = data;
        displaySystemInfo(data);
    } catch (error) {
        console.error('System info load error:', error);
        document.getElementById('system-info').innerHTML = 
            '<div class="alert alert-danger">システム情報の取得に失敗しました</div>';
    }
}

// システム情報表示
function displaySystemInfo(info) {
    const container = document.getElementById('system-info');
    
    let html = `
        <div class="system-info-item">
            <span><i class="bi bi-cpu"></i> CPU:</span>
            <span>${info.cpu_count} cores</span>
        </div>
        <div class="system-info-item">
            <span><i class="bi bi-memory"></i> RAM:</span>
            <span>${info.memory_used}GB / ${info.memory_total}GB</span>
        </div>
    `;
    
    if (info.gpu_count > 0) {
        html += `
            <div class="system-info-item">
                <span><i class="bi bi-gpu-card"></i> GPU:</span>
                <span>${info.gpu_count} devices</span>
            </div>
        `;
        
        info.gpu_info.forEach((gpu, index) => {
            html += `
                <div class="gpu-info">
                    <small><strong>GPU ${index}:</strong> ${gpu.name}</small><br>
                    <small>VRAM: ${gpu.memory_used}GB / ${gpu.memory_total}GB</small>
                </div>
            `;
        });
    } else {
        html += `
            <div class="alert alert-warning alert-sm">
                <small>GPU が検出されませんでした</small>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// モデル情報取得
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        availableModels = data.available_models;
        savedModels = data.saved_models;
        displayModels(data);
        return data;
    } catch (error) {
        console.error('Models load error:', error);
        return null;
    }
}

// モデル表示
function displayModels(data) {
    // 利用可能なモデル
    const availableContainer = document.getElementById('available-models');
    let availableHtml = '';
    
    data.available_models.forEach(model => {
        // ステータスバッジの色を判定
        let statusBadgeClass = 'bg-success';
        let statusText = model.status;
        if (model.status === 'requires_auth') {
            statusBadgeClass = 'bg-warning';
            statusText = '認証必要';
        }
        
        // 追加情報の構築
        let additionalInfo = '';
        if (model.recommended) {
            additionalInfo += '<span class="badge bg-success me-1">🌟 推奨</span>';
        }
        if (model.gpu_required) {
            additionalInfo += `<span class="badge bg-danger me-1">GPU: ${model.gpu_required}</span>`;
        }
        if (model.warning) {
            additionalInfo += `<span class="badge bg-warning text-dark me-1">⚠️ ${model.warning}</span>`;
        }
        if (model.test_only) {
            additionalInfo += '<span class="badge bg-secondary me-1">🧪 テスト用</span>';
        }
        
        availableHtml += `
            <div class="model-card card mb-3">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <h6 class="card-title">${model.name}</h6>
                            <p class="card-text text-muted">${model.description}</p>
                            ${additionalInfo ? `<div class="mt-2">${additionalInfo}</div>` : ''}
                        </div>
                        <div class="col-md-4 text-end">
                            <span class="badge bg-primary">${model.size}</span>
                            <span class="badge ${statusBadgeClass} status-badge">${statusText}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    availableContainer.innerHTML = availableHtml || '<p class="text-muted">利用可能なモデルがありません</p>';
    
    // 保存済みモデル
    const savedContainer = document.getElementById('saved-models');
    let savedHtml = '';
    
    data.saved_models.forEach(model => {
        savedHtml += `
            <div class="model-card card mb-3">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <h6 class="card-title">${model.name}</h6>
                            <p class="card-text text-muted">パス: ${model.path}</p>
                        </div>
                        <div class="col-md-4 text-end">
                            <span class="badge bg-info">${model.type}</span>
                            <span class="badge bg-secondary">${model.size}</span>
                            <br><br>
                            <button class="btn btn-sm btn-primary" onclick="useModel('${model.path}')">
                                <i class="bi bi-play"></i> 使用
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    savedContainer.innerHTML = savedHtml || '<p class="text-muted">保存済みモデルがありません</p>';
}

// 推論用モデル選択肢読み込み
function loadSavedModelsForInference() {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '<option value="">モデルを選択してください</option>';
    
    savedModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = `${model.name} (${model.type})`;
        select.appendChild(option);
    });
}

// トレーニング用ベースモデル選択肢読み込み
function loadBaseModelsForTraining() {
    const select = document.getElementById('baseModel');
    if (!select) return; // 要素が存在しない場合は終了
    
    select.innerHTML = '<option value="">モデルを選択してください</option>';
    
    if (availableModels && availableModels.length > 0) {
        // カテゴリ別にグループ化
        const groups = {
            '🌟 推奨モデル': [],
            '💪 小・中規模 (1B-7B)': [],
            '🚀 中・大規模 (8B-22B)': [],
            '🔥 超大規模 (32B+)': [],
            '🧪 テスト用': []
        };
        
        availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            
            // オプションテキストの構築
            let optionText = `${model.name} (${model.size})`;
            if (model.gpu_required) {
                optionText += ` - ${model.gpu_required}以上推奨`;
            }
            option.textContent = optionText;
            
            // 追加情報をdata属性として保存
            if (model.warning) {
                option.setAttribute('data-warning', model.warning);
            }
            
            // カテゴリ別に分類
            if (model.test_only) {
                groups['🧪 テスト用'].push(option);
            } else if (model.recommended) {
                groups['🌟 推奨モデル'].push(option);
            } else if (model.size.includes('32B') || model.size.includes('70B')) {
                groups['🔥 超大規模 (32B+)'].push(option);
            } else if (model.size.includes('22B') || model.size.includes('17B') || model.size.includes('13B') || model.size.includes('8B')) {
                groups['🚀 中・大規模 (8B-22B)'].push(option);
            } else {
                groups['💪 小・中規模 (1B-7B)'].push(option);
            }
        });
        
        // グループごとにオプションを追加
        Object.entries(groups).forEach(([groupName, options]) => {
            if (options.length > 0) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = groupName;
                options.forEach(option => optgroup.appendChild(option));
                select.appendChild(optgroup);
            }
        });
        
        // モデル選択時の警告表示
        select.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            const warning = selectedOption.getAttribute('data-warning');
            const warningDiv = document.getElementById('model-warning');
            
            if (warning && warningDiv) {
                warningDiv.innerHTML = `<div class="alert alert-warning">⚠️ ${warning}</div>`;
                warningDiv.style.display = 'block';
            } else if (warningDiv) {
                warningDiv.style.display = 'none';
            }
        });
    } else {
        // フォールバック：利用できない場合はデフォルトオプションを追加
        console.log('Available models not loaded yet');
        const defaultOptions = [
            {name: 'distilgpt2', description: '軽量な英語モデル（テスト用）', size: '82MB'},
            {name: 'stabilityai/japanese-stablelm-3b-4e1t-instruct', description: 'Japanese StableLM 3B Instruct', size: '3B'}
        ];
        
        defaultOptions.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = `${model.name} - ${model.description} (${model.size})`;
            select.appendChild(option);
        });
    }
}

// ファイルアップロード
async function uploadFile() {
    const fileInput = document.getElementById('trainingFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('upload-status', 'ファイルを選択してください', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showAlert('upload-status', 'アップロード中...', 'info');
    
    try {
        const response = await fetch('/api/upload-data', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            let message = `アップロード完了: ${result.filename} (${(result.size / 1024).toFixed(1)}KB)`;
            if (result.sample_data && result.sample_data.length > 0) {
                message += `<br><strong>サンプルデータ:</strong><pre>${JSON.stringify(result.sample_data[0], null, 2)}</pre>`;
            }
            showAlert('upload-status', message, 'success');
        } else {
            showAlert('upload-status', `エラー: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showAlert('upload-status', 'アップロードに失敗しました', 'danger');
    }
}

// トレーニング方法に応じたオプションの更新
function updateTrainingOptions() {
    const method = document.getElementById('trainingMethod').value;
    const loraRankInput = document.getElementById('loraRank');
    const learningRateInput = document.getElementById('learningRate');
    
    if (method === 'full') {
        loraRankInput.disabled = true;
        learningRateInput.value = '0.00005'; // フルファインチューニング用の低い学習率
    } else {
        loraRankInput.disabled = false;
        learningRateInput.value = '0.0003'; // LoRA用の学習率
    }
}

// ファインチューニング開始
async function startTraining() {
    const baseModel = document.getElementById('baseModel').value;
    const trainingMethod = document.getElementById('trainingMethod').value;
    const loraRank = parseInt(document.getElementById('loraRank').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    
    // サンプルデータ（実際の実装では、アップロードされたデータを使用）
    const sampleData = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さは何メートルですか？\n回答: 富士山の高さは3,776メートルです。",
        "質問: 桜の季節はいつ頃ですか？\n回答: 桜の開花時期は地域により異なりますが、一般的に3月下旬から5月上旬にかけてです。"
    ];
    
    const requestData = {
        model_name: baseModel,
        training_data: sampleData,
        training_method: trainingMethod,
        lora_config: {
            r: loraRank,
            lora_alpha: loraRank * 2,
            target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout: 0.05,
            use_qlora: trainingMethod === 'qlora'
        },
        training_config: {
            learning_rate: learningRate,
            batch_size: batchSize,
            num_epochs: epochs,
            gradient_accumulation_steps: 4,
            output_dir: `/workspace/web_training_${Date.now()}`,
            fp16: trainingMethod !== 'full',
            load_in_8bit: trainingMethod === 'qlora'
        }
    };
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentTrainingTask = result.task_id;
            showAlert('training-status', 'ファインチューニングを開始しました...', 'info');
            
            // ステータス監視開始
            checkTrainingStatus();
            const statusInterval = setInterval(() => {
                checkTrainingStatus().then(status => {
                    if (status && (status.status === 'completed' || status.status === 'failed')) {
                        clearInterval(statusInterval);
                        currentTrainingTask = null; // タスクを完了にセット
                        if (status.status === 'completed') {
                            loadModels().then(() => {
                                // 現在テキスト生成タブにいる場合は選択肢も更新
                                const inferenceSection = document.getElementById('inference');
                                if (inferenceSection && inferenceSection.style.display !== 'none') {
                                    loadSavedModelsForInference();
                                }
                            });
                        }
                    }
                });
            }, 5000);
        } else {
            showAlert('training-status', `エラー: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Training start error:', error);
        showAlert('training-status', 'ファインチューニングの開始に失敗しました', 'danger');
    }
}

// トレーニングステータス確認
async function checkTrainingStatus() {
    if (!currentTrainingTask) return;
    
    try {
        const response = await fetch(`/api/training-status/${currentTrainingTask}`);
        const status = await response.json();
        
        if (response.ok) {
            displayTrainingStatus(status);
            return status;
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
    return null;
}

// トレーニングステータス表示
function displayTrainingStatus(status) {
    const container = document.getElementById('training-status');
    
    let alertType = 'info';
    if (status.status === 'completed') alertType = 'success';
    else if (status.status === 'failed') alertType = 'danger';
    
    let html = `
        <div class="alert alert-${alertType}">
            <div class="d-flex justify-content-between align-items-center">
                <span><strong>ステータス:</strong> ${status.message}</span>
                <span><strong>進捗:</strong> ${status.progress.toFixed(1)}%</span>
            </div>
            <div class="progress mt-2">
                <div class="progress-bar progress-bar-custom" style="width: ${status.progress}%"></div>
            </div>
        </div>
    `;
    
    if (status.model_path) {
        html += `
            <div class="alert alert-success">
                <strong>完了!</strong> モデルが保存されました: ${status.model_path}
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // 実行中のタスク表示更新
    updateRunningTasks(status);
}

// 実行中タスク表示更新
function updateRunningTasks(status) {
    const container = document.getElementById('running-tasks');
    
    if (status.status === 'completed' || status.status === 'failed') {
        container.innerHTML = '<p class="text-muted">実行中のタスクはありません</p>';
        currentTrainingTask = null;
    } else {
        container.innerHTML = `
            <div class="task-item ${status.status}">
                <small><strong>ファインチューニング</strong></small><br>
                <small>${status.message}</small><br>
                <div class="progress mt-1" style="height: 0.5rem;">
                    <div class="progress-bar" style="width: ${status.progress}%"></div>
                </div>
            </div>
        `;
    }
}

// テキスト生成
async function generateText() {
    const modelPath = document.getElementById('modelSelect').value;
    const prompt = document.getElementById('promptText').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxLength = parseInt(document.getElementById('maxLength').value);
    
    if (!modelPath) {
        showAlert('generation-result', 'モデルを選択してください', 'warning');
        return;
    }
    
    if (!prompt.trim()) {
        showAlert('generation-result', 'プロンプトを入力してください', 'warning');
        return;
    }
    
    showAlert('generation-result', 'テキストを生成中...', 'info');
    
    const requestData = {
        model_path: modelPath,
        prompt: prompt,
        temperature: temperature,
        max_length: maxLength
    };
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayGenerationResult(result);
        } else {
            showAlert('generation-result', `エラー: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Generation error:', error);
        showAlert('generation-result', 'テキスト生成に失敗しました', 'danger');
    }
}

// 生成結果表示
function displayGenerationResult(result) {
    const container = document.getElementById('generation-result');
    
    const html = `
        <div class="alert alert-success">
            <strong>生成完了!</strong>
        </div>
        <div class="generation-output">
            ${result.generated_text}
        </div>
        <small class="text-muted">モデル: ${result.model_path}</small>
    `;
    
    container.innerHTML = html;
    container.classList.add('slide-up');
}

// モデル使用
function useModel(modelPath) {
    // 推論セクションに移動してモデルを選択
    showSection('inference');
    document.getElementById('modelSelect').value = modelPath;
    
    // ナビゲーションも更新
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector('a[href="#inference"]').classList.add('active');
}

// アラート表示ユーティリティ
function showAlert(containerId, message, type) {
    const container = document.getElementById(containerId);
    const alertClass = `alert alert-${type} alert-custom`;
    
    container.innerHTML = `
        <div class="${alertClass} fade-in">
            ${message}
        </div>
    `;
    
    // 5秒後に自動で非表示（success以外）
    if (type !== 'success') {
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                alert.style.opacity = '0';
                setTimeout(() => {
                    container.innerHTML = '';
                }, 300);
            }
        }, 5000);
    }
}