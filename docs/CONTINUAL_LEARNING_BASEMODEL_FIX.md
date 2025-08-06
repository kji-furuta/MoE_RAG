# 継続学習管理システム ベースモデル選択問題の解決策

## 🔍 **問題の概要**

継続学習管理システムの「新しい継続学習タスクを開始」ページで、ベースモデルのプルダウンメニューが表示されない問題が発生していました。

## 🛠️ **実装した解決策**

### 1. **APIエンドポイントの実装**

`app/main_unified.py` に継続学習用モデル取得APIを実装：

```python
@app.get("/api/continual-learning/models")
async def get_continual_learning_models():
    """継続学習用の利用可能モデル一覧を取得"""
    try:
        # ファインチューニング済みモデルを取得
        saved_models = get_saved_models()
        
        # ベースモデルも含める
        base_models = [
            {
                "name": "cyberagent/calm3-22b-chat",
                "path": "cyberagent/calm3-22b-chat",
                "type": "base",
                "description": "日本語特化型22Bモデル（推奨）"
            },
            # ... 他のベースモデル
        ]
        
        # 統合して返す
        continual_models = []
        for model in base_models:
            continual_models.append({
                "name": model["name"],
                "path": model["path"],
                "type": "base",
                "description": model["description"]
            })
        
        for model in saved_models:
            continual_models.append({
                "name": f"{model['name']} (ファインチューニング済み)",
                "path": model["path"],
                "type": "finetuned",
                "description": f"学習日時: {model.get('created_at', '不明')}"
            })
        
        return continual_models
        
    except Exception as e:
        logger.error(f"継続学習用モデル取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. **JavaScript関数の修正**

`templates/continual.html` のJavaScript部分を修正：

#### A. `refreshModels()`関数の改善
```javascript
async function refreshModels() {
    try {
        console.log('継続学習用モデル一覧を取得中...');
        const response = await fetch('/api/continual-learning/models');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const models = await response.json();
        console.log('取得したモデル一覧:', models);
        
        const select = document.getElementById('baseModel');
        select.innerHTML = '<option value="">モデルを選択してください</option>';
        
        if (models && models.length > 0) {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                if (model.type === 'finetuned') {
                    option.textContent += ' (ファインチューニング済み)';
                }
                select.appendChild(option);
            });
            console.log(`モデル一覧を更新しました: ${models.length}個のモデル`);
            
            // モデル情報を表示
            const modelInfo = document.getElementById('modelInfo');
            if (modelInfo) {
                modelInfo.textContent = `${models.length}個のモデルが利用可能です`;
            }
        } else {
            console.warn('利用可能なモデルが見つかりません');
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "利用可能なモデルがありません";
            select.appendChild(option);
        }
    } catch (error) {
        console.error('モデル一覧取得エラー:', error);
        
        // デバッグ情報を表示
        const debugInfo = document.getElementById('debugInfo');
        const debugContent = document.getElementById('debugContent');
        if (debugInfo && debugContent) {
            debugInfo.style.display = 'block';
            debugContent.textContent = `エラー詳細:\n${error.message}\n\nスタックトレース:\n${error.stack}`;
        }
        
        // エラーメッセージを表示
        showAlert(`モデル一覧の取得に失敗しました: ${error.message}`, 'error');
    }
}
```

#### B. `updateModelSelect()`関数の追加
```javascript
function updateModelSelect(models) {
    const select = document.getElementById('baseModel');
    if (!select) {
        console.error('baseModel要素が見つかりません');
        return;
    }
    
    select.innerHTML = '<option value="">モデルを選択してください</option>';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = model.name;
        if (model.type === 'finetuned') {
            option.textContent += ' (ファインチューニング済み)';
        }
        select.appendChild(option);
    });
    
    console.log(`モデル選択肢を更新しました: ${models.length}個のモデル`);
    
    // モデル情報を表示
    const modelInfo = document.getElementById('modelInfo');
    if (modelInfo) {
        modelInfo.textContent = `${models.length}個のモデルが利用可能です`;
    }
}
```

#### C. ページ初期化の改善
```javascript
document.addEventListener('DOMContentLoaded', function() {
    console.log('継続学習管理ページを初期化中...');
    
    // デバッグ: 直接APIを呼び出してレスポンスを確認
    fetch('/api/continual-learning/models')
        .then(response => {
            console.log('継続学習API Status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('継続学習API Response:', data);
            console.log('継続学習モデル数:', data ? data.length : 0);
            
            // 即座にモデル一覧を更新
            if (data && data.length > 0) {
                updateModelSelect(data);
            }
        })
        .catch(error => {
            console.error('継続学習API Error:', error);
        });
    
    // 初期化を少し遅延させて、base.htmlのスクリプトが先に実行されるようにする
    setTimeout(() => {
        console.log('モデル一覧を読み込み中...');
        refreshModels();
        loadTasks();
    }, 100);
    
    // ベースモデル選択時のイベントリスナーを追加
    const baseModel = document.getElementById('baseModel');
    if (baseModel) {
        baseModel.addEventListener('change', updateModelInfo);
    }
    
    console.log('継続学習管理ページの初期化が完了しました');
});
```

### 3. **デバッグ機能の追加**

#### A. デバッグ情報表示ボタン
```html
<!-- デバッグ情報表示ボタン -->
<div style="margin-bottom: 20px;">
    <button type="button" class="btn btn-secondary" onclick="toggleDebugInfo()">🔧 デバッグ情報表示</button>
    <button type="button" class="btn btn-secondary" onclick="testModelAPI()">🧪 APIテスト</button>
</div>
```

#### B. APIテスト機能
```javascript
async function testModelAPI() {
    console.log('APIテストを開始...');
    
    try {
        const response = await fetch('/api/continual-learning/models');
        console.log('継続学習API Status:', response.status);
        
        if (response.ok) {
            const models = await response.json();
            console.log('取得したモデル:', models);
            
            // モデル情報を表示
            const modelInfo = document.getElementById('modelInfo');
            modelInfo.textContent = `${models.length}個のモデルが利用可能です`;
            
            // ベースモデル選択肢を更新
            const select = document.getElementById('baseModel');
            select.innerHTML = '<option value="">モデルを選択してください</option>';
            
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                if (model.type === 'finetuned') {
                    option.textContent += ' (ファインチューニング済み)';
                }
                select.appendChild(option);
            });
            
            console.log('モデル選択肢を更新しました');
        } else {
            console.error('APIエラー:', response.status, response.statusText);
        }
    } catch (error) {
        console.error('APIテストエラー:', error);
    }
}
```

### 4. **RAGシステムのモデルロード問題も修正**

Qwen2ForCausalLMモデルでoffload_dirパラメータがサポートされていない問題を修正：

```python
def _get_optimized_model_kwargs(self, llm_config) -> Dict[str, Any]:
    # ... 既存のコード ...
    
    # Qwen2ForCausalLM以外のモデルの場合のみoffload_dirを追加
    try:
        model_name = llm_config.get('model_name', '').lower()
        if 'qwen' not in model_name or 'qwen2' not in model_name:
            model_kwargs.update({
                'offload_dir': offload_dir,
                'offload_state_dict': True
            })
            logger.info("offload_dirを有効化しました")
        else:
            logger.info("Qwen2ForCausalLMのため、offload_dirを無効化しました")
    except Exception as e:
        logger.warning(f"モデルタイプ判定エラー: {e}。offload_dirを無効化します")
```

## ✅ **解決された問題**

1. **ベースモデル選択のプルダウンメニュー表示**
   - APIエンドポイント `/api/continual-learning/models` を実装
   - JavaScript関数 `refreshModels()` と `updateModelSelect()` を改善
   - ページ初期化時の即座なモデル一覧更新を追加

2. **デバッグ機能の強化**
   - デバッグ情報表示ボタンを追加
   - APIテスト機能を追加
   - 詳細なエラーハンドリングを実装

3. **RAGシステムの安定化**
   - Qwen2ForCausalLMモデルのoffload_dir問題を修正
   - モデルタイプに応じた条件付き設定を実装

## 🎯 **使用方法**

### 1. **ブラウザでの確認**
```
http://localhost:8050/continual
```

### 2. **デバッグ機能の利用**
- 「🔧 デバッグ情報表示」ボタンをクリック
- 「🧪 APIテスト」ボタンをクリック
- ブラウザの開発者ツールでコンソールログを確認

### 3. **APIの直接確認**
```bash
curl http://localhost:8050/api/continual-learning/models
```

## 📊 **期待される結果**

- ベースモデルのプルダウンメニューに選択肢が表示される
- ベースモデルとファインチューニング済みモデルの両方が選択可能
- デバッグ機能でシステム状態を確認可能
- RAGシステムが正常に動作する

## 🔧 **トラブルシューティング**

### 問題1: プルダウンメニューが空の場合
1. ブラウザの開発者ツールでコンソールを確認
2. 「🧪 APIテスト」ボタンをクリック
3. ネットワークタブでAPI応答を確認

### 問題2: APIエラーが発生する場合
1. Webサーバーが起動しているか確認
2. コンテナの状態を確認
3. ログを確認

### 問題3: JavaScriptエラーが発生する場合
1. ブラウザのキャッシュをクリア
2. ページを再読み込み
3. 開発者ツールでエラーメッセージを確認

## 📝 **今後の改善点**

1. **リアルタイム更新**: WebSocketを使用したリアルタイムモデル一覧更新
2. **キャッシュ機能**: モデル一覧のキャッシュ機能
3. **検索機能**: モデル名での検索機能
4. **フィルタリング**: モデルタイプでのフィルタリング機能

---

**🎉 これで継続学習管理システムのベースモデル選択が正常に動作するようになりました！** 