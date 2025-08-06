# 継続学習管理システム トラブルシューティングガイド

## 🔍 問題の概要

継続学習管理システムでベースモデルが表示されない問題は、主に以下の原因で発生します：

1. **APIエンドポイントの未実装**
2. **コンテナとの統合問題**
3. **Webサーバーの起動問題**

## 🛠️ 解決手順

### 1. コンテナの状態確認

```bash
# コンテナが実行中か確認
docker ps | grep ai-ft-container

# コンテナが停止している場合は起動
cd docker
docker-compose up -d
```

### 2. Webサーバーの起動確認

```bash
# Webサーバーが起動しているか確認
docker exec ai-ft-container ps aux | grep uvicorn

# 起動していない場合は起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 3. APIエンドポイントの動作確認

```bash
# 継続学習APIの動作確認
curl http://localhost:8050/api/continual-learning/models

# システム情報APIの動作確認
curl http://localhost:8050/api/system-info
```

### 4. 統合テストの実行

```bash
# 統合テストスクリプトを実行
python3 scripts/test_continual_learning_integration.py
```

## 🔧 具体的な問題と解決策

### 問題1: ベースモデルが表示されない

**症状**: 継続学習管理ページでベースモデルの選択肢が空

**原因**: `/api/continual-learning/models` APIエンドポイントが未実装

**解決策**:
1. `app/main_unified.py` にAPIエンドポイントが実装されているか確認
2. Webサーバーを再起動

```bash
# Webサーバーを再起動
docker exec ai-ft-container pkill -f uvicorn
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 問題2: コンテナとの統合エラー

**症状**: 継続学習機能がコンテナ内で動作しない

**原因**: コンテナ内のファイルが不足している

**解決策**:
```bash
# 必要なファイルをコンテナにコピー
docker cp app/ ai-ft-container:/workspace/
docker cp templates/ ai-ft-container:/workspace/
docker cp src/ ai-ft-container:/workspace/
docker cp scripts/ ai-ft-container:/workspace/

# Webサーバーを再起動
docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
```

### 問題3: JavaScriptエラー

**症状**: ブラウザのコンソールでJavaScriptエラーが発生

**原因**: APIエンドポイントの応答形式が不正

**解決策**:
1. ブラウザの開発者ツールでコンソールを確認
2. ネットワークタブでAPI応答を確認
3. 必要に応じてWebサーバーを再起動

### 問題4: 継続学習APIが404エラー

**症状**: `/api/continual-learning/models` が404エラーを返す

**原因**: APIエンドポイントが正しく登録されていない

**解決策**:
```bash
# コンテナ内でAPIエンドポイントを確認
docker exec ai-ft-container python3 -c "
from app.main_unified import app
print('登録されているルート:')
for route in app.routes:
    if hasattr(route, 'path'):
        print(f'  {route.methods} {route.path}')
"
```

## 📊 デバッグ情報の確認

### 1. ログの確認

```bash
# コンテナのログを確認
docker logs ai-ft-container --tail 50

# Webサーバーのログを確認
docker exec ai-ft-container tail -f /workspace/logs/web_server.log
```

### 2. ブラウザでのデバッグ

1. ブラウザで `http://localhost:8050/continual` にアクセス
2. F12キーで開発者ツールを開く
3. コンソールタブでエラーメッセージを確認
4. ネットワークタブでAPI呼び出しを確認

### 3. API応答の確認

```bash
# 継続学習APIの詳細確認
curl -v http://localhost:8050/api/continual-learning/models

# システム情報APIの確認
curl -v http://localhost:8050/api/system-info
```

## 🚀 正常動作の確認

### 正常動作時の状態

1. **コンテナ状態**: `ai-ft-container` が実行中
2. **Webサーバー**: ポート8050でuvicornが動作中
3. **API応答**: `/api/continual-learning/models` が200ステータスで応答
4. **Webページ**: 継続学習管理ページが正常に表示
5. **モデル一覧**: ベースモデルとファインチューニング済みモデルが表示

### 確認コマンド

```bash
# 統合テストの実行
python3 scripts/test_continual_learning_integration.py

# 手動での動作確認
curl http://localhost:8050/api/continual-learning/models | jq '.[0:3]'
```

## 📝 よくある質問

### Q1: 継続学習管理システムとは何ですか？

A1: EWC（Elastic Weight Consolidation）ベースの継続学習機能を提供するシステムです。破滅的忘却を防ぎながら、段階的にモデルを改善できます。

### Q2: ベースモデルが表示されない原因は？

A2: 主な原因は以下の通りです：
- APIエンドポイントが未実装
- Webサーバーが起動していない
- コンテナとの統合問題

### Q3: 継続学習とファインチューニングの違いは？

A3: 
- **ファインチューニング**: 一度の学習で特定タスクに特化
- **継続学習**: 複数のタスクを順次学習し、以前の知識を保持

### Q4: トラブルシューティングの順序は？

A4: 以下の順序で確認してください：
1. コンテナの状態確認
2. Webサーバーの起動確認
3. APIエンドポイントの動作確認
4. ブラウザでの動作確認

## 📞 サポート

問題が解決しない場合は、以下の情報を収集してサポートに連絡してください：

1. **システム情報**:
```bash
docker exec ai-ft-container python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

2. **ログ情報**:
```bash
docker logs ai-ft-container --tail 100
```

3. **API応答**:
```bash
curl -v http://localhost:8050/api/continual-learning/models
```

4. **ブラウザのコンソールログ**: F12キーで開発者ツールを開き、コンソールタブのエラーメッセージをコピー 