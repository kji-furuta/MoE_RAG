# 量子化処理ガイド

## 問題と解決策

### 問題
量子化処理中にシステムがリセットされる問題が発生していました：
- WatchFilesがllama.cppディレクトリの変更を検知
- 自動リロードによりWebサーバーが再起動
- 量子化処理が中断される

### 解決策
本番モード（production mode）を使用することで、自動リロードを無効化し、安定した量子化処理が可能になりました。

## 使用方法

### 1. Dockerコンテナで本番モードを起動

```bash
# Dockerコンテナに入る
docker exec -it ai-ft-container bash

# 本番モードでWebインターフェースを起動
/workspace/scripts/start_web_interface.sh production
```

### 2. 開発モードと本番モードの違い

#### 開発モード（デフォルト）
```bash
# 自動リロード有効（開発中に便利）
/workspace/scripts/start_web_interface.sh
# または
/workspace/scripts/start_web_interface.sh development
```
- ✅ コード変更時に自動リロード
- ❌ 量子化処理中に中断される可能性

#### 本番モード
```bash
# 自動リロード無効（量子化処理に推奨）
/workspace/scripts/start_web_interface.sh production
```
- ✅ 量子化処理が中断されない
- ✅ Ollama登録まで安定して実行
- ❌ コード変更時は手動再起動が必要

## 量子化処理の流れ

### 1. ファインチューニング
- ファインチューニングページでLoRAトレーニングを実行

### 2. 本番モードでサーバー起動
```bash
docker exec -it ai-ft-container /workspace/scripts/start_web_interface.sh production
```

### 3. RAGシステムで量子化
- RAGページの「ファインチューニング済みモデル量子化」セクションへ
- LoRAモデルを選択
- 「量子化開始」ボタンをクリック

### 4. 処理の監視
以下の処理が順番に実行されます：
1. LoRAアダプターとベースモデルのマージ（FP16）
2. GGUF形式への変換
3. Q4_K_M形式での量子化
4. Ollamaへの登録

## トラブルシューティング

### Ollamaサービスが起動しない場合
```bash
# 別ターミナルで手動起動
docker exec -it ai-ft-container ollama serve
```

### 量子化が失敗する場合
1. メモリを確認：
```bash
nvidia-smi  # GPU使用状況
free -h     # システムメモリ
```

2. ログを確認：
```bash
docker logs ai-ft-container --tail 100
```

3. 出力ディレクトリを確認：
```bash
ls -la /workspace/outputs/ollama_conversion/
```

## 改善点

### 1. 自動リロード除外設定
開発モードでも以下のディレクトリは監視対象外：
- `**/llama.cpp/**`
- `**/outputs/**`
- `**/*.gguf`
- `**/*.safetensors`
- `**/*.bin`

### 2. Ollama自動起動
スクリプトがOllamaサービスを自動的に起動するように改善

### 3. エラーハンドリング強化
- ファイル存在確認
- タイムアウト設定（10分）
- デバッグ情報の出力

## 推奨設定

32Bモデルの量子化には以下の環境を推奨：
- GPU: 24GB以上のVRAM
- システムメモリ: 64GB以上
- ディスク空き容量: 100GB以上

## 注意事項

- 量子化処理には30分〜1時間程度かかる場合があります
- 処理中はブラウザを閉じても大丈夫です
- 進捗はターミナルログで確認できます