# ✅ LoRA→GGUF変換チェックリスト

## 📦 準備完了項目
- [x] LoRAアダプタの圧縮ファイル作成済み
  - ファイル: `/home/kjifu/MoE_RAG/outputs/lora_20250830_223432.tar.gz`
  - サイズ: 833MB
- [x] 変換スクリプト準備完了
  - `cloud_gpu_conversion.py` - 完全版スクリプト
  - `QUICK_COLAB_GUIDE.md` - 簡易実行ガイド
  - `CLOUD_GPU_GUIDE.md` - 詳細ガイド

## 🚀 次のステップ

### 1. Google Driveへアップロード（約10分）
```bash
# WSL2からWindowsにコピー
cp /home/kjifu/MoE_RAG/outputs/lora_20250830_223432.tar.gz /mnt/c/Users/[ユーザー名]/Desktop/
```
→ Google Driveの「マイドライブ」直下にアップロード

### 2. Google Colab Pro+の準備
- [ ] Google Colab Pro+のサブスクリプション確認（$49.99/月）
- [ ] または無料トライアルの利用

### 3. Colabで変換実行（約60分）
- [ ] 新しいノートブック作成
- [ ] A100 GPUを選択
- [ ] `QUICK_COLAB_GUIDE.md`の手順に従って実行

### 4. ファイルダウンロード（約30分）
- [ ] `deepseek-finetuned-q4_k_m.gguf` (約18-20GB)
- [ ] `Modelfile` (1KB)

### 5. ローカル環境設定（約10分）
- [ ] WSL2にファイル転送
- [ ] Dockerコンテナにコピー
- [ ] Ollamaに登録
- [ ] RAG設定更新

## 📊 期待される結果

### 変換前（現在の状態）
- モデル: `llama3.2:3b` (Ollama)
- テスト質問: "設計速度100km/hの最小曲線半径は？"
- 現在の回答: **30m（誤答）**

### 変換後（目標）
- モデル: `deepseek-finetuned` (ファインチューニング済み)
- テスト質問: "設計速度100km/hの最小曲線半径は？"
- 期待される回答: **460m（正答）**

## 🔧 トラブルシューティング

### Google Colab Pro+が使えない場合
→ **Kaggle**の無料枠（週30時間）を利用
```python
# Kaggleでも同じスクリプトが使用可能
# GPU設定でP100またはT4を選択
```

### メモリ不足エラー
→ より強力なGPUに切り替え
- A100 40GB → A100 80GB
- またはPaperspace Gradient（$3.09/時）を利用

### ダウンロードが遅い場合
→ Google Driveから分割ダウンロード
```bash
# Google Driveのリンクを共有設定にして
# wgetやcurlでダウンロード
```

## 📝 重要な注意事項

1. **必要な時間**: 合計約2時間
   - アップロード: 10分
   - 変換処理: 60分
   - ダウンロード: 30分
   - ローカル設定: 10分

2. **必要な容量**:
   - Google Drive: 40GB以上
   - ローカルディスク: 25GB以上

3. **コスト**:
   - Google Colab Pro+: $49.99/月（推奨）
   - または Kaggle: 無料（週30時間制限）

## 🎯 最終確認コマンド

変換完了後、以下のコマンドで動作確認：

```bash
# Ollamaでモデル確認
docker exec ai-ft-container ollama list

# テスト実行
docker exec ai-ft-container ollama run deepseek-finetuned "設計速度100km/hの最小曲線半径は？"

# RAGシステムでテスト
curl -X POST "http://localhost:8050/rag/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "設計速度100km/hの最小曲線半径は？"}'
```

---

**準備は完了しています！** Google Driveへのアップロードから始めてください。