# RARdata.json 操作ガイド - クイックリファレンス

**最終更新**: 2025年12月9日 00:45

---

## 🚀 1分でわかる！RARdata.json更新フロー

### **質問**: RARdata.jsonを増加・変更した場合はどうする？

### **回答**: Web UIで3ステップ！

```
1️⃣ データ品質チェック（任意だが推奨）
   python3 scripts/check_rar_data.py RARdata.json

2️⃣ 継続学習UIにアクセス
   http://localhost:8050/ → Continual Learning タブ

3️⃣ ファイルアップロードして学習開始
   ✅ Use Previous Fisher Matrix: ON（重要！）
   ✅ Base Model: 前回の学習済みモデルを選択
```

---

## 📖 詳細ガイド

### **初めての方へ**

👉 **[UI統合版ガイド](./rardata_ui_integrated_guide.md)** ← **最も簡単**

- Web UIでの操作手順（画面付き）
- 設定パラメータの詳細説明
- トラブルシューティング

---

### **開発者向け**

👉 **[CLI/API操作ガイド](./rardata_operations_summary.md)**

- コマンドラインでの操作
- REST APIでの自動化
- スクリプト統合

---

### **学習済みモデルを使いたい**

👉 **[モデル使用ガイド](./rardata_current_status.md)**

- RAGシステムでの使用方法
- REST API経由での利用
- 性能評価方法

---

## ⚙️ 重要な設定

### **必ず ON にする設定**

| 設定項目 | 値 | 理由 |
|---------|---|------|
| **Use Previous Fisher Matrix** | ✅ **ON** | 既存知識を保持（破滅的忘却防止） |

### **選ぶべきモデル**

| 項目 | 正しい選択 | 間違った選択 |
|------|-----------|-------------|
| **Base Model** | `outputs/continual_rardata_*` | `cyberagent/DeepSeek-...` |
| | ✅ 前回の学習済みモデル | ❌ 元のベースモデル |

### **推奨パラメータ**

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| **EWC Lambda** | `5000` | バランスの取れた学習 |
| **Epochs** | `3` | 標準的な学習回数 |
| **Learning Rate** | `2e-5` | 安定した学習速度 |

---

## 🛠️ よくあるエラーと対処

### ❌ 「既存の知識が失われた」

**原因**: Use Previous Fisher Matrix が OFF

**解決**: 次回は必ず ON にする

---

### ❌ 「新しいデータが学習されない」

**原因**: EWC Lambda が高すぎる

**解決**: `5000` → `3000` に下げる

---

### ❌ 「メモリ不足エラー」

**原因**: GPUメモリ不足

**解決**:
```bash
docker restart ai-ft-container
```

---

## 📊 学習フロー図

```
RARdata.json 更新
    ↓
品質チェック（推奨）
    ↓
継続学習UI
    ↓
ファイルアップロード
    ↓
設定入力
 ✅ Use Previous Fisher: ON
 ✅ Base Model: 前回のモデル
    ↓
学習開始
    ↓
完了（10-15分）
    ↓
RAGシステムで使用
```

---

## 🎯 チェックリスト

### **学習前**

- [ ] データ品質チェック実行
- [ ] **Use Previous Fisher Matrix: ON**
- [ ] Base Model: 前回のモデル選択
- [ ] EWC Lambda: 5000

### **学習後**

- [ ] Tasks タブで完了確認
- [ ] RAGでテストクエリ実行
- [ ] 新データの回答精度確認
- [ ] 既存データの回答精度確認

---

## 📚 全ドキュメント

| ドキュメント | 対象 | 推奨度 |
|------------|------|--------|
| [rardata_ui_integrated_guide.md](./rardata_ui_integrated_guide.md) | 全員 | ⭐⭐⭐ |
| [rardata_current_status.md](./rardata_current_status.md) | ユーザー | ⭐⭐⭐ |
| [rardata_operations_summary.md](./rardata_operations_summary.md) | 開発者 | ⭐⭐ |
| [rardata_update_guide.md](./rardata_update_guide.md) | 開発者 | ⭐ |

---

## 🔧 ツール

| ツール | 用途 | 使用方法 |
|-------|------|---------|
| **品質チェック** | データ検証 | `python3 scripts/check_rar_data.py RARdata.json` |
| **継続学習UI** | Web学習 | http://localhost:8050/ |
| **CLI学習** | スクリプト実行 | `docker exec ai-ft-container python3 scripts/update_rardata_model.py` |

---

## ❓ サポート

**質問・問題がある場合**:

1. [トラブルシューティング](./rardata_ui_integrated_guide.md#トラブルシューティング)を確認
2. [詳細ガイド](./rardata_ui_integrated_guide.md)を参照
3. ログを確認: `docker logs ai-ft-container --tail 50`

---

**作成日**: 2025年12月9日 00:45
**対象**: RARdata.json操作のクイックリファレンス
