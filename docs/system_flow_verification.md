# システムフロー構造整合性検証レポート

## 目標フロー
1. **ファインチューニング（LoRA）**: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
2. **ファインチューニング済モデルの追加学習**
3. **RAGシステム**: モデル設定（ファインチューニング済+継続学習）の量子化（Ollamaで使用）
4. **文書アップロード**（PDFファイル）
5. **ハイブリッド検索・質問応答**: ファインチューニング済（量子化Ollamaモデル）

## 現状の構造検証結果

### ✅ Step 1: ファインチューニング（LoRA）
**状態: 部分的に整合**

| 項目 | 現状 | 目標との整合性 |
|------|------|--------------|
| LoRAアダプタ | `lora_20250830_223432` | ✅ 一致 |
| ベースモデル | DeepSeek-R1-Distill-Qwen-32B-Japanese | ✅ 完全一致 |
| ファイル存在 | adapter_model.safetensors (128MB) | ✅ 正常 |
| RAG設定での参照 | `outputs/lora_20250829_170202` | ❌ 不一致 |

**問題点**: RAG設定が存在しないLoRAパスを参照

---

### ✅ Step 2: ファインチューニング済モデルの追加学習
**状態: 整合**

| 項目 | 現状 | 目標との整合性 |
|------|------|--------------|
| 継続学習パイプライン | `continual_learning_pipeline.py` | ✅ 実装済み |
| EWC実装 | Fisher Information Matrix対応 | ✅ 実装済み |
| タスク管理 | 16タスク登録済み | ✅ 動作中 |
| LoRAからの継続学習 | PEFTモデル対応 | ✅ 可能 |

**構造的整合性**: 継続学習システムは正常に構築されている

---

### ⚠️ Step 3: RAGシステムモデル設定と量子化
**状態: 要修正**

| 項目 | 現状 | 目標との整合性 |
|------|------|--------------|
| モデル設定 | `use_finetuned: false` | ❌ 未使用 |
| Ollama使用 | `use_ollama_fallback: true` | ✅ 有効 |
| 量子化スクリプト | 作成済み | ✅ 準備完了 |
| 統合状態 | llama3.2:3b使用中 | ❌ 未統合 |

**問題点**: 
1. ファインチューニング済みモデルが未設定
2. 継続学習モデルの統合パスが未確立
3. 量子化→Ollama登録が未実行

---

### ✅ Step 4: 文書アップロード（PDF）
**状態: 整合**

| 項目 | 現状 | 目標との整合性 |
|------|------|--------------|
| PDFアップロード | `/rag/upload-document` | ✅ 実装済み |
| PDF処理 | OCR、表抽出対応 | ✅ 完全対応 |
| ベクトル化 | multilingual-e5-large | ✅ 正常 |
| インデックス | Qdrant (road_design_docs) | ✅ 動作中 |

**構造的整合性**: 文書処理パイプラインは完全に機能

---

### ⚠️ Step 5: ハイブリッド検索・質問応答
**状態: 部分的整合**

| 項目 | 現状 | 目標との整合性 |
|------|------|--------------|
| ハイブリッド検索 | 有効（Vector 0.7 + Keyword 0.3） | ✅ 正常 |
| 回答生成モデル | Ollama llama3.2:3b | ❌ 目標モデル未使用 |
| 引用機能 | 正常動作 | ✅ 実装済み |
| 量子化モデル統合 | 未実装 | ❌ 要対応 |

**問題点**: ファインチューニング済み量子化モデルが未統合

---

## 構造的ギャップ分析

### 🔴 重要なギャップ

1. **モデルパスの不整合**
   ```yaml
   # 現在のRAG設定
   model_path: outputs/lora_20250829_170202  # 存在しない
   
   # 実際のLoRA
   outputs/lora_20250830_223432  # DeepSeek-R1（目標モデル）
   ```

2. **継続学習→量子化の接続**
   - 継続学習後のモデル保存パス: `outputs/continual_task_*`
   - 量子化スクリプトの入力期待: LoRAまたは完全モデル
   - **ギャップ**: 継続学習出力を量子化入力に接続する処理が未定義

3. **Ollama統合の未完了**
   - 量子化スクリプト: 作成済み
   - GGUF変換: 未実行
   - Ollama登録: 未実行
   - RAG設定更新: 未実行

---

## 修正アクションプラン

### Phase 1: 即時修正（1-2時間）
```bash
# 1. RAG設定を修正
sed -i 's|outputs/lora_20250829_170202|outputs/lora_20250830_223432|g' \
  src/rag/config/rag_config.yaml

# 2. 既存LoRAを量子化
python scripts/quantize_finetuned_for_ollama.py \
  --lora-path outputs/lora_20250830_223432 \
  --quantization Q4_K_M

# 3. Ollamaに登録
ollama create deepseek-r1-japanese-q4 -f Modelfile
```

### Phase 2: 継続学習統合（3-4時間）
```python
# 継続学習→量子化パイプライン
class ContinualToOllamaPipeline:
    def train_task(self, task_data):
        # 1. 継続学習実行
        model = continual_pipeline.train_new_task(task_data)
        
        # 2. 自動量子化
        quantized = self.quantize_model(model, "Q4_K_M")
        
        # 3. Ollama登録
        self.register_to_ollama(quantized)
        
        # 4. RAG設定更新
        self.update_rag_config(model_name)
```

### Phase 3: 完全統合（1日）
1. Web UIでモデル選択機能追加
2. 自動量子化ワークフロー
3. A/Bテスト機能（llama3.2 vs 量子化モデル）

---

## 結論

### 構造的整合性評価: **70%**

✅ **整合している部分**:
- ファインチューニング基盤（LoRA）
- 継続学習システム
- 文書処理パイプライン
- ハイブリッド検索機能

❌ **不整合な部分**:
- RAG設定のモデルパス
- 継続学習→量子化の接続
- 量子化モデルのOllama統合
- 最終的な回答生成での利用

### 推奨事項

**優先度1（必須）**:
1. RAG設定のモデルパスを修正
2. DeepSeek-R1 LoRAを量子化してOllama登録

**優先度2（重要）**:
3. 継続学習出力の量子化パイプライン構築
4. RAG設定の自動更新機能

**優先度3（改善）**:
5. Web UIでのモデル管理機能
6. 性能比較ダッシュボード

これらの修正により、目標フローが完全に実現可能になります。