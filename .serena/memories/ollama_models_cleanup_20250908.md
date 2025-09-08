# Ollama Models Directory Cleanup - 2025-09-08

## 整理前の状態
- 3つの重複Modelfile（同じ内容、異なる命名）
  - road-engineering-expert.Modelfile
  - road_engineering_expert.Modelfile  
  - roadexpert.Modelfile
- 参照先モデルパスが存在しない（古いパス）
- ドキュメントなし

## 実施した作業

### 1. 削除したファイル
- **road-engineering-expert.Modelfile** - 重複（ハイフン版）
- **roadexpert.Modelfile** - 重複（短縮版）

### 2. 更新・修正
- **road_engineering_expert.Modelfile**
  - 存在しないモデルパスを修正
  - FROM句を `llama3.2:3b` に更新（利用可能なOllamaモデル）
  - 誤字修正（"対って" → "対して"）

### 3. 追加したファイル
- **README.md** - 使用方法とモデル情報のドキュメント

## 整理後の構成
```
ollama_models/
├── road_engineering_expert.Modelfile  # 道路工学専門モデル設定
└── README.md                         # ドキュメント
```

## Modelfile内容
- **ベースモデル**: llama3.2:3b（Ollama標準モデル）
- **用途**: 道路工学専門の日本語質問応答
- **パラメータ**: 
  - temperature: 0.7
  - top_p: 0.9
  - top_k: 40
  - repeat_penalty: 1.1

## 使用方法
```bash
# モデル作成
ollama create road_engineering_expert -f road_engineering_expert.Modelfile

# モデル実行
ollama run road_engineering_expert
```

## 整理効果
- 重複ファイル削除（3→1）
- 設定の正規化と修正
- ドキュメント追加による使いやすさ向上