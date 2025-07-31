# DeepSeek-R1-Distill-Qwen-32B-Japanese セットアップガイド

## 概要
CyberAgentのDeepSeek-R1-Distill-Qwen-32B-Japaneseは、DeepSeek-R1の推論能力を日本語に特化させた32Bパラメータの大規模言語モデルです。

## システム要件
- **GPU**: 最低16GB VRAM（4bit量子化使用時）
- **ディスク容量**: 約65GB（モデルファイル）
- **メモリ**: 32GB以上推奨

## セットアップ手順

### 1. 仮想環境の有効化
```bash
cd /home/kjifu/AI_finet
source venv/bin/activate
```

### 2. モデルのダウンロード
```bash
# huggingface-cliを使用してダウンロード
./download_deepseek_simple.sh
```

または、Pythonスクリプトを使用：
```bash
python download_deepseek_model.py
```

### 3. モデルの動作確認
```bash
python test_deepseek_fixed.py
```

## Webインターフェースでの使用

1. Webアプリケーションの起動：
```bash
python app/main_full.py
```

2. ブラウザで `http://localhost:8050` にアクセス

3. モデル選択で「DeepSeek R1 Distill Qwen 32B 日本語特化モデル」を選択

## トラブルシューティング

### モデルが読み込めない場合
```bash
# 診断スクリプトを実行
python debug_deepseek_issue.py
```

### メモリ不足エラー
- 4bit量子化が自動的に適用されますが、それでも不足する場合は他のアプリケーションを終了してください

### ダウンロードが中断される
- ネットワーク接続を確認し、`./download_deepseek_simple.sh`を再実行（自動的に再開されます）

## 特徴
- **日本語推論**: 思考過程を日本語で表示
- **高性能**: OpenAI-o1-miniを上回るベンチマークスコア
- **メモリ効率**: 4bit量子化により16GB GPUでも動作可能

## 使用例
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"

# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# 推論実行
prompt = "AIによって私たちの暮らしはどのように変わりますか？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 参考リンク
- [HuggingFace モデルページ](https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese)
- [CyberAgent公式発表](https://www.cyberagent.co.jp/news/detail/id=30723)