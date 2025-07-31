# Hugging Face Token設定手順

## 1. Hugging Faceトークンの取得

1. https://huggingface.co/settings/tokens にアクセス
2. ログインまたはアカウントを作成
3. "New token"をクリック
4. Token nameに任意の名前を入力（例: "ai-ft-access"）
5. Token typeで"Read"を選択
6. "Generate a token"をクリック
7. 生成されたトークン（hf_で始まる文字列）をコピー

## 2. .envファイルの更新

```bash
cd ~/AI_FT_3/AI_FT/docker
nano .env
```

以下の行を見つけて：
```
HF_TOKEN=your-huggingface-token-here
```

実際のトークンに置き換え：
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Ctrl+X → Y → Enter で保存

## 3. Dockerコンテナの再起動

```bash
cd ~/AI_FT_3/AI_FT/docker
docker-compose down
docker-compose up -d
```

## 4. 確認

```bash
docker exec ai-ft-container printenv | grep HF_TOKEN
```

正しいトークンが表示されることを確認してください。