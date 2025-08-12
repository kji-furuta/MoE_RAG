# AI_FT_7 開発環境

このディレクトリには、GitHub リポジトリ https://github.com/kji-furuta/AI_FT_7 のクローンが作成されています。

## 📁 ディレクトリ構造

```
AI_FT_7/
├── app/                    # Webアプリケーション
├── src/                    # ソースコード
├── docker/                 # Docker設定
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                # ユーティリティスクリプト
├── data/                   # データディレクトリ
├── outputs/                # 出力ディレクトリ
├── configs/                # 設定ファイル
├── templates/              # HTMLテンプレート
├── tests/                  # テストコード
├── docs/                   # ドキュメント
├── requirements.txt        # Python依存関係
├── requirements_rag.txt    # RAG依存関係
├── pyproject.toml          # プロジェクト設定
├── setup.sh                # Linux/Mac用セットアップ
├── setup.bat               # Windows用セットアップ
└── README.md               # プロジェクト説明

```

## 🚀 セットアップ

### Linux/Mac (WSL)
```bash
chmod +x setup.sh
./setup.sh
```

### Windows
```cmd
setup.bat
```

## 📝 次のステップ

1. **環境変数の設定**
   - `.env`ファイルを編集してトークンを設定

2. **Docker環境の起動**
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Webインターフェースの起動**
   ```bash
   docker exec ai-ft-container bash /workspace/scripts/start_web_interface.sh
   ```

4. **ブラウザでアクセス**
   - http://localhost:8050

## 🔗 GitHubとの同期

完全なファイルを取得するには：
```bash
git remote add origin https://github.com/kji-furuta/AI_FT_7.git
git fetch origin
git checkout -b main origin/main
```

または、不足しているファイルを個別に取得：
```bash
git checkout origin/main -- <filepath>
```

## 📚 詳細情報

詳細な使用方法については、README.mdを参照してください。