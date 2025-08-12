@echo off
echo AI_FT_7 開発環境のセットアップを開始します...

REM 1. 現在のディレクトリを確認
echo 現在のディレクトリ: %CD%

REM 2. Python仮想環境の作成
echo Python仮想環境を作成しています...
python -m venv venv
call venv\Scripts\activate.bat

REM 3. pipのアップグレード
echo pipをアップグレードしています...
python -m pip install --upgrade pip

REM 4. 基本パッケージのインストール
echo 基本パッケージをインストールしています...
pip install -r requirements.txt

REM 5. RAGパッケージのインストール（オプション）
set /p install_rag="RAGシステムの依存関係もインストールしますか？ (y/n): "
if "%install_rag%"=="y" (
    echo RAGパッケージをインストールしています...
    pip install -r requirements_rag.txt
)

REM 6. 必要なディレクトリの作成
echo 必要なディレクトリを作成しています...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\rag_documents" mkdir data\rag_documents
if not exist "outputs" mkdir outputs
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "temp_uploads" mkdir temp_uploads
if not exist "qdrant_data" mkdir qdrant_data

REM 7. 環境変数ファイルのテンプレート作成
if not exist ".env" (
    echo 環境変数ファイルのテンプレートを作成しています...
    (
        echo # AI_FT_7 環境変数設定
        echo.
        echo # Hugging Face Token
        echo HF_TOKEN=your_huggingface_token_here
        echo.
        echo # Weights ^& Biases API Key
        echo WANDB_API_KEY=your_wandb_api_key_here
        echo.
        echo # OpenAI API Key ^(RAGで使用する場合^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # Jupyter Lab Token
        echo JUPYTER_TOKEN=your_secure_token_here
    ) > .env
    echo .envファイルが作成されました。必要に応じてトークンを設定してください。
)

REM 8. Dockerが利用可能か確認
docker --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Dockerが利用可能です。
    set /p build_docker="Docker環境をビルドしますか？ (y/n): "
    if "%build_docker%"=="y" (
        echo Docker環境をビルドしています...
        cd docker
        docker-compose build
        cd ..
        echo Docker環境のビルドが完了しました。
    )
) else (
    echo Dockerがインストールされていません。手動でインストールしてください。
)

echo.
echo ==========================================
echo セットアップが完了しました！
echo ==========================================
echo.
echo 次のステップ：
echo 1. .envファイルを編集して必要なトークンを設定
echo 2. Docker環境を起動: cd docker ^&^& docker-compose up -d
echo 3. Webインターフェースにアクセス: http://localhost:8050
echo.
echo 詳細な使用方法はREADME.mdを参照してください。
pause