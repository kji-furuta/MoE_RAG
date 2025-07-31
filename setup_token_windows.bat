@echo off
echo === Hugging Face Token Setup for Windows ===
echo.

echo 方法1: 環境変数を直接設定
echo 以下のコマンドを実行してください:
echo set HF_TOKEN=your_token_here
echo.

echo 方法2: トークンファイルから設定
echo 1. .hf_tokenファイルを作成し、トークンを保存
echo 2. 以下のコマンドを実行:
echo for /f "delims=" %%i in (.hf_token) do set HF_TOKEN=%%i
echo.

echo 方法3: PowerShellを使用
echo $env:HF_TOKEN = "your_token_here"
echo.

echo 設定後、以下のコマンドでテスト:
echo python setup_hf_token.py
echo.

pause 