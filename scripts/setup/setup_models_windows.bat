@echo off
echo === AI_FT Model Setup for Windows ===
echo.

echo 現在のプロジェクトディレクトリを設定...
set PROJECT_DIR=%~dp0
set MODELS_DIR=%PROJECT_DIR%models

echo プロジェクトディレクトリ: %PROJECT_DIR%
echo モデルディレクトリ: %MODELS_DIR%
echo.

if not exist "%MODELS_DIR%" (
    echo モデルディレクトリを作成中...
    mkdir "%MODELS_DIR%"
    echo ✅ モデルディレクトリを作成しました
) else (
    echo ✅ モデルディレクトリが既に存在します
)

echo.
echo 環境変数を設定中...
set HF_HOME=%MODELS_DIR%
set TRANSFORMERS_CACHE=%MODELS_DIR%

echo HF_HOME=%HF_HOME%
echo TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo.

echo 既存のモデルを確認中...
if exist "%MODELS_DIR%\*" (
    dir /b "%MODELS_DIR%"
) else (
    echo 既存のモデルはありません
)

echo.
echo === 使用方法 ===
echo 1. このバッチファイルを実行して環境変数を設定
echo 2. Pythonスクリプトを実行:
echo    python setup_model_download.py
echo.
echo または、手動でモデルをダウンロード:
echo from transformers import AutoTokenizer, AutoModelForCausalLM
echo model_name = 'Qwen/Qwen2.5-14B-Instruct'
echo tokenizer = AutoTokenizer.from_pretrained(model_name)
echo model = AutoModelForCausalLM.from_pretrained(model_name)
echo.

pause 