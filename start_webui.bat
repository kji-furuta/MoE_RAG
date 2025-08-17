@echo off
REM MoE WebUI Windows Launcher
REM Windows用の簡単起動バッチファイル

echo =========================================== 
echo MoE Civil Engineering AI System
echo Windows Quick Start
echo =========================================== 

REM Python確認
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!

REM 仮想環境の作成または活性化
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Flaskインストール
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask...
    pip install flask
)

echo.
echo =========================================== 
echo Starting Web UI...
echo =========================================== 
echo.
echo Open your browser and navigate to:
echo.
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo =========================================== 
echo.

REM WebUI起動
python app\moe_simple_ui.py

pause
