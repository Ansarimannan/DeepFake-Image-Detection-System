@echo off
REM Create the virtual environment and install dependencies.
REM Everything is derived from this script's own location, so the project can
REM be moved or renamed without breaking anything.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo ========================================
echo  Deepfake Image Detection - setup
echo ========================================
echo.

py -3.11 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.11 not found.
    echo TensorFlow 2.15 has no Windows wheels for Python 3.12 or later.
    echo Install from https://www.python.org/downloads/release/python-3119/
    echo and tick "Add Python 3.11 to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: could not create .venv
        echo If files are locked, close Python/Jupyter/your IDE and delete .venv by hand.
        pause
        exit /b 1
    )
) else (
    echo Reusing existing .venv
)

echo.
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
)

echo.
.venv\Scripts\python.exe -c "import tensorflow as tf; print('TensorFlow', tf.__version__); print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
echo.
echo Setup complete. Next: run launcher.bat, or
echo   .venv\Scripts\activate  ^&^&  set PYTHONPATH=src  ^&^&  python -m deepfake.cli smoke
pause
