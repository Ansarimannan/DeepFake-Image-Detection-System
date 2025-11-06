@echo off
cd /d "%~dp0"

echo ========================================
echo TensorFlow Checker
echo ========================================
echo.

REM Activate venv if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Run setup.bat first.
    pause
    exit /b 1
)

echo Python version:
python --version
python -c "import sys; v = sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}'); compatible = (v.major == 3 and v.minor >= 7 and v.minor <= 11); print('TensorFlow Compatible:', '✓ YES' if compatible else '✗ NO (needs 3.7-3.11)')" 2>nul
echo.

REM Try to import TensorFlow
echo Checking for TensorFlow installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))" 2>nul

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✓ TensorFlow is installed and working!
    echo ========================================
    echo.
    python -c "import tensorflow as tf; print('Version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPUs:', len(gpus) if gpus else 0)"
) else (
    echo.
    echo ========================================
    echo ✗ TensorFlow is NOT installed or not working!
    echo ========================================
    echo.
    echo To install TensorFlow, run:
    echo   launcher.bat → option 1 (Setup)
    echo   OR
    echo   setup.bat
    echo.
    echo Troubleshooting:
    echo 1. Make sure you have Python 3.11 (TensorFlow requires 3.7-3.11)
    echo 2. Run setup.bat to create venv and install packages
    echo 3. Check Python version: python --version
)

echo.
pause

