@echo off
cd /d "%~dp0"

set "NOTEBOOK=Deepfake Image Detection — Simple Transfer Learning.ipynb"

if not exist "%NOTEBOOK%" (
    echo ERROR: Notebook not found!
    pause
    exit /b 1
)

REM Activate venv if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Check TensorFlow
python -c "import tensorflow" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: TensorFlow not installed!
    echo Run setup.bat first.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Create data directory
if not exist "data" mkdir data

echo Opening notebook...
jupyter notebook "%NOTEBOOK%"

pause

