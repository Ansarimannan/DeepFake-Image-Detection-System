@echo off
REM Open the notebook. That is all this does.
REM The notebook is self-contained: it needs no PYTHONPATH and imports nothing
REM from src/, so there is nothing else to set up.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -c "import notebook" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing Jupyter into .venv ...
    .venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
)

echo Opening "Deepfake Detection.ipynb" ...
echo Then in Jupyter: Cell -^> Run All
.venv\Scripts\python.exe -m jupyter notebook "Deepfake Detection.ipynb"
pause
