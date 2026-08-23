@echo off
REM Thin menu over the CLI. Every option is one command you can also type
REM yourself; nothing happens here that the CLI does not do.
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0src"
title Deepfake Image Detection

if not exist "%PY%" (
    echo Virtual environment not found. Running setup first.
    call setup.bat
)

:menu
cls
echo ========================================
echo   Deepfake Image Detection
echo ========================================
echo.
echo   1. Smoke test        one tiny epoch, proves the wiring

echo   2. Build split       train/val/test manifest
echo   3. Dataset info      sizes, balance, majority baseline
echo   4. Train             two-stage transfer learning
echo   5. Evaluate          sealed-test metrics
echo   6. Grad-CAM          attention figure
echo   7. Run everything    split, train, evaluate, gradcam
echo   8. Tests             pytest
echo   9. Open notebook     walkthrough.ipynb in Jupyter
echo  10. Setup / reinstall
echo   0. Exit
echo.
set /p choice="Choice: "

if "%choice%"=="1" ( %PY% -m deepfake.cli smoke & pause & goto menu )
if "%choice%"=="2" ( %PY% -m deepfake.cli split --force & pause & goto menu )
if "%choice%"=="3" ( %PY% -m deepfake.cli info & pause & goto menu )
if "%choice%"=="4" ( %PY% -m deepfake.cli train & pause & goto menu )
if "%choice%"=="5" ( %PY% -m deepfake.cli evaluate & pause & goto menu )
if "%choice%"=="6" ( %PY% -m deepfake.cli gradcam & pause & goto menu )
if "%choice%"=="7" ( %PY% -m deepfake.cli all & pause & goto menu )
if "%choice%"=="8" ( %PY% -m pytest & pause & goto menu )
if "%choice%"=="9" ( call open.bat & goto menu )
if "%choice%"=="10" ( call setup.bat & goto menu )
if "%choice%"=="0" exit /b 0
goto menu
