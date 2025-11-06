@echo off
cd /d "%~dp0"
title Deepfake Detection

:menu
cls
echo.
echo ========================================
echo   Deepfake Image Detection
echo ========================================
echo.
echo   1. Setup (Install Python 3.11 + Packages)
echo   2. Cleanup (Remove .venv if locked)
echo   3. Download Dataset
echo   4. Open Notebook (Interactive)
echo   5. Run Notebook (Automated)
echo   6. Check TensorFlow
echo   7. Setup Kaggle API
echo   8. Exit
echo.
echo ========================================
set /p choice="Choice (1-8): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto cleanup
if "%choice%"=="3" goto download
if "%choice%"=="4" goto open
if "%choice%"=="5" goto run
if "%choice%"=="6" goto checktf
if "%choice%"=="7" goto kaggle
if "%choice%"=="8" goto end

goto menu

:setup
cls
call setup.bat
goto menu

:cleanup
cls
call cleanup.bat
goto menu

:download
cls
call download_dataset.bat
goto menu

:open
cls
call open.bat
goto menu

:run
cls
call run.bat
goto menu

:checktf
cls
call check_tensorflow.bat
goto menu

:kaggle
cls
call setup_kaggle.bat
goto menu

:end
exit
