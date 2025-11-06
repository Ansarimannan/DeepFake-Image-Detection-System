@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo Kaggle API Setup Helper
echo ========================================
echo.
echo This script helps you set up Kaggle API credentials.
echo.
echo To use automatic dataset download, you need:
echo 1. A Kaggle account (sign up at https://www.kaggle.com)
echo 2. Your API credentials (download from https://www.kaggle.com/settings)
echo.
echo Steps:
echo 1. Go to https://www.kaggle.com/settings
echo 2. Scroll to "API" section
echo 3. Click "Create New Token" to download kaggle.json
echo 4. Place kaggle.json in: %USERPROFILE%\.kaggle\
echo.
echo Press any key to open the Kaggle settings page...
pause >nul

REM Open Kaggle settings in default browser
start https://www.kaggle.com/settings

echo.
echo ========================================
echo After downloading kaggle.json:
echo ========================================
echo.
echo 1. Create the .kaggle folder if it doesn't exist:
echo    mkdir %USERPROFILE%\.kaggle
echo.
echo 2. Copy kaggle.json to that folder:
echo    copy kaggle.json %USERPROFILE%\.kaggle\
echo.
echo 3. Set proper permissions (Windows):
echo    icacls %USERPROFILE%\.kaggle\kaggle.json /inheritance:r
echo    icacls %USERPROFILE%\.kaggle\kaggle.json /grant:r "%USERNAME%:R"
echo.
echo ========================================
echo.

REM Check if kaggle.json already exists
if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo ✓ Kaggle credentials found at: %USERPROFILE%\.kaggle\kaggle.json
    echo   You're all set! The notebook will automatically download the dataset.
) else (
    echo ⚠ Kaggle credentials not found.
    echo   Follow the steps above to set up your credentials.
    echo   Or use the manual ZIP upload option in the notebook.
)

echo.
pause

