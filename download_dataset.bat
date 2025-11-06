@echo off
cd /d "%~dp0"

echo ========================================
echo Download Dataset Helper
echo ========================================
echo.


REM Activate venv if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Checking Kaggle credentials...
if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo ✓ Kaggle credentials found!
    echo.
    echo Downloading dataset from Kaggle...
    echo This may take a few minutes...
    echo.
    
    python -c "import kaggle; from pathlib import Path; import os; os.makedirs('data', exist_ok=True); kaggle.api.dataset_download_files('ciplab/real-and-fake-face-detection', path='data', unzip=True); print('✓ Download complete!')"
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✓ Dataset downloaded successfully!
        echo Location: data\real-and-fake-face-detection
        echo.
        echo The notebook should now detect the dataset.
    ) else (
        echo.
        echo ERROR: Download failed!
        echo.
        echo Try manual download:
        echo   1. Go to: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
        echo   2. Click "Download" button
        echo   3. Extract ZIP to: data\real_fake
        echo   4. Make sure it has 'real' and 'fake' subfolders
    )
) else (
    echo ⚠ Kaggle credentials not found!
    echo.
    echo OPTION 1: Setup Kaggle API (Recommended)
    echo   Run: setup_kaggle.bat
    echo   Then run this script again
    echo.
    echo OPTION 2: Manual Download
    echo   1. Go to: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
    echo   2. Click "Download" (requires Kaggle account)
    echo   3. Extract ZIP file
    echo   4. In the notebook, use: load_from_zip('path/to/file.zip')
    echo.
    echo OPTION 3: Use Helper Cell in Notebook
    echo   - Find the "HELPER: Load dataset from ZIP file" cell
    echo   - Uncomment and add your ZIP file path
    echo   - Run that cell
)

echo.
pause

