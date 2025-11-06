@echo off
cd /d "%~dp0"

echo ========================================
echo Deepfake Detection - Setup
echo ========================================
echo.

REM Check Python 3.11
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.11 not found!
    echo.
    echo Download from: https://www.python.org/downloads/release/python-3119/
    echo Check "Add Python 3.11 to PATH" during installation.
    pause
    exit /b 1
)

echo Python 3.11 found: 
py -3.11 --version
echo.

REM Deactivate if in venv
if defined VIRTUAL_ENV (
    echo Deactivating current virtual environment...
    call deactivate 2>nul
    timeout /t 2 >nul
)

REM Check for running Python processes
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo WARNING: Python processes are running!
    echo.
    echo Please close:
    echo   - Jupyter Notebook (if open)
    echo   - Any Python scripts
    echo   - IDEs using this virtual environment
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
    echo.
    echo Attempting to close Python processes...
    taskkill /F /IM python.exe /T >nul 2>&1
    timeout /t 2 >nul
)

REM Remove old venv with retry
if exist ".venv" (
    echo Removing old virtual environment...
    rmdir /s /q .venv 2>nul
    if exist ".venv" (
        echo Waiting for files to be released...
        timeout /t 3 >nul
        rmdir /s /q .venv 2>nul
        if exist ".venv" (
            echo ERROR: Cannot remove .venv - files are locked!
            echo.
            echo Please:
            echo   1. Close all Python/Jupyter processes
            echo   2. Close any IDEs (VS Code, PyCharm, etc.)
            echo   3. Try again
            pause
            exit /b 1
        )
    )
    echo ✓ Old virtual environment removed.
    echo.
)

REM Create new venv
echo Creating virtual environment with Python 3.11...
py -3.11 -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

REM Activate
call .venv\Scripts\activate.bat

REM Verify Python version
python --version
python -c "import sys; v=sys.version_info; assert v.major==3 and v.minor==11; print('✓ Python 3.11 confirmed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Virtual environment is not using Python 3.11!
    pause
    exit /b 1
)

REM Clean pip cache if corrupted
if exist ".venv\Lib\site-packages\~ip" (
    echo Cleaning corrupted pip cache...
    rmdir /s /q ".venv\Lib\site-packages\~ip" 2>nul
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet --no-cache-dir
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: pip upgrade failed, trying without cache...
    python -m pip install --upgrade pip --no-cache-dir
)

REM Install requirements
echo.
echo Installing packages (this may take 5-10 minutes)...
echo Please wait, do not close this window...
echo.
python -m pip install --no-cache-dir -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✓ Setup Complete!
    echo ========================================
    echo.
    python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)" 2>nul
    echo.
    echo To activate: .venv\Scripts\activate
) else (
    echo.
    echo ERROR: Installation failed!
    echo Check errors above.
)

pause

