@echo off
cd /d "%~dp0"

echo ========================================
echo Cleanup - Force Remove Virtual Environment
echo ========================================
echo.

REM Deactivate if active
if defined VIRTUAL_ENV (
    echo Deactivating virtual environment...
    call deactivate 2>nul
    timeout /t 1 >nul
)

REM Kill Python processes
echo Closing Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM jupyter.exe /T >nul 2>&1
timeout /t 2 >nul

REM Remove venv
if exist ".venv" (
    echo Removing .venv directory...
    rmdir /s /q .venv 2>nul
    if exist ".venv" (
        echo.
        echo ERROR: Could not remove .venv
        echo Some files may still be locked.
        echo.
        echo Try:
        echo   1. Close all programs
        echo   2. Restart computer
        echo   3. Run this script again
    ) else (
        echo ✓ Virtual environment removed successfully.
    )
) else (
    echo No .venv directory found.
)

echo.
pause

