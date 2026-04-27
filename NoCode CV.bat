@echo off
title NoCode CV Trainer
cd /d "%~dp0"

:: ═══════════════════════════════════════════════════════════════════════════════
::  Verify the app is installed
:: ═══════════════════════════════════════════════════════════════════════════════

if not exist "%~dp0venv\Scripts\python.exe" (
    echo.
    echo  ERROR: NoCode CV Trainer is not installed.
    echo  Please run "Install NoCode CV.bat" first.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0launcher.py" (
    echo.
    echo  ERROR: launcher.py not found – installation may be incomplete.
    echo  Please re-run "Install NoCode CV.bat" to repair.
    echo.
    pause
    exit /b 1
)

:: ── Quick sanity check: make sure the venv Python actually works ───────────────
"%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: The Python virtual environment is broken.
    echo  This can happen if the project folder was moved or copied.
    echo  Please re-run "Install NoCode CV.bat" to rebuild it.
    echo.
    pause
    exit /b 1
)

:: ═══════════════════════════════════════════════════════════════════════════════
::  Launch (prefer pythonw to suppress the console window)
:: ═══════════════════════════════════════════════════════════════════════════════

if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0launcher.py"
) else (
    start "" "%~dp0venv\Scripts\python.exe"  "%~dp0launcher.py"
)
