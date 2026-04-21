@echo off
:: ─────────────────────────────────────────────────────────────────────────────
::  NoCode CV Trainer – Installer Entry Point
::  Double-click this file to open the graphical installer.
:: ─────────────────────────────────────────────────────────────────────────────

:: Verify Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python is not installed or not on PATH.
    echo  Please install Python 3.9+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Try pythonw first (no console window), fall back to python
where pythonw >nul 2>&1
if %errorlevel% equ 0 (
    start "" pythonw "%~dp0installer.py"
) else (
    start "" python "%~dp0installer.py"
)
