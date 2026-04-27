@echo off
setlocal EnableDelayedExpansion
title NoCode CV Trainer – Installer
cd /d "%~dp0"

echo.
echo  NoCode CV Trainer  ^|  Installer
echo  ═══════════════════════════════════════════════════════════
echo.

:: ═══════════════════════════════════════════════════════════════════════════════
::  FIND PYTHON 3.9+
::
::  Tries in order: python, python3, py (Windows Launcher)
::  The version check (-c "sys.exit(...)") automatically skips:
::    - Microsoft Store stubs  (they open the Store instead of running)
::    - Python versions older than 3.9
::  Falls back to common install locations if not found on PATH.
:: ═══════════════════════════════════════════════════════════════════════════════

set "PY="

:: ── 1. Check commands on PATH ─────────────────────────────────────────────────
for %%C in (python python3 py) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set "PY=%%C"
    )
)

:: ── 2. Fallback: common install directories (if not on PATH) ──────────────────
if not defined PY (
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
        "C:\Python39\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
    ) do (
        if not defined PY (
            if exist "%%~P" (
                "%%~P" -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
                if !errorlevel! equ 0 set "PY=%%~P"
            )
        )
    )
)

:: ── 3. Not found → helpful error ──────────────────────────────────────────────
if not defined PY (
    echo  ERROR: Python 3.9 or newer was not found on this system.
    echo.
    echo  Download from:  https://www.python.org/downloads/
    echo.
    echo  TIP: During installation, check "Add Python to PATH".
    echo       If Python is already installed, it may be in a conda/venv
    echo       that is not active. Activate it, then re-run this installer.
    echo.
    pause
    exit /b 1
)

:: Print discovered version for the user's confidence
for /f "tokens=*" %%V in ('"!PY!" --version 2^>^&1') do echo  Found: %%V
echo.

:: ═══════════════════════════════════════════════════════════════════════════════
::  LAUNCH INSTALLER GUI
::
::  Prefer the "windowless" variant (pythonw / python3w) so there is no
::  console flash behind the GUI.  Falls back to the normal interpreter.
:: ═══════════════════════════════════════════════════════════════════════════════

set "LAUNCH=!PY!"

:: Bare command on PATH → try windowless sibling
if /I "!PY!"=="python" (
    where pythonw >nul 2>&1
    if !errorlevel! equ 0 set "LAUNCH=pythonw"
)
if /I "!PY!"=="python3" (
    where python3w >nul 2>&1
    if !errorlevel! equ 0 set "LAUNCH=python3w"
)

:: Full path (from fallback scan) → check for pythonw.exe in the same folder
if not "!PY!"=="python" (
    if not "!PY!"=="python3" (
        if not "!PY!"=="py" (
            for %%F in ("!PY!") do (
                if exist "%%~dpFpythonw.exe" set "LAUNCH=%%~dpFpythonw.exe"
            )
        )
    )
)

echo  Launching installer window...
start "" "!LAUNCH!" "%~dp0installer.py"
