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
::  Rebuild frontend if source is newer than the dist bundle
:: ═══════════════════════════════════════════════════════════════════════════════

if exist "%~dp0frontend\node_modules\.bin\vite.cmd" (
    set NEED_BUILD=0
    if not exist "%~dp0frontend\dist\index.html" set NEED_BUILD=1
    if %NEED_BUILD%==0 (
        powershell -NoProfile -Command ^
            "$dist = (Get-Item '%~dp0frontend\dist\index.html').LastWriteTime;" ^
            "$src  = (Get-ChildItem '%~dp0frontend\src' -Recurse -Include *.tsx,*.ts,*.css | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime;" ^
            "if ($src -gt $dist) { exit 1 } else { exit 0 }" >nul 2>&1
        if %errorlevel% neq 0 set NEED_BUILD=1
    )
    if %NEED_BUILD%==1 (
        echo  Rebuilding frontend…
        pushd "%~dp0frontend"
        call "%~dp0frontend\node_modules\.bin\vite.cmd" build >nul 2>&1
        popd
    )
)

:: ═══════════════════════════════════════════════════════════════════════════════
::  Launch (prefer pythonw to suppress the console window)
:: ═══════════════════════════════════════════════════════════════════════════════

if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0launcher.py"
) else (
    start "" "%~dp0venv\Scripts\python.exe"  "%~dp0launcher.py"
)
