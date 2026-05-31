@echo off
setlocal EnableDelayedExpansion
title NoCode CV — DUSt3R Setup
cd /d "%~dp0"

echo.
echo  NoCode CV  ^|  DUSt3R 3D Reconstruction Setup
echo  ═══════════════════════════════════════════════
echo.

:: Check venv exists
if not exist "venv\Scripts\pip.exe" (
    echo  ERROR: Virtual environment not found.
    echo  Please run "Install NoCode CV.bat" first.
    echo.
    pause & exit /b 1
)

:: Check git is available
git --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  ERROR: git is not installed or not on PATH.
    echo  Download from: https://git-scm.com/download/win
    echo.
    pause & exit /b 1
)

:: ── Step 1: Python dependencies ───────────────────────────────────────────
echo  [1/3]  Installing Python dependencies...
venv\Scripts\pip.exe install "roma>=1.5.0" "einops>=0.7.0" "scipy>=1.10.0" ^
    "trimesh>=4.0.0" "matplotlib>=3.7.0" --quiet
if !errorlevel! neq 0 (
    echo  ERROR: Failed to install dependencies.
    pause & exit /b 1
)
echo         Done.
echo.

:: ── Step 2: Clone DUSt3R ──────────────────────────────────────────────────
echo  [2/3]  Cloning DUSt3R repository...
if exist "dust3r\.git" (
    echo         dust3r/ already exists — updating...
    cd dust3r
    git pull --quiet
    git submodule update --init --recursive --quiet
    cd ..
) else (
    git clone --recursive https://github.com/naver/dust3r.git
    if !errorlevel! neq 0 (
        echo  ERROR: git clone failed. Check your internet connection.
        pause & exit /b 1
    )
)
echo         Done.
echo.

:: ── Step 3: Register dust3r + croco on Python path via .pth file ─────────
::
::  DUSt3R has no setup.py / pyproject.toml so pip install -e . does not work.
::  Instead we write a .pth file into the venv site-packages directory.
::  Python reads every .pth file at startup and adds each line as a sys.path
::  entry — identical effect to pip install -e, no build step required.
::
echo  [3/3]  Registering dust3r on Python path...

:: Write .pth file directly into venv\Lib\site-packages
:: (getsitepackages()[0] returns the venv root, not Lib\site-packages)
set SITE=%~dp0venv\Lib\site-packages

if not exist "%SITE%" (
    echo  ERROR: %SITE% not found.
    pause & exit /b 1
)

:: Two entries: repo root (dust3r package) + croco submodule (croco package)
echo %~dp0dust3r>  "%SITE%\dust3r_path.pth"
echo %~dp0dust3r\croco>> "%SITE%\dust3r_path.pth"

:: Verify Python can now import dust3r
venv\Scripts\python.exe -c "import dust3r; print('  dust3r import OK')" 2>&1
if !errorlevel! neq 0 (
    echo  WARNING: dust3r import test failed — check the output above.
) else (
    echo         Done.
)
echo.

echo  ═══════════════════════════════════════════════
echo  DUSt3R is ready!
echo.
echo  Model (~330 MB) downloads automatically on first
echo  use from HuggingFace Hub.
echo.
echo  Next steps:
echo    1. Restart NoCode CV server
echo    2. Open any project
echo    3. Click  Reconstruct 3D
echo  ═══════════════════════════════════════════════
echo.
pause
