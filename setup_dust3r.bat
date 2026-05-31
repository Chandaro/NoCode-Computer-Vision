@echo off
setlocal EnableDelayedExpansion
title NoCode CV — DUSt3R Setup
cd /d "%~dp0"

echo.
echo  NoCode CV  ^|  DUSt3R 3D Reconstruction Setup
echo  ═══════════════════════════════════════════════════════
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

:: Install Python dependencies
echo  [1/3]  Installing Python dependencies (roma, einops, scipy, trimesh)...
venv\Scripts\pip.exe install "roma>=1.5.0" "einops>=0.7.0" "scipy>=1.10.0" "trimesh>=4.0.0" "matplotlib>=3.7.0" --quiet
if !errorlevel! neq 0 (
    echo  ERROR: Failed to install dependencies.
    pause & exit /b 1
)
echo         Done.
echo.

:: Clone DUSt3R
echo  [2/3]  Cloning DUSt3R repository...
if exist "dust3r\.git" (
    echo         dust3r/ already exists — pulling latest...
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

:: Install DUSt3R as editable package
echo  [3/3]  Installing DUSt3R package into venv...
venv\Scripts\pip.exe install -e dust3r --quiet
if !errorlevel! neq 0 (
    echo  ERROR: pip install -e dust3r failed.
    pause & exit /b 1
)
echo         Done.
echo.

echo  ═══════════════════════════════════════════════════════
echo  DUSt3R is ready!
echo.
echo  The model (~330 MB) will download automatically on first
echo  use from HuggingFace Hub.
echo.
echo  Restart NoCode CV and open any project → Reconstruct 3D
echo  ═══════════════════════════════════════════════════════
echo.
pause
