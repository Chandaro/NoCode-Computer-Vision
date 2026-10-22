@echo off
setlocal EnableDelayedExpansion
title NoCode CV - Scan the World Setup
cd /d "%~dp0"

echo.
echo  NoCode CV  ^|  Scan the World setup (MapAnything + SAM 2)
echo  =========================================================
echo   MapAnything (Apache-2.0)  real-world 3D from photos or video
echo   SAM 2.1     (Apache-2.0)  click an object, track it through every frame
echo   No CUDA compiling needed. Model weights (~5 GB) download on first scan.
echo.

if not exist "venv\Scripts\python.exe" (
    echo  ERROR: Virtual environment not found. Run "Install NoCode CV.bat" first.
    pause & exit /b 1
)
git --version >nul 2>&1
if !errorlevel! neq 0 (
    echo  ERROR: git is not installed or not on PATH. Get it from https://git-scm.com/download/win
    pause & exit /b 1
)
if not exist "third_party" mkdir third_party

:: 1. Shared dependencies. MapAnything pins opencv-python-headless, which would
::    clash with the opencv-python the app already uses, so its dependencies are
::    installed by hand and the package itself with --no-deps.
echo  [1/4]  Installing dependencies...
venv\Scripts\pip.exe install hydra-core natsort orjson python-box plyfile "uniception==0.1.7" ^
    safetensors huggingface_hub iopath trimesh --quiet
echo         Updating the video downloader (yt-dlp + Deno for YouTube)...
venv\Scripts\pip.exe install -U "yt-dlp[default]" deno --quiet
if !errorlevel! neq 0 ( echo  ERROR: dependency install failed. & pause & exit /b 1 )
echo         Done.

:: 2. MapAnything
echo  [2/4]  Getting MapAnything...
if exist "third_party\map-anything\.git" (
    git -C third_party\map-anything pull --quiet
) else (
    git clone --depth 1 https://github.com/facebookresearch/map-anything.git third_party\map-anything
    if !errorlevel! neq 0 ( echo  ERROR: git clone failed. Check your internet. & pause & exit /b 1 )
)
venv\Scripts\pip.exe install --no-deps -e third_party\map-anything --quiet
if !errorlevel! neq 0 ( echo  ERROR: MapAnything install failed. & pause & exit /b 1 )
echo         Done.

:: 3. SAM 2 (pure-Python build; the optional CUDA extension only smooths holes)
echo  [3/4]  Getting SAM 2...
if exist "third_party\sam2\.git" (
    git -C third_party\sam2 pull --quiet
) else (
    git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party\sam2
    if !errorlevel! neq 0 ( echo  ERROR: git clone failed. Check your internet. & pause & exit /b 1 )
)
set SAM2_BUILD_CUDA=0
venv\Scripts\pip.exe install --no-deps -e third_party\sam2 --quiet
if !errorlevel! neq 0 ( echo  ERROR: SAM 2 install failed. & pause & exit /b 1 )
echo         Done.

:: 4. Check
echo  [4/4]  Checking...
venv\Scripts\python.exe -c "import torch, mapanything.models, sam2.build_sam; print('  OK  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU will be very slow)')"
if !errorlevel! neq 0 ( echo  ERROR: import check failed - see the message above. & pause & exit /b 1 )

echo.
echo  Scan the World is ready. Restart NoCode CV, open a project, then "Scan the world".
echo.
pause
