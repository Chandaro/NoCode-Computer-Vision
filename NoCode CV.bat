@echo off
title NoCode CV Trainer
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0launcher.py"
) else (
    echo ERROR: App not installed. Please run "Install NoCode CV.bat" first.
    pause
)
