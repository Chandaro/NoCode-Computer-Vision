@echo off
cd /d "%~dp0"
echo Updating the video downloader (yt-dlp + Deno)...
venv\Scripts\pip.exe install -U "yt-dlp[default]" deno
echo.
echo Done. Restart NoCode CV.
pause
