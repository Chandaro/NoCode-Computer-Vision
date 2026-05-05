#!/usr/bin/env bash
# NoCode CV Trainer — Launcher (macOS / Linux)
# Run this after installation to start the server and open the browser.

cd "$(dirname "$0")"

VENV_PY="venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo ""
    echo "  ERROR: NoCode CV Trainer is not installed."
    echo "  Please run:  bash 'Install NoCode CV.sh'"
    echo ""
    exit 1
fi

if [ ! -f "launcher.py" ]; then
    echo ""
    echo "  ERROR: launcher.py not found — installation may be incomplete."
    echo "  Please re-run the installer."
    echo ""
    exit 1
fi

"$VENV_PY" -c "import sys" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: The Python virtual environment is broken."
    echo "  This can happen if the project folder was moved or copied."
    echo "  Please re-run the installer."
    echo ""
    exit 1
fi

"$VENV_PY" launcher.py
