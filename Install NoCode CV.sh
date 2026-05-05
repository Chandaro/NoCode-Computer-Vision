#!/usr/bin/env bash
# NoCode CV Trainer — Installer (macOS / Linux)
# Run once to set up the app.  After this completes, use "NoCode CV.sh" to launch.

set -e
cd "$(dirname "$0")"

echo ""
echo "  NoCode CV Trainer  |  Installer"
echo "  ════════════════════════════════════════════"
echo ""

# ── Find Python 3.9+ ──────────────────────────────────────────────────────────
PY=""

for cmd in python3 python python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PY="$cmd"
            break
        fi
    fi
done

# macOS: also check common Homebrew locations
if [ -z "$PY" ]; then
    for path in \
        /opt/homebrew/bin/python3 \
        /usr/local/bin/python3 \
        /opt/homebrew/opt/python@3.13/bin/python3.13 \
        /opt/homebrew/opt/python@3.12/bin/python3.12 \
        /opt/homebrew/opt/python@3.11/bin/python3.11; do
        if [ -f "$path" ]; then
            if "$path" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
                PY="$path"
                break
            fi
        fi
    done
fi

if [ -z "$PY" ]; then
    echo "  ERROR: Python 3.9 or newer was not found on this system."
    echo ""
    echo "  Install options:"
    echo "    macOS:  brew install python@3.13     (requires Homebrew)"
    echo "    Any:    https://www.python.org/downloads/"
    echo ""
    exit 1
fi

echo "  Found: $($PY --version)"
echo ""
echo "  Launching installer window..."
echo ""
"$PY" installer.py
