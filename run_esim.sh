#!/bin/bash
# eSim launcher script (with PRs #520 and #521 from VaradhaCodes integrated)

ESIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ESIM_DIR/venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
fi

# Run eSim
cd "$ESIM_DIR/src/frontEnd"
python3 Application.py "$@"
