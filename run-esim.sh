#!/bin/bash
ESIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ESIM_DIR/src:$PYTHONPATH"
export QT_LOGGING_RULES="qt5.*=false"
cd "$ESIM_DIR/src/frontEnd"
exec "$ESIM_DIR/.venv/bin/python3" ./Application.py "$@"
