#!/usr/bin/env bash
# ============================================================
# FocusGuard launcher
# Creates a venv on first run, installs deps, starts the demo.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
PY="${VENV_DIR}/bin/python"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[run_demo] Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install -r requirements.txt
    echo "[run_demo] Setup complete."
else
    # Verify opencv is importable; if not, install requirements again
    if ! "${PY}" -c "import cv2" >/dev/null 2>&1; then
        echo "[run_demo] opencv not found in venv; installing requirements..."
        "${VENV_DIR}/bin/pip" install -r requirements.txt
    fi
fi

echo "[run_demo] Starting FocusGuard..."
exec "${PY}" focusguard.py