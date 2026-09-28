#!/usr/bin/env bash
# One-step install for Linux and macOS: creates a private Python environment in
# server/.venv and installs everything, with the PyTorch build that matches your GPU
# (NVIDIA, AMD, Apple Silicon, or CPU-only if there's no usable GPU).
#
#   ./install.sh                 install
#   ./install.sh --dry-run       show what would be installed
#   ./install.sh --target cpu    force a build: cpu | nvidia | amd | apple
#   PYTHON=python3.12 ./install.sh   use a specific Python
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.9+ is needed. Install it first, e.g.:"
  echo "  Arch:          sudo pacman -S python"
  echo "  Debian/Ubuntu: sudo apt install python3 python3-venv"
  echo "  Fedora:        sudo dnf install python3"
  echo "  macOS:         brew install python   (or python.org)"
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  echo "Python 3.9+ is needed; $PY is $("$PY" -V 2>&1). Set PYTHON=... to pick another one."
  exit 1
fi

VENV=server/.venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating the Python environment in $VENV ($("$PY" -V 2>&1))"
  if ! "$PY" -m venv "$VENV"; then
    echo "Couldn't create it. On Debian/Ubuntu: sudo apt install python3-venv"
    exit 1
  fi
fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" server/install.py "$@"
chmod +x run.sh 2>/dev/null || true

# Start the server right away (not for --dry-run). Later: ./run.sh
case " $* " in *" --dry-run "*) exit 0 ;; esac
echo
echo "Starting the server (Ctrl+C to stop; next time just run ./run.sh)"
exec ./run.sh
