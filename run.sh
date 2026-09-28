#!/usr/bin/env bash
# Start the music-removal server (after ./install.sh). Stop it with Ctrl+C.
# Extra arguments and settings go through the environment, e.g.
#   MR_DEVICE=cpu ./run.sh
set -euo pipefail
cd "$(dirname "$0")/server"
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet: run ./install.sh first."
  exit 1
fi
exec .venv/bin/python server.py "$@"
