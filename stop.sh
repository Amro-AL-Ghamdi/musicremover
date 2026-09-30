#!/usr/bin/env bash
# Stop the music-removal server started by run.sh / install.sh (or the Linux app),
# including an install or update that is still running.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd -P)"
PORT="${MR_PORT:-8765}"

# Our processes: server.py / install.py from this folder (source checkout: server/,
# app: app/server/). launcher.py exits by itself once they're gone.
targets() {
  ps -eo pid=,args= | awk -v a="$DIR/server/" -v b="$DIR/app/server/" '
    { cmd = substr($0, index($0, $2)) }
    $2 ~ /python/ && (index(cmd, a "server.py") || index(cmd, a "install.py") ||
     index(cmd, b "server.py") || index(cmd, b "install.py")) { print $1 }'
  # Whatever listens on the server's port, if it is a Music Remover server.py (e.g. an
  # AppImage, whose folder changes on every start).
  local p=""
  if command -v lsof >/dev/null 2>&1; then
    p="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null)"
  elif command -v ss >/dev/null 2>&1; then
    p="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)"
  fi
  for pid in $p; do
    ps -o args= -p "$pid" 2>/dev/null | grep -q 'server\.py' && echo "$pid"
  done
}

# A process and everything it started (pip during an install, ffmpeg, ...).
tree() {
  echo "$1"
  for child in $(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p { print $1 }'); do tree "$child"; done
}

pids="$(for t in $(targets | sort -u); do tree "$t"; done | sort -u)"
if [ -z "$pids" ]; then
  echo "Music Remover isn't running."
  exit 0
fi
echo "Stopping Music Remover (processes: $(echo $pids))"
kill $pids 2>/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  alive=""
  for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
  [ -z "$alive" ] && { echo "Stopped."; exit 0; }
  sleep 0.5
done
echo "Still running after 5 s; forcing:$alive"
kill -9 $alive 2>/dev/null
echo "Stopped."
