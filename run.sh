#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# VoiceGuard — one-command setup + run (macOS / Linux)
#
#   ./run.sh                    start on http://localhost:8000
#   PORT=8080 ./run.sh          use a different port
#   VG_DETECTOR_MODE=heuristic ./run.sh    skip the ML model entirely
#
# Creates .venv/, installs the CORE dependencies only (light, ~30 s), starts the
# server. Nothing else needs to be installed — not Node, not ffmpeg, not torch.
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")" || exit 1

PORT="${PORT:-8000}"
# 127.0.0.1 keeps the OS firewall quiet. Set HOST=0.0.0.0 to reach the app
# from another device on your network.
HOST="${HOST:-127.0.0.1}"
VENV=".venv"
say() { printf '\n\033[1;36m==>\033[0m %s\n' "$1"; }
fail() { printf '\n\033[1;31mX\033[0m %s\n\n' "$1"; exit 1; }

# ---------------------------------------------------------------- 1. python --
PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done

if [ -z "$PY" ]; then
  found=$(command -v python3 || command -v python || echo "none")
  version=$([ "$found" != "none" ] && "$found" -V 2>&1 || echo "not installed")
  fail "VoiceGuard needs Python 3.10 or newer (found: $version).
   Install it from https://www.python.org/downloads/  (macOS: brew install python@3.12)
   then run ./run.sh again."
fi
say "Using $("$PY" -V) at $(command -v "$PY")"

# ------------------------------------------------------------------ 2. venv --
VPY="$VENV/bin/python"
USE_UV=0

venv_is_usable() {
  [ -x "$VPY" ] && "$VPY" -m pip --version >/dev/null 2>&1
}

if venv_is_usable; then
  say "Reusing existing virtual environment ($VENV)"
else
  say "Creating virtual environment in $VENV ..."
  "$PY" -m venv "$VENV" >/dev/null 2>&1
  if ! venv_is_usable; then
    # Some Python installs ship a broken pip bootstrap (a mis-linked pyexpat on
    # Homebrew Python is the usual culprit). uv can build the environment
    # without it, so use that rather than dead-ending the teammate.
    if command -v uv >/dev/null 2>&1; then
      say "This Python's pip bootstrap is broken — falling back to 'uv' ..."
      rm -rf "$VENV"
      uv venv "$VENV" || fail "uv could not create the virtual environment."
      USE_UV=1
    else
      fail "Could not create a working virtual environment with $PY.
   This is a problem with the Python install, not with VoiceGuard. Pick one:
     * macOS  :  brew reinstall python@3.12 expat   (then re-run ./run.sh)
     * Debian :  sudo apt install python3-venv
     * any OS :  curl -LsSf https://astral.sh/uv/install.sh | sh   (then re-run ./run.sh)
     * or skip Python entirely:  docker compose up"
    fi
  fi
fi

# ------------------------------------------------------------ 3. dependencies --
say "Installing core dependencies (this is quick and only happens once) ..."
if [ "$USE_UV" = "1" ]; then
  install_cmd() { uv pip install --python "$VPY" "$@"; }
else
  "$VPY" -m pip install --upgrade pip --quiet
  install_cmd() { "$VPY" -m pip install "$@"; }
fi

if ! install_cmd -r backend/requirements.txt --quiet; then
  fail "Dependency install failed.
   Most likely causes: no internet, or a corporate proxy blocking PyPI.
   Retry with a proxy:   $VENV/bin/pip install --proxy http://user:pass@host:port -r backend/requirements.txt
   Or use Docker instead:   docker compose up"
fi

# ------------------------------------------------------------ 4. ml (optional) --
if "$VPY" -c 'import torch, transformers' >/dev/null 2>&1; then
  say "ML deps detected — the pretrained deepfake model will be used."
else
  say "ML deps not installed — running on the built-in heuristic detector (fully working).
    To enable the pretrained ML model (~400 MB packages + a 378 MB model):
      $VENV/bin/pip install -r backend/requirements-ml.txt"
fi

# ----------------------------------------------------------------- 5. serve --
say "Starting VoiceGuard on http://localhost:$PORT   (press Ctrl+C to stop)"
echo
exec "$VPY" -m uvicorn main:app --app-dir backend --host "$HOST" --port "$PORT"
