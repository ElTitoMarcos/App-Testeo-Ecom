#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [ -f "requirements.txt" ]; then
  python -m pip install --quiet -r requirements.txt >> logs/setup.out.log 2>> logs/setup.err.log || true
fi

nohup python -m product_research_app.web_app >> logs/run.out.log 2>> logs/run.err.log & disown

# Abrir navegador tras breve espera
( sleep 2; open "http://127.0.0.1:8000" ) >/dev/null 2>&1 &
