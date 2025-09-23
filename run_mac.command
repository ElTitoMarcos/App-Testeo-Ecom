#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

if [ ! -d ".venv" ]; then
  /usr/bin/python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

nohup python3 main.py >> "logs/run.out.log" 2>> "logs/run.err.log" &

sleep 2
open "http://127.0.0.1:8000"
