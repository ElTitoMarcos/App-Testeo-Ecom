#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"   # <<< CAMBIAR AQUÍ si tu entrypoint no es main.py
URL="http://127.0.0.1:8000"
# ========================

mkdir -p logs

# Crear venv si no existe
if [ ! -x ".venv/bin/python3" ]; then
  /usr/bin/python3 -m venv .venv
fi

# Activar venv
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Instalar deps si hay requirements
if [ -f "requirements.txt" ]; then
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Lanzar servidor en background
nohup python3 "$ENTRYPOINT" >> "logs/run.out.log" 2>> "logs/run.err.log" &

# Abrir navegador cuando el servidor responda (timeout ~60s)
i=0
while [ $i -lt 60 ]; do
  if curl -sSf -o /dev/null "$URL"; then
    open "$URL"
    break
  fi
  i=$((i+1))
  sleep 1
done
