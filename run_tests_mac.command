#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"          # <<< cambia si tu entrada no es main.py
TEST_URL="http://127.0.0.1:8000"
TEST_TIMEOUT="60"
# ========================

echo "[TEST] Preparando entorno macOS..."
if [ ! -x ".venv/bin/python3" ]; then
  /usr/bin/python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

if [ -f "requirements.txt" ]; then
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[TEST] Ejecutando smoke test..."
python3 smoketest.py --entrypoint "$ENTRYPOINT" --url "$TEST_URL" --timeout "$TEST_TIMEOUT" --python "$(pwd)/.venv/bin/python3"
RC=$?

echo
if [ "$RC" -eq 0 ]; then
  echo "[OK] Smoke test macOS PASADO."
else
  echo "[FAIL] Smoke test macOS FALLO (codigo $RC). Revisa logs/test_server.err.log"
fi
echo
read -r -p "Pulsa Enter para cerrar..." _
