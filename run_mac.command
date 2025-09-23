#!/usr/bin/env bash
set -euo pipefail
# Ejecutar siempre desde la carpeta del script
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"   # <<< CAMBIAR SI TU ENTRADA NO ES main.py
# ========================

# Crear venv si no existe (usa el Python del sistema)
if [ ! -x ".venv/bin/python3" ]; then
  /usr/bin/python3 -m venv .venv
fi

# Activar venv
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Instalar deps si hay requirements.txt
if [ -f "requirements.txt" ]; then
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "============================"
echo "Iniciando ${ENTRYPOINT} ..."
echo "(cierra esta ventana para detener el servidor)"
echo "============================"
echo

python3 "$ENTRYPOINT"
