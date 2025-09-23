#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOGFILE="logs/session.log"
mkdir -p logs

# Crear venv si no existe
if [ ! -x ".venv/bin/python3" ]; then
  echo "[INFO] Creando entorno virtual .venv ..."
  /usr/bin/python3 -m venv .venv
fi

# Activar venv
# shellcheck disable=SC1091
source ".venv/bin/activate" || { echo "[ERROR] No se pudo activar .venv"; read -r -p "Enter para salir" _; exit 1; }

echo "[INFO] Python en venv: $(python3 -V || echo 'no disponible')"

# Instalar deps si existe requirements.txt
if [ -f "requirements.txt" ]; then
  echo "[INFO] Instalando dependencias (si faltan)..."
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo
echo "============================"
echo "Ejecutando: python3 -u -m product_research_app"
echo "(consola + log en $LOGFILE)"
echo "Ctrl+C para parar el servidor."
echo "============================"
echo

python3 -u -m product_research_app 2>&1 | tee -a "$LOGFILE"
RC=${PIPESTATUS[0]}
echo
if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Salio con codigo $RC"
fi
read -r -p "Enter para cerrar..." _
exit "$RC"
