#!/usr/bin/env bash
set -euo pipefail
# Ejecutar siempre desde la carpeta del script
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"          # <<< CAMBIAR si tu entrada no es main.py
LOGFILE="logs/session.log"
# ========================

echo "[INFO] Directorio: $(pwd)"
mkdir -p logs

# Crear venv si no existe
if [ ! -x ".venv/bin/python3" ]; then
  echo "[INFO] Creando entorno virtual .venv ..."
  /usr/bin/python3 -m venv .venv || { echo "[ERROR] No se pudo crear .venv"; read -r -p "Enter para salir" _; exit 1; }
fi

# Activar venv
# shellcheck disable=SC1091
source ".venv/bin/activate" || { echo "[ERROR] No se pudo activar .venv"; read -r -p "Enter para salir" _; exit 1; }

echo "[INFO] Python en venv: $(python3 -V || echo 'no disponible')"

# Instalar deps si existe requirements.txt
if [ -f "requirements.txt" ]; then
  echo "[INFO] Instalando dependencias (si faltan)..."
  python3 -m pip install --upgrade pip >/dev/null
  if ! pip3 install -r requirements.txt; then
    echo "[ERROR] Fallo instalando requirements.txt"
    read -r -p "Enter para salir" _
    exit 1
  fi
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo
echo "============================"
echo "Ejecutando: python3 -u \"$ENTRYPOINT\""
echo "(se vera en consola y se guardara en $LOGFILE)"
echo "Ctrl+C para parar el servidor."
echo "============================"
echo

# -u para salida sin buffer; tee duplica a consola y a fichero
set +e
python3 -u "$ENTRYPOINT" 2>&1 | tee -a "$LOGFILE"
RC=${PIPESTATUS[0]}
set -e

echo
if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Salio con codigo $RC"
fi
read -r -p "Enter para cerrar..." _
exit "$RC"
