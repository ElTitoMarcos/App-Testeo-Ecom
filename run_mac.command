#!/usr/bin/env bash
set -euo pipefail
# Ejecutar desde la carpeta del script
cd "$(dirname "$0")"

# ======== CONFIG ========
MODULE="product_research_app.web_app"   # <<< Cambia a product_research_app.main si prefieres la CLI
HOST="127.0.0.1"
PORT="8000"
# ========================

echo "[INFO] Directorio: $(pwd)"
mkdir -p logs

# Crear venv si no existe
if [ ! -x ".venv/bin/python3" ]; then
  echo "[INFO] Creando entorno virtual .venv ..."
  /usr/bin/python3 -m venv .venv
fi

# Activar venv
# shellcheck disable=SC1091
source ".venv/bin/activate" || { echo "[ERROR] No se pudo activar el venv"; read -r -p "Enter para salir"; exit 1; }

echo "[INFO] Python: $(python3 -V || echo 'no disponible')"

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
echo "Iniciando ${MODULE} en http://${HOST}:${PORT}"
echo "(esta ventana quedara abierta)"
echo "============================"
echo

python3 -m "$MODULE" --host "$HOST" --port "$PORT" || {
  rc=$?
  echo "[ERROR] El proceso termino con codigo $rc"
  read -r -p "Enter para cerrar..." _
  exit "$rc"
}

echo
read -r -p "Enter para cerrar..." _
