#!/usr/bin/env bash
set -euo pipefail
# Ejecutar siempre desde la carpeta del script
cd "$(dirname "$0")"

echo "[INFO] Directorio: $(pwd)"
mkdir -p logs

# ====== Crear venv si no existe ======
if [ ! -x ".venv/bin/python3" ]; then
  echo "[INFO] Creando entorno virtual .venv ..."
  /usr/bin/python3 -m venv .venv
fi

# ====== Activar venv ======
# shellcheck disable=SC1091
source ".venv/bin/activate" || { echo "[ERROR] No se pudo activar .venv"; read -r -p "Enter para salir" _; exit 1; }

echo "[INFO] Python en venv: $(python3 -V || echo 'no disponible')"

# ====== Instalar dependencias si procede ======
if [ -f "requirements.txt" ]; then
  echo "[INFO] Instalando dependencias (si faltan)..."
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ====== Autodeteccion de entrypoint ======
RUN_CMD=""
if [ -f "product_research_app/web_app.py" ]; then
  RUN_CMD='python3 -m product_research_app.web_app'
elif [ -f "main.py" ]; then
  RUN_CMD='python3 main.py'
elif [ -f "product_research_app/__main__.py" ]; then
  RUN_CMD='python3 -m product_research_app'
fi

if [ -z "$RUN_CMD" ]; then
  echo "[ERROR] No se encontro entrypoint: busca web_app.py, main.py o __main__.py"
  read -r -p "Enter para salir" _
  exit 1
fi

echo
echo "============================"
echo "Ejecutando: $RUN_CMD"
echo "(esta ventana quedara abierta)"
echo "============================"
echo

# Ejecutar en primer plano para ver errores en la misma ventana
eval "$RUN_CMD" || { rc=$?; echo "[ERROR] Salio con codigo $rc"; read -r -p "Enter para salir" _; exit "$rc"; }

echo
read -r -p "Enter para cerrar..." _
