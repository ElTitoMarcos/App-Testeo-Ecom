#!/usr/bin/env bash
set -euo pipefail
# Ejecutar siempre desde la carpeta del script
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"    # <<< CAMBIAR si tu entrada no es main.py
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
echo "Ejecutando: python3 \"$ENTRYPOINT\""
echo "(esta ventana quedara abierta; errores se veran aqui)"
echo "============================"
echo

python3 "$ENTRYPOINT" || { rc=$?; echo "[ERROR] Salio con codigo $rc"; read -r -p "Enter para salir" _; exit "$rc"; }

echo
read -r -p "Enter para cerrar..." _
