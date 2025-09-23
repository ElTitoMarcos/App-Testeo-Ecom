#!/usr/bin/env bash
set -euo pipefail
# Ejecutar desde la carpeta del script
cd "$(dirname "$0")"

# ======== CONFIG ========
ENTRYPOINT="main.py"        # <<< CAMBIAR si tu entrada no es main.py
URL="http://127.0.0.1:8000"
TIMEOUT_SECS=60
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

# Instalar deps si hay requirements.txt
if [ -f "requirements.txt" ]; then
  echo "[INFO] Instalando dependencias (si faltan)..."
  python3 -m pip install --upgrade pip >/dev/null
  pip3 install -r requirements.txt || { echo "[ERROR] Fallo instalando requirements.txt"; read -r -p "Enter para salir"; exit 1; }
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[INFO] Lanzando $ENTRYPOINT ..."
# Arranca en background para que podamos hacer la espera activa
nohup python3 "$ENTRYPOINT" >> "logs/run.out.log" 2>> "logs/run.err.log" &

echo "[INFO] Esperando a que $URL responda (timeout ${TIMEOUT_SECS}s)..."
i=0
until curl -sSf -o /dev/null "$URL"; do
  i=$((i+1))
  if [ "$i" -ge "$TIMEOUT_SECS" ]; then
    echo "[FAIL] No hubo respuesta a tiempo. Revisa logs/run.err.log"
    read -r -p "Enter para salir" _
    exit 2
  fi
  sleep 1
done

echo "[OK] El servidor responde en $URL"
read -r -p "Enter para cerrar esta ventana (el servidor sigue en background)..." _
