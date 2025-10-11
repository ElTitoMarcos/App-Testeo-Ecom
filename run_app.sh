#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/"
LOG_DIR="${ROOT}logs"
LOG_FILE="${LOG_DIR}/session.log"
PY_EXPECTED_MIN="3.11"
PY_SETUP_VER="3.12.6"
PIP_MIN_VERSION="23.0"
APP_DEFAULT_PORT=8000

mkdir -p "$LOG_DIR"
{
  echo "=================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') — run_app.sh"
} >>"$LOG_FILE"

source "${ROOT}scripts/lib/setup_utils.sh"

CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      ;;
    *)
      log_warn "Argumento desconocido: $1"
      ;;
  esac
  shift
done

if (( CLEAN )); then
  log_info "Ejecutando limpieza forzada (--clean)."
  clean_environment_artifacts
fi

log_info "Preparando entorno de Python (mínimo ${PY_EXPECTED_MIN})."
if ! bootstrap_dependencies; then
  rc=$?
  case $rc in
    1)
      log_error "No se pudo preparar el intérprete de Python."
      ;;
    2)
      log_error "No se pudo inicializar pip."
      ;;
    3)
      log_error "Falló la instalación de dependencias."
      ;;
    *)
      log_error "Fallo inesperado durante la preparación (código $rc)."
      ;;
  esac
  exit $rc
fi

log_info "Entorno listo con Python ${PY_VERSION} (origen: ${PY_SOURCE})."
launch_application

