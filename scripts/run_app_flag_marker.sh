#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)/"
LOG_DIR="${ROOT}logs"
LOG_FILE="${LOG_DIR}/setup_flag.log"
FLAG_FILE="${ROOT}config/setup_complete.flag"
PY_EXPECTED_MIN="3.11"
PY_SETUP_VER="3.12.6"
PIP_MIN_VERSION="23.0"
APP_DEFAULT_PORT=8000

mkdir -p "$LOG_DIR" "${ROOT}config"
{
  echo "============================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') — run_app_flag_marker.sh"
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
  rm -f "$FLAG_FILE"
  log_info "Marcador de configuración eliminado."
fi

if [[ -f "$FLAG_FILE" ]]; then
  log_info "Marcador existente detectado; se reutilizará el entorno actual."
  if ! ensure_python_runtime; then
    log_error "No fue posible reutilizar el entorno Python existente."
    exit 1
  fi
  if ! ensure_virtualenv; then
    log_error "No se pudo activar el entorno virtual existente."
    exit 1
  fi
else
  log_info "Preparando dependencias iniciales (sin marcador previo)."
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
    rm -f "$FLAG_FILE"
    exit $rc
  fi
  {
    echo "setup completed on $(date '+%Y-%m-%d %H:%M:%S')"
    echo "python=${PYEXE}"
    echo "version=${PY_VERSION}"
  } >"$FLAG_FILE"
  log_info "Marcador creado en $FLAG_FILE."
fi

log_info "Entorno listo con Python ${PY_VERSION} (origen: ${PY_SOURCE})."
launch_application

