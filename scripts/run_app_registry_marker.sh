#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)/"
LOG_DIR="${ROOT}logs"
LOG_FILE="${LOG_DIR}/setup_registry.log"
DEFAULTS_DOMAIN="com.ecomtesting.app"
DEFAULTS_KEY="SetupComplete"
PY_EXPECTED_MIN="3.11"
PY_SETUP_VER="3.12.6"
PIP_MIN_VERSION="23.0"
APP_DEFAULT_PORT=8000

mkdir -p "$LOG_DIR"
{
  echo "============================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') — run_app_registry_marker.sh"
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

remove_defaults_marker() {
  if command -v defaults >/dev/null 2>&1; then
    if defaults read "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" >/dev/null 2>&1; then
      defaults delete "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" >/dev/null 2>&1 || true
      log_info "Entrada defaults ${DEFAULTS_DOMAIN}:${DEFAULTS_KEY} eliminada."
    fi
  fi
}

if (( CLEAN )); then
  log_info "Ejecutando limpieza forzada (--clean)."
  clean_environment_artifacts
  remove_defaults_marker
fi

marker_present=0
if command -v defaults >/dev/null 2>&1; then
  if defaults read "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" >/dev/null 2>&1; then
    marker_present=1
  fi
else
  log_warn "El comando defaults no está disponible. No se podrá registrar la instalación en preferencias."
fi

if (( marker_present )); then
  log_info "Entrada defaults existente detectada; se reutilizará el entorno actual."
  if ! ensure_python_runtime; then
    log_error "No fue posible reutilizar el entorno Python existente."
    exit 1
  fi
  if ! ensure_virtualenv; then
    log_error "No se pudo activar el entorno virtual existente."
    exit 1
  fi
else
  log_info "Preparando dependencias iniciales (sin marcador de defaults)."
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
    remove_defaults_marker
    exit $rc
  fi
  if command -v defaults >/dev/null 2>&1; then
    marker_value="setup completed on $(date '+%Y-%m-%d %H:%M:%S'); python=${PYEXE}; version=${PY_VERSION}"
    defaults write "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" "$marker_value" >/dev/null 2>&1 || true
    log_info "Marcador defaults actualizado (${DEFAULTS_DOMAIN}:${DEFAULTS_KEY})."
  fi
fi

log_info "Entorno listo con Python ${PY_VERSION} (origen: ${PY_SOURCE})."
launch_application

