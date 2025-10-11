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
APP_DEFAULT_PORT=8000

mkdir -p "$LOG_DIR"
{
  echo "============================================================"
  echo "$(date) — run_app_registry_marker.sh"
} >>"$LOG_FILE"

log_info() {
  local msg="$*"
  echo "[INFO] $msg" | tee -a "$LOG_FILE"
}

log_warn() {
  local msg="$*"
  echo "[WARN] $msg" | tee -a "$LOG_FILE" >&2
}

log_error() {
  local msg="$*"
  echo "[ERROR] $msg" | tee -a "$LOG_FILE" >&2
}

PYEXE=""
PY_VERSION=""
PY_SOURCE=""
REQ_MAJOR="${PY_EXPECTED_MIN%%.*}"
REQ_MINOR="${PY_EXPECTED_MIN#*.}"

python_version_string() {
  local exe="$1"
  "$exe" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null
}

python_meets_min() {
  local exe="$1"
  "$exe" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (${REQ_MAJOR}, ${REQ_MINOR}) else 1)" 2>/dev/null
}

use_python_candidate() {
  local candidate="$1"
  if [ ! -x "$candidate" ]; then
    return 1
  fi
  if python_meets_min "$candidate"; then
    PYEXE="$candidate"
    PY_VERSION="$(python_version_string "$candidate")"
    return 0
  fi
  return 1
}

find_system_python() {
  local candidate
  if command -v python3 >/dev/null 2>&1; then
    candidate="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [ -n "$candidate" ] && use_python_candidate "$candidate"; then
      PY_SOURCE="system"
      return 0
    fi
  fi
  for cmd in python3 python; do
    candidate="$(command -v "$cmd" 2>/dev/null || true)"
    if [ -n "$candidate" ] && use_python_candidate "$candidate"; then
      PY_SOURCE="system"
      return 0
    fi
  done
  return 1
}

prepare_embedded_python() {
  local embed_dir="${ROOT}python_embed"
  local embed_python="${embed_dir}/Library/Frameworks/Python.framework/Versions/${REQ_MAJOR}.${REQ_MINOR}/bin/python3"
  if [ -x "$embed_python" ]; then
    PYEXE="$embed_python"
    PY_VERSION="$(python_version_string "$embed_python")"
    PY_SOURCE="embedded"
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    log_error "curl no está disponible. Instala Python manualmente."
    return 1
  fi
  if ! command -v pkgutil >/dev/null 2>&1 || ! command -v cpio >/dev/null 2>&1; then
    log_error "pkgutil y cpio son necesarios para preparar la distribución embebida."
    return 1
  fi

  local pkg_url="https://www.python.org/ftp/python/${PY_SETUP_VER}/python-${PY_SETUP_VER}-macos11.pkg"
  local tmp_pkg
  tmp_pkg="$(mktemp "/tmp/python-${PY_SETUP_VER}.XXXXXX.pkg")"

  log_info "Descargando distribución embebida de Python (${PY_SETUP_VER})..."
  if ! curl -L "$pkg_url" -o "$tmp_pkg"; then
    log_error "No se pudo descargar ${pkg_url}."
    rm -f "$tmp_pkg"
    return 1
  fi

  local tmp_expand
  tmp_expand="$(mktemp -d "/tmp/python-expand-XXXXXX")"
  if ! pkgutil --expand-full "$tmp_pkg" "$tmp_expand" >/dev/null 2>&1; then
    log_error "Error al expandir el paquete de Python."
    rm -f "$tmp_pkg"
    rm -rf "$tmp_expand"
    return 1
  fi

  rm -rf "$embed_dir"
  mkdir -p "$embed_dir"

  local payload
  for payload in "Python_Framework.pkg" "Python_CommandLine_Tools.pkg"; do
    if [ -f "$tmp_expand/$payload/Payload" ]; then
      log_info "Extrayendo $payload en python_embed..."
      if ! (cd "$embed_dir" && gzip -dc "$tmp_expand/$payload/Payload" | cpio -idm >/dev/null 2>&1); then
        log_error "Falló la extracción de $payload."
        rm -f "$tmp_pkg"
        rm -rf "$tmp_expand"
        return 1
      fi
    fi
  done

  rm -f "$tmp_pkg"
  rm -rf "$tmp_expand"

  embed_python="${embed_dir}/Library/Frameworks/Python.framework/Versions/${REQ_MAJOR}.${REQ_MINOR}/bin/python3"
  if [ -x "$embed_python" ]; then
    ln -snf "Library/Frameworks/Python.framework/Versions/${REQ_MAJOR}.${REQ_MINOR}/bin" "${embed_dir}/bin"
    PYEXE="$embed_python"
    PY_VERSION="$(python_version_string "$embed_python")"
    PY_SOURCE="embedded"
    return 0
  fi

  log_error "No se encontró python3 en la distribución embebida."
  return 1
}

install_python_with_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    return 1
  fi
  log_info "Instalando Python ${PY_SETUP_VER} mediante Homebrew..."
  if ! brew list --versions python@3.12 >/dev/null 2>&1; then
    if ! brew install python@3.12; then
      log_warn "No se pudo instalar python@3.12 con Homebrew."
      return 1
    fi
  else
    brew upgrade python@3.12 >/dev/null 2>&1 || true
  fi
  local brew_prefix
  brew_prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
  if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/python3.12" ]; then
    PYEXE="$brew_prefix/bin/python3.12"
    PY_VERSION="$(python_version_string "$PYEXE")"
    PY_SOURCE="homebrew"
    return 0
  fi
  if find_system_python; then
    return 0
  fi
  return 1
}

install_python_with_pkg() {
  if ! command -v installer >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  local pkg_url="https://www.python.org/ftp/python/${PY_SETUP_VER}/python-${PY_SETUP_VER}-macos11.pkg"
  local tmp_pkg
  tmp_pkg="$(mktemp "/tmp/python-${PY_SETUP_VER}.XXXXXX.pkg")"
  log_info "Descargando instalador oficial de Python (${PY_SETUP_VER})..."
  if ! curl -L "$pkg_url" -o "$tmp_pkg"; then
    log_error "Error al descargar ${pkg_url}."
    rm -f "$tmp_pkg"
    return 1
  fi
  log_info "Ejecutando instalador de Python. Puede requerir privilegios de administrador."
  if ! sudo installer -pkg "$tmp_pkg" -target / >/dev/null; then
    log_warn "El instalador oficial falló o fue cancelado."
    rm -f "$tmp_pkg"
    return 1
  fi
  rm -f "$tmp_pkg"
  if find_system_python; then
    PY_SOURCE="system"
    return 0
  fi
  return 1
}

ensure_python() {
  PYEXE=""
  PY_VERSION=""
  PY_SOURCE=""

  if find_system_python; then
    log_info "Python del sistema detectado: $PYEXE (versión ${PY_VERSION})."
  else
    log_warn "No se encontró un Python >= ${PY_EXPECTED_MIN}."
    if install_python_with_brew; then
      log_info "Python instalado mediante Homebrew: $PYEXE."
    elif install_python_with_pkg; then
      log_info "Python instalado mediante el paquete oficial: $PYEXE."
    elif prepare_embedded_python; then
      log_info "Python embebido preparado en ${ROOT}python_embed."
    else
      log_error "No se pudo preparar Python. Instala Python ${PY_EXPECTED_MIN}+ manualmente y vuelve a intentar."
      return 1
    fi
  fi

  if [ -z "$PYEXE" ] || [ ! -x "$PYEXE" ]; then
    log_error "Python no disponible tras la instalación."
    return 1
  fi
  return 0
}

is_embedded_python() {
  [[ "$PYEXE" == "${ROOT}"python_embed* ]]
}

ensure_virtualenv() {
  if is_embedded_python; then
    log_info "Usando distribución embebida. No se creará .venv."
    return 0
  fi
  if [ ! -d "${ROOT}.venv" ]; then
    log_info "Creando entorno virtual .venv..."
    if ! "$PYEXE" -m venv "${ROOT}.venv"; then
      log_error "No se pudo crear el entorno virtual."
      return 1
    fi
  fi
  PYEXE="${ROOT}.venv/bin/python"
  PY_SOURCE="venv"
  PY_VERSION="$(python_version_string "$PYEXE")"
  log_info "Entorno virtual listo en .venv (${PY_VERSION})."
  return 0
}

ensure_pip() {
  if "$PYEXE" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  log_warn "pip no disponible; ejecutando ensurepip."
  if ! "$PYEXE" -m ensurepip --upgrade >/dev/null 2>&1; then
    log_error "No se pudo preparar pip en el intérprete seleccionado."
    return 1
  fi
  return 0
}

install_requirements() {
  if ! ensure_pip; then
    return 1
  fi
  log_info "Actualizando pip, setuptools y wheel..."
  if ! "$PYEXE" -m pip install --upgrade pip setuptools wheel; then
    log_error "Falló la actualización de herramientas de instalación."
    return 1
  fi

  if [ -f "${ROOT}requirements.txt" ]; then
    log_info "Instalando dependencias desde requirements.txt..."
    if ! "$PYEXE" -m pip install -r "${ROOT}requirements.txt"; then
      log_error "Error instalando dependencias. Revisa $LOG_FILE."
      return 1
    fi
  else
    log_warn "No se encontró requirements.txt en ${ROOT}."
  fi
  return 0
}

prepare_runtime_python() {
  PYEXE=""
  PY_VERSION=""
  PY_SOURCE=""
  if [ -x "${ROOT}.venv/bin/python" ]; then
    PYEXE="${ROOT}.venv/bin/python"
    PY_VERSION="$(python_version_string "$PYEXE")"
    PY_SOURCE="venv"
    return 0
  fi
  local embed="${ROOT}python_embed/Library/Frameworks/Python.framework/Versions/${REQ_MAJOR}.${REQ_MINOR}/bin/python3"
  if [ -x "$embed" ]; then
    PYEXE="$embed"
    PY_VERSION="$(python_version_string "$PYEXE")"
    PY_SOURCE="embedded"
    return 0
  fi
  if find_system_python; then
    return 0
  fi
  log_error "No se encontró un intérprete de Python listo para ejecutar la aplicación."
  return 1
}

open_browser_when_ready() {
  command -v open >/dev/null 2>&1 || return 0
  command -v curl >/dev/null 2>&1 || return 0
  (
    for _ in {1..60}; do
      if curl -Is "http://127.0.0.1:${APP_DEFAULT_PORT}" >/dev/null 2>&1; then
        open "http://127.0.0.1:${APP_DEFAULT_PORT}" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 1
    done
  ) &
}

run_application() {
  local pkg="product_research_app"
  if [ ! -f "${ROOT}${pkg}/__init__.py" ]; then
    for dir in "${ROOT}"*/; do
      if [ -f "${dir}/__init__.py" ]; then
        pkg="$(basename "$dir")"
        break
      fi
    done
  fi

  log_info "Lanzando aplicación con módulo ${pkg}.web_app usando ${PYEXE}."
  open_browser_when_ready
  if ! "$PYEXE" -m "${pkg}.web_app"; then
    local rc=$?
    log_error "La aplicación finalizó con código $rc."
    return $rc
  fi
  return 0
}

setup_failed() {
  log_error "La configuración inicial falló."
  defaults delete "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" >/dev/null 2>&1 || true
  exit 1
}

if defaults read "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" >/dev/null 2>&1; then
  log_info "Marcador en defaults detectado. Omitiendo instalación."
  if ! prepare_runtime_python; then
    exit 1
  fi
  run_application
  exit $?
fi

if ! ensure_python; then
  setup_failed
fi

if ! ensure_virtualenv; then
  setup_failed
fi

if ! install_requirements; then
  setup_failed
fi

defaults write "$DEFAULTS_DOMAIN" "$DEFAULTS_KEY" -string "$(date)" >/dev/null 2>&1 || log_warn "No se pudo guardar el marcador en defaults."

run_application
