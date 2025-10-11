#!/bin/bash
# Shared bootstrap helpers for macOS shell wrappers.
# Expected globals before sourcing:
#   ROOT, LOG_DIR, LOG_FILE
# Optional globals:
#   PY_EXPECTED_MIN, PY_SETUP_VER, PY_INSTALL_VERSION,
#   PIP_MIN_VERSION, APP_DEFAULT_PORT

: "${PY_EXPECTED_MIN:=3.11}"
: "${PY_SETUP_VER:=3.12.6}"
: "${PY_INSTALL_VERSION:=${PY_SETUP_VER}}"
: "${PIP_MIN_VERSION:=23.0}"
: "${APP_DEFAULT_PORT:=8000}"
DEPENDENCY_LOG_FILE="${LOG_DIR}/dependencies_installed.log"

log_msg() {
  local level="$1"
  shift || true
  local msg="$*"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s [%s] %s\n' "$ts" "$level" "$msg" | tee -a "$LOG_FILE"
}

log_info() { log_msg "INFO" "$@"; }
log_warn() { log_msg "WARN" "$@" >&2; }
log_error() { log_msg "ERROR" "$@" >&2; }

version_ge() {
  local a="$1" b="$2"
  local IFS=.
  local -a pa pb
  read -r -a pa <<<"$a"
  read -r -a pb <<<"$b"
  local len=${#pa[@]}
  if [ ${#pb[@]} -gt $len ]; then
    len=${#pb[@]}
  fi
  local i
  for ((i=0; i<len; i++)); do
    local ai=${pa[i]:-0}
    local bi=${pb[i]:-0}
    if ((10#$ai > 10#$bi)); then
      return 0
    fi
    if ((10#$ai < 10#$bi)); then
      return 1
    fi
  done
  return 0
}

python_version_string() {
  local exe="$1"
  "$exe" -c 'import platform; print(platform.python_version())' 2>/dev/null
}

python_meets_min() {
  local exe="$1"
  "$exe" -c "import sys; sys.exit(0 if sys.version_info[:2] >= tuple(map(int, '${PY_EXPECTED_MIN}'.split('.'))) else 1)" 2>/dev/null
}

clean_environment_artifacts() {
  local removed=()
  if [ -d "${ROOT}.venv" ]; then
    rm -rf "${ROOT}.venv"
    removed+=(".venv")
  fi
  if [ -d "${ROOT}python_embed" ]; then
    rm -rf "${ROOT}python_embed"
    removed+=("python_embed")
  fi
  if [ -f "${ROOT}.python-version" ]; then
    rm -f "${ROOT}.python-version"
    removed+=(".python-version")
  fi
  if [ -f "${ROOT}config/setup_complete.flag" ]; then
    rm -f "${ROOT}config/setup_complete.flag"
    removed+=("setup marker")
  fi
  if [ ${#removed[@]} -gt 0 ]; then
    log_info "Entorno limpio. Se eliminaron: ${removed[*]}"
  else
    log_info "No se encontraron artefactos previos que limpiar."
  fi
}

PYEXE=""
PY_SOURCE=""
PY_VERSION=""

use_python_candidate() {
  local candidate="$1"
  if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
    return 1
  fi
  if python_meets_min "$candidate"; then
    PYEXE="$candidate"
    PY_VERSION="$(python_version_string "$candidate")"
    return 0
  fi
  return 1
}

find_existing_python() {
  local candidate
  if command -v python3 >/dev/null 2>&1; then
    candidate="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if use_python_candidate "$candidate"; then
      PY_SOURCE="system"
      return 0
    fi
  fi
  for candidate in python3 python; do
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
    if use_python_candidate "$candidate"; then
      PY_SOURCE="system"
      return 0
    fi
  done
  return 1
}

ensure_pyenv_python() {
  local pyenv_root="${PYENV_ROOT:-$HOME/.pyenv}"
  if [ ! -d "$pyenv_root" ]; then
    return 1
  fi
  if ! command -v pyenv >/dev/null 2>&1; then
    if [ -x "$pyenv_root/bin/pyenv" ]; then
      export PATH="$pyenv_root/bin:$PATH"
    else
      log_warn "pyenv no está en PATH pese a existir ${pyenv_root}."
      return 1
    fi
  fi
  local pyenv_bin
  pyenv_bin="$(command -v pyenv 2>/dev/null || true)"
  if [ -z "$pyenv_bin" ]; then
    return 1
  fi
  eval "$($pyenv_bin init - 2>/dev/null)" >/dev/null 2>&1 || true

  local desired="${PY_INSTALL_VERSION}"
  log_info "Detectado pyenv en ${pyenv_root}. Verificando Python ${desired}..."
  if ! pyenv versions --bare | grep -Fxq "$desired"; then
    log_info "Instalando Python ${desired} mediante pyenv (puede tardar)."
    if ! pyenv install -s "$desired"; then
      log_warn "pyenv no pudo instalar ${desired}. Si usas un proxy configura HTTP_PROXY/HTTPS_PROXY y vuelve a intentar."
      return 1
    fi
  fi
  local prefix
  prefix="$(pyenv prefix "$desired" 2>/dev/null || true)"
  if [ -n "$prefix" ] && [ -x "$prefix/bin/python3" ]; then
    (cd "$ROOT" && pyenv local "$desired" >/dev/null 2>&1 || true)
    PYEXE="$prefix/bin/python3"
    PY_SOURCE="pyenv"
    PY_VERSION="$(python_version_string "$PYEXE")"
    log_info "Usando Python ${PY_VERSION} proporcionado por pyenv."
    return 0
  fi
  log_warn "pyenv no pudo proporcionar un intérprete utilizable."
  return 1
}

ensure_brew_python() {
  if ! command -v brew >/dev/null 2>&1; then
    log_warn "Homebrew no está instalado. Instálalo con: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    return 1
  fi
  local series="python@${PY_SETUP_VER%.*}"
  local install_target="${series}"
  log_info "Verificando Python via Homebrew (${install_target})."
  if ! brew list --versions "$install_target" >/dev/null 2>&1; then
    log_info "Instalando ${install_target} con Homebrew..."
    if ! brew install "$install_target"; then
      log_warn "Falló la instalación con Homebrew. Ejecuta 'brew update' o revisa permisos. Configura HTTP_PROXY/HTTPS_PROXY si aplica."
      return 1
    fi
  else
    brew upgrade "$install_target" >/dev/null 2>&1 || true
  fi
  local prefix
  prefix="$(brew --prefix "$install_target" 2>/dev/null || true)"
  local candidate=""
  if [ -n "$prefix" ] && [ -x "$prefix/bin/python3" ]; then
    candidate="$prefix/bin/python3"
  fi
  if [ -z "$candidate" ]; then
    prefix="$(brew --prefix 2>/dev/null || true)"
    if [ -n "$prefix" ] && [ -x "$prefix/opt/${install_target}/bin/python3" ]; then
      candidate="$prefix/opt/${install_target}/bin/python3"
    fi
  fi
  if use_python_candidate "$candidate"; then
    PY_SOURCE="homebrew"
    log_info "Python encontrado en Homebrew: ${PYEXE} (versión ${PY_VERSION})."
    return 0
  fi
  log_warn "Homebrew instaló ${install_target} pero no se encontró python3 ejecutable."
  return 1
}

install_official_pkg_python() {
  if ! command -v installer >/dev/null 2>&1; then
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log_warn "curl no está disponible para descargar el instalador oficial."
    return 1
  fi
  local pkg_url="https://www.python.org/ftp/python/${PY_SETUP_VER}/python-${PY_SETUP_VER}-macos11.pkg"
  local tmp_pkg
  tmp_pkg="$(mktemp "/tmp/python-${PY_SETUP_VER}.XXXXXX.pkg")"
  log_info "Descargando instalador oficial de Python (${PY_SETUP_VER})..."
  if ! curl -L "$pkg_url" -o "$tmp_pkg"; then
    log_error "No se pudo descargar ${pkg_url}. Si estás detrás de un proxy exporta HTTP_PROXY y HTTPS_PROXY."
    rm -f "$tmp_pkg"
    return 1
  fi
  log_info "Ejecutando instalador de Python (se puede requerir contraseña de administrador)."
  if ! sudo installer -pkg "$tmp_pkg" -target / >/dev/null 2>&1; then
    log_warn "La instalación oficial falló o fue cancelada."
    rm -f "$tmp_pkg"
    return 1
  fi
  rm -f "$tmp_pkg"
  if find_existing_python; then
    PY_SOURCE="system"
    log_info "Python instalado mediante paquete oficial: ${PYEXE}."
    return 0
  fi
  return 1
}

prepare_embedded_python() {
  local embed_dir="${ROOT}python_embed"
  local embed_python="${embed_dir}/Library/Frameworks/Python.framework/Versions/${PY_SETUP_VER%.*}/bin/python3"
  if [ -x "$embed_python" ]; then
    if use_python_candidate "$embed_python"; then
      PY_SOURCE="embedded"
      log_info "Usando distribución embebida existente (${PY_VERSION})."
      return 0
    fi
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log_warn "curl no está disponible para descargar la distribución embebida."
    return 1
  fi
  if ! command -v pkgutil >/dev/null 2>&1 || ! command -v cpio >/dev/null 2>&1; then
    log_warn "Se requieren pkgutil y cpio para preparar la distribución embebida."
    return 1
  fi
  local pkg_url="https://www.python.org/ftp/python/${PY_SETUP_VER}/python-${PY_SETUP_VER}-macos11.pkg"
  local tmp_pkg
  tmp_pkg="$(mktemp "/tmp/python-${PY_SETUP_VER}.XXXXXX.pkg")"
  log_info "Descargando distribución embebida de Python (${PY_SETUP_VER})..."
  if ! curl -L "$pkg_url" -o "$tmp_pkg"; then
    log_error "No se pudo descargar ${pkg_url}. Configura HTTP_PROXY/HTTPS_PROXY si usas proxy."
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
  for payload in Python_Framework.pkg Python_CommandLine_Tools.pkg; do
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
  embed_python="${embed_dir}/Library/Frameworks/Python.framework/Versions/${PY_SETUP_VER%.*}/bin/python3"
  if [ -x "$embed_python" ]; then
    ln -snf "Library/Frameworks/Python.framework/Versions/${PY_SETUP_VER%.*}/bin" "${embed_dir}/bin"
    if use_python_candidate "$embed_python"; then
      PY_SOURCE="embedded"
      log_info "Distribución embebida preparada correctamente (${PY_VERSION})."
      return 0
    fi
  fi
  log_warn "No se encontró python3 utilizable en la distribución embebida."
  return 1
}

ensure_python_runtime() {
  PYEXE=""
  PY_SOURCE=""
  PY_VERSION=""

  if [ -x "${ROOT}.venv/bin/python" ]; then
    if use_python_candidate "${ROOT}.venv/bin/python"; then
      PY_SOURCE="venv"
      log_info "Reutilizando intérprete de .venv (${PY_VERSION})."
      return 0
    else
      log_warn "La versión de Python en .venv no cumple el mínimo. Se recreará."
      rm -rf "${ROOT}.venv"
    fi
  fi

  if ensure_pyenv_python; then
    return 0
  fi
  if find_existing_python; then
    log_info "Python del sistema detectado: ${PYEXE} (${PY_VERSION})."
    return 0
  fi
  if ensure_brew_python; then
    return 0
  fi
  if install_official_pkg_python; then
    return 0
  fi
  if prepare_embedded_python; then
    return 0
  fi
  log_error "No se pudo preparar un intérprete Python ${PY_EXPECTED_MIN}+.
Instala manualmente desde https://www.python.org/downloads/macos/ y reintenta."
  return 1
}
ensure_virtualenv() {
  local venv_dir="${ROOT}.venv"
  if [ -d "$venv_dir" ] && [ -x "$venv_dir/bin/python" ]; then
    if use_python_candidate "$venv_dir/bin/python"; then
      PY_SOURCE="venv"
      log_info "Entorno virtual reutilizado (${PY_VERSION})."
      # shellcheck disable=SC1091
      source "$venv_dir/bin/activate"
      return 0
    else
      log_warn "El entorno virtual existente usa Python incompatible. Se recreará."
      rm -rf "$venv_dir"
    fi
  fi
  mkdir -p "$venv_dir"
  log_info "Creando entorno virtual en .venv..."
  if ! "$PYEXE" -m venv "$venv_dir"; then
    log_error "No se pudo crear el entorno virtual. Si el repositorio está en una ruta sin permisos, ejecuta:"
    log_error "    python3 -m venv ~/app-testeo-ecom-venv"
    log_error "y ajusta la variable PYEXE apuntando a ~/app-testeo-ecom-venv/bin/python antes de relanzar."
    return 1
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  PYEXE="$venv_dir/bin/python"
  PY_SOURCE="venv"
  PY_VERSION="$(python_version_string "$PYEXE")"
  log_info "Entorno virtual listo (${PY_VERSION})."
  return 0
}

ensure_pip_and_tools() {
  if ! "$PYEXE" -m ensurepip --upgrade >/dev/null 2>&1; then
    log_warn "ensurepip no pudo ejecutarse. Intentando continuar con pip existente."
  fi
  if ! "$PYEXE" -m pip --version >/dev/null 2>&1; then
    log_error "pip no está disponible incluso tras ensurepip."
    return 1
  fi
  local pip_version_output
  pip_version_output=$("$PYEXE" -m pip --version 2>/dev/null)
  local pip_version=""
  if [ -n "$pip_version_output" ]; then
    set -- $pip_version_output
    pip_version="$2"
  fi
  if [ -n "$pip_version" ] && ! version_ge "$pip_version" "$PIP_MIN_VERSION"; then
    log_info "Actualizando pip a >= ${PIP_MIN_VERSION} (actual: ${pip_version})."
    if ! "$PYEXE" -m pip install --upgrade "pip>=${PIP_MIN_VERSION}"; then
      log_error "No se pudo actualizar pip. Configura HTTP_PROXY/HTTPS_PROXY si hay restricciones de red."
      return 1
    fi
  else
    log_info "pip ${pip_version:-desconocido} cumple el mínimo requerido."
  fi
  log_info "Verificando setuptools y wheel..."
  if ! "$PYEXE" -m pip install --upgrade setuptools wheel >/dev/null 2>&1; then
    log_warn "No se pudieron actualizar setuptools/wheel. Continuando."
  fi
  return 0
}

collect_dependency_actions() {
  local req_file="$1"
  "$PYEXE" - "$req_file" <<'PY'
import sys
from pkg_resources import Requirement, WorkingSet, parse_version

req_path = sys.argv[1]
requirements = []
with open(req_path, encoding='utf-8') as fh:
    for raw in fh:
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        requirements.append(raw)

ws = WorkingSet()
installed = {dist.project_name.lower(): dist.version for dist in ws}
for raw in requirements:
    try:
        req = Requirement.parse(raw)
    except Exception:
        print("\t".join(['install', raw, raw, '', '', 'formato_desconocido']))
        continue
    name = req.project_name
    key = name.lower()
    installed_version = installed.get(key)
    spec = str(req.specifier)
    if installed_version is None:
        print("\t".join(['install', raw, name, '', spec, 'no_instalado']))
        continue
    if not req.specifier:
        continue
    if req.specifier.contains(installed_version, prereleases=True):
        continue
    newer_conflict = False
    spec_parts = []
    for spec_item in req.specifier:
        spec_parts.append(f"{spec_item.operator}{spec_item.version}")
        if spec_item.operator in ('==', '===', '<', '<='):
            if parse_version(installed_version) > parse_version(spec_item.version):
                newer_conflict = True
    if newer_conflict:
        print("\t".join(['conflict', raw, name, installed_version, ','.join(spec_parts) or spec, 'installed_newer']))
        continue
    print("\t".join(['upgrade', raw, name, installed_version, ','.join(spec_parts) or spec, 'version_inferior']))
PY
}

summarize_installed_versions() {
  "$PYEXE" - <<'PY'
import sys
from pkg_resources import WorkingSet
names = [line.strip() for line in sys.stdin if line.strip()]
ws = WorkingSet()
for name in names:
    dist = ws.by_key.get(name.lower())
    if dist is None:
        print(f"{name}=no_instalado")
    else:
        print(f"{dist.project_name}={dist.version}")
PY
}

sync_requirements() {
  local req_file="${ROOT}requirements.txt"
  if [ ! -f "$req_file" ]; then
    log_warn "No se encontró requirements.txt en ${ROOT}."
    return 0
  fi
  local entries=()
  local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    entries+=("$line")
  done < <(collect_dependency_actions "$req_file")

  local install_list=()
  local install_names=()
  local upgrade_names=()
  local conflicts=()
  local entry action raw name installed spec reason
  for entry in "${entries[@]}"; do
    IFS=$'\t' read -r action raw name installed spec reason <<<"$entry"
    case "$action" in
      install)
        install_list+=("$raw")
        install_names+=("$name")
        ;;
      upgrade)
        install_list+=("$raw")
        upgrade_names+=("$name (instalada ${installed:-ninguna}, requerido ${spec:-sin especificación})")
        ;;
      conflict)
        conflicts+=("$name (instalada ${installed}, requerido ${spec:-sin especificación})")
        ;;
    esac
  done

  if [ ${#conflicts[@]} -gt 0 ]; then
    log_warn "Conflictos detectados (versiones superiores ya instaladas):"
    local conflict
    for conflict in "${conflicts[@]}"; do
      log_warn "  - ${conflict}. Considera actualizar requirements.txt."
    done
  fi

  if [ ${#install_list[@]} -eq 0 ]; then
    log_info "Las dependencias ya están satisfechas."
  else
    log_info "Instalando/actualizando dependencias necesarias..."
    log_info "  $(printf '%s ' "${install_list[@]}")"
    if ! "$PYEXE" -m pip install --upgrade "${install_list[@]}"; then
      log_error "pip no pudo instalar todas las dependencias."
      log_error "Si observas errores relacionados con bibliotecas nativas instala los paquetes del sistema necesarios:"
      log_error "  brew install libjpeg zlib"
      return 1
    fi
  fi

  local names_to_capture=()
  if [ ${#install_names[@]} -gt 0 ]; then
    names_to_capture+=("${install_names[@]}")
  fi
  if [ ${#upgrade_names[@]} -gt 0 ]; then
    local item
    for item in "${upgrade_names[@]}"; do
      names_to_capture+=("${item%% *}")
    done
  fi
  local summary_lines=""
  if [ ${#names_to_capture[@]} -gt 0 ]; then
    summary_lines="$(printf '%s\n' "${names_to_capture[@]}" | summarize_installed_versions)"
  fi
  local pip_version
  pip_version=$("$PYEXE" -m pip --version 2>/dev/null)
  if [ -n "$pip_version" ]; then
    set -- $pip_version
    pip_version="$2"
  fi
  {
    printf '--- %s ---\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Python: %s (%s)\n' "$PY_VERSION" "$PY_SOURCE"
    printf 'pip: %s\n' "${pip_version:-desconocido}"
    if [ ${#install_names[@]} -gt 0 ]; then
      printf 'Instalados: %s\n' "${install_names[*]}"
    fi
    if [ ${#upgrade_names[@]} -gt 0 ]; then
      printf 'Actualizados: %s\n' "${upgrade_names[*]}"
    fi
    if [ ${#conflicts[@]} -gt 0 ]; then
      printf 'Conflictos: %s\n' "${conflicts[*]}"
    fi
    if [ -n "$summary_lines" ]; then
      printf 'Versiones instaladas:\n%s\n' "$summary_lines"
    fi
  } >>"$DEPENDENCY_LOG_FILE"
  return 0
}

bootstrap_dependencies() {
  if ! ensure_python_runtime; then
    return 1
  fi
  if ! ensure_virtualenv; then
    return 1
  fi
  if ! ensure_pip_and_tools; then
    return 2
  fi
  if ! sync_requirements; then
    return 3
  fi
  return 0
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

launch_application() {
  local pkg="product_research_app"
  if [ ! -f "${ROOT}${pkg}/__init__.py" ]; then
    local dir
    for dir in "${ROOT}"*/; do
      if [ -f "${dir}/__init__.py" ]; then
        pkg="$(basename "$dir")"
        break
      fi
    done
  fi
  log_info "Lanzando aplicación ${pkg}.web_app con ${PYEXE}."
  open_browser_when_ready
  if ! "$PYEXE" -m "${pkg}.web_app"; then
    local rc=$?
    log_error "La aplicación finalizó con código ${rc}."
    return $rc
  fi
  return 0
}
