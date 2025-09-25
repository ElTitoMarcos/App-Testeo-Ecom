#!/usr/bin/env bash
set -Eeuo pipefail

# ====== Config (proyecto puede estar en USB solo-lectura) ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_ID="product_research_app"
STATE_DIR="${APP_STATE_DIR:-$HOME/Library/Application Support/$APP_ID}"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/$APP_ID}"
VENV_DIR="$STATE_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.req.sha256"
PY_PATH_FILE="$VENV_DIR/.py.path"
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Quitar cuarentena si vino de Descargas
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

# ====== Utilidades ======
bold() { printf "\033[1m%s\033[0m" "$*"; }
msg()  { printf "\033[1;34m[mac_run]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[mac_run ERROR]\033[0m %s\n" "$*" >&2; }

# Spinner/heartbeat para evitar apariencia de cuelgue
run_with_spinner() {
  local -a cmd=("$@")
  "${cmd[@]}" &
  local pid=$!
  local spin='-\|/'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    i=$(( (i+1) % 4 ))
    printf "\r\033[1;34m[mac_run]\033[0m %s trabajando..." "${spin:$i:1}"
    sleep 0.3
  done
  printf "\r"
  wait "$pid"
}

require_clt() {
  # CLT instaladas si xcode-select devuelve ruta válida
  if xcode-select -p >/dev/null 2>&1; then return 0; fi
  err "Faltan las Command Line Tools de Xcode."
  printf "%s\n" "Ejecuta en una terminal aparte: $(bold "xcode-select --install")"
  printf "%s\n" "Cuando termine la instalación, vuelve a lanzar este script."
  exit 1
}

ensure_brew() {
  export HOMEBREW_NO_AUTO_UPDATE=1
  export HOMEBREW_NO_ANALYTICS=1
  export HOMEBREW_NO_ENV_HINTS=1

  if command -v brew >/dev/null 2>&1; then
    eval "$(brew shellenv)"
    return 0
  fi
  msg "Instalando Homebrew (puede requerir confirmación del sistema)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if command -v brew >/dev/null 2>&1; then
    eval "$(brew shellenv)"
  elif [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  else
    err "Homebrew no quedó disponible tras la instalación."
    exit 1
  fi
}

persist_path_line() {
  local line="$1"
  local file="$HOME/.zprofile"
  touch "$file"
  grep -Fqs "$line" "$file" || printf "%s\n" "$line" >> "$file"
}

detect_py_ge310() {
  local cand
  for cand in \
    "${PYTHON_BIN:-}" \
    "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3.11" "/opt/homebrew/bin/python3.10" \
    "/usr/local/bin/python3.12"    "/usr/local/bin/python3.11"    "/usr/local/bin/python3.10" \
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3" \
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" \
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" \
    "python3"
  do
    [[ -n "$cand" ]] || continue
    if command -v "$cand" >/dev/null 2>&1; then
      cand="$(command -v "$cand")"
      local maj min
      maj="$("$cand" -c 'import sys; print(sys.version_info[0])')" || true
      min="$("$cand" -c 'import sys; print(sys.version_info[1])')" || true
      if [[ "$maj" =~ ^[0-9]+$ && "$min" =~ ^[0-9]+$ && ( $maj -gt 3 || ( $maj -eq 3 && $min -ge 10 ) ) ]]; then
        echo "$cand"; return 0
      fi
    fi
  done
  return 1
}

ensure_py311() {
  local py
  if py="$(detect_py_ge310)"; then
    echo "$py"; return 0
  fi
  ensure_brew
  msg "Instalando python@3.11 con Homebrew (esto puede tardar, mostrando progreso)..."
  # Usamos spinner para feedback visual mientras brew trabaja
  run_with_spinner brew install python@3.11
  local prefix pybin
  prefix="$(brew --prefix)"
  eval "$(brew shellenv)"
  persist_path_line "eval \"$(brew shellenv)\""
  persist_path_line "export PATH=\"$prefix/opt/python@3.11/bin:\$PATH\""
  export PATH="$prefix/opt/python@3.11/bin:$PATH"
  pybin="$prefix/opt/python@3.11/bin/python3.11"
  if [[ ! -x "$pybin" ]]; then
    pybin="$(command -v python3.11 || true)"
  fi
  [[ -x "$pybin" ]] || { err "python3.11 no encontrado tras brew."; exit 1; }
  echo "$pybin"
}

wait_for_url() {
  local url="${1:-http://127.0.0.1:8000}"
  local max="${2:-40}"
  local code="000"
  for ((i=1; i<=max; i++)); do
    code="$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo 000)"
    # Considera 2xx, 3xx y 4xx como “arriba” (al menos hay servidor)
    if [[ "$code" != "000" && "$code" != "000" && "$code" -ge 200 && "$code" -lt 500 ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# ====== Flujo principal ======
require_clt

PY_BIN="$(ensure_py311)"
msg "Usando Python: $PY_BIN ($("$PY_BIN" -V 2>/dev/null || true))"

# Rehacer venv si no existe, si cambió intérprete o si es <3.10
needs_rebuild=0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  needs_rebuild=1
elif [[ -f "$PY_PATH_FILE" && "$(cat "$PY_PATH_FILE")" != "$PY_BIN" ]]; then
  needs_rebuild=1
else
  "$VENV_DIR/bin/python" - <<'PY' || needs_rebuild=1
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)
PY
fi

if [[ "$needs_rebuild" -eq 1 ]]; then
  msg "Creando venv con $PY_BIN ..."
  rm -rf "$VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# Activar venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
printf "%s" "$PY_BIN" > "$PY_PATH_FILE"

# Instalar deps solo si cambia requirements o se rehace venv
calc_hash() { [[ -f "$REQ_FILE" ]] && shasum -a 256 "$REQ_FILE" | awk '{print $1}'; }
old="$(cat "$HASH_FILE" 2>/dev/null || true)"
new="$(calc_hash || true)"
if [[ "$needs_rebuild" -eq 1 || "$new" != "$old" ]]; then
  msg "Actualizando pip/setuptools/wheel y requisitos..."
  python -m pip install --upgrade pip wheel setuptools
  if [[ -n "$new" ]]; then
    pip install -r "$REQ_FILE"
    printf "%s" "$new" > "$HASH_FILE"
  fi
fi

export PYTHONUNBUFFERED=1

# URL a abrir (cámbiala exportando APP_URL si tu servidor no usa 8000)
APP_URL="${APP_URL:-http://127.0.0.1:8000}"

# Abrir navegador solo cuando responda el servidor (2xx–4xx)
( msg "Esperando a $APP_URL ..."; if wait_for_url "$APP_URL" 60; then open -g "$APP_URL"; else msg "No se pudo verificar $APP_URL; puedes abrirlo manualmente."; fi ) >/dev/null 2>&1 &

msg "Lanzando product_research_app..."
python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
