#!/usr/bin/env bash
set -Eeuo pipefail

# ===== Config base (proyecto puede estar en USB solo-lectura) =====
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

# Quitar cuarentena si viene de Descargas
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

# ===== Utilidades =====
bold() { printf "\033[1m%s\033[0m" "$*"; }
msg()  { printf "\033[1;34m[mac_run]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[mac_run ERROR]\033[0m %s\n" "$*" >&2; }

require_clt() {
  if xcode-select -p >/dev/null 2>&1; then return 0; fi
  err "Faltan las Command Line Tools de Xcode."
  printf "%s\n" "Ejecuta en una terminal aparte: $(bold "xcode-select --install") y cuando finalice, vuelve a lanzar este script."
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

  msg "Instalando Homebrew (esto puede abrir prompts del sistema)..."
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

ensure_py311() {
  # Intentar un Python >=3.10 ya disponible
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

  # No hay >=3.10 → Brew python@3.11
  ensure_brew
  msg "Instalando python@3.11 con Homebrew..."
  brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11

  local prefix py
  prefix="$(brew --prefix)"
  # Priorizar en ESTA sesión y persistir para las siguientes
  eval "$(brew shellenv)"
  persist_path_line "eval \"\$(brew shellenv)\""
  persist_path_line "export PATH=\"$prefix/opt/python@3.11/bin:\$PATH\""
  export PATH="$prefix/opt/python@3.11/bin:$PATH"

  if [[ -x "$prefix/opt/python@3.11/bin/python3.11" ]]; then
    py="$prefix/opt/python@3.11/bin/python3.11"
  else
    py="$(command -v python3.11 || true)"
  fi
  [[ -x "$py" ]] || { err "python3.11 no encontrado tras brew."; exit 1; }
  echo "$py"
}

wait_for_url() {
  local url="${1:-http://127.0.0.1:8000}"
  local max="${2:-30}"
  for ((i=1; i<=max; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

# ===== Flujo =====
require_clt
PY_BIN="$(ensure_py311)"
msg "Usando Python: $PY_BIN ($("$PY_BIN" -V 2>/dev/null || true))"

# Venv: recrear si no existe, cambió intérprete o es <3.10
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

# Deps: solo si cambia requirements o se reconstruye venv
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

# URL destino (variable para personalizar, por defecto 8000)
APP_URL="${APP_URL:-http://127.0.0.1:8000}"

# Abrir navegador cuando esté arriba
( wait_for_url "$APP_URL" 40 && open -g "$APP_URL" ) >/dev/null 2>&1 &

msg "Lanzando product_research_app..."
python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
