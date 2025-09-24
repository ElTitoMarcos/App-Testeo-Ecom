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
msg() { printf "\033[1;34m[mac_run]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[mac_run ERROR]\033[0m %s\n" "$*" >&2; }

ensure_brew() {
  if command -v brew >/dev/null 2>&1; then
    # Cargar entorno brew en esta sesión
    local bp
    bp="$(command -v brew)"
    eval "$("$bp" shellenv)"
    return 0
  fi
  msg "Instalando Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Post-instalación: shellenv
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  else
    err "No se pudo ubicar brew tras la instalación."
    return 1
  fi
  return 0
}

persist_path_line() {
  local line="$1"
  local file="$HOME/.zprofile"
  touch "$file"
  if ! grep -Fqs "$line" "$file"; then
    printf "%s\n" "$line" >> "$file"
  fi
}

ensure_py311() {
  # Si ya hay python3.12/3.11/3.10 preferirlos; si no, instalar 3.11 con brew
  local cand
  for cand in \
    "${PYTHON_BIN:-}" \
    "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3.11" "/opt/homebrew/bin/python3.10" \
    "/usr/local/bin/python3.12" "/usr/local/bin/python3.11" "/usr/local/bin/python3.10" \
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
        echo "$cand"
        return 0
      fi
    fi
  done

  # No hay >=3.10 → instalar python@3.11 con brew
  ensure_brew || { err "Homebrew no disponible."; return 1; }
  msg "Instalando python@3.11 (puede tardar)..."
  brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11

  local bp prefix py
  bp="$(command -v brew)"
  prefix="$("$bp" --prefix)"
  # Cargar brew para esta sesión y futuras
  eval "$("$bp" shellenv)"
  persist_path_line "eval \"\$($bp shellenv)\""
  persist_path_line "export PATH=\"$prefix/opt/python@3.11/bin:\$PATH\""
  export PATH="$prefix/opt/python@3.11/bin:$PATH"

  # Ruta final de python3.11
  if [[ -x "$prefix/opt/python@3.11/bin/python3.11" ]]; then
    py="$prefix/opt/python@3.11/bin/python3.11"
  else
    py="$(command -v python3.11 || true)"
  fi
  [[ -x "$py" ]] || { err "No se encontró python3.11 tras brew."; return 1; }
  echo "$py"
}

# ===== Selección/instalación de Python =====
PY_BIN="$(ensure_py311)" || {
  err "No se pudo preparar un Python >=3.10. Instálalo manualmente e inténtalo de nuevo."
  exit 1
}
msg "Usando Python: $PY_BIN ($("$PY_BIN" -V))"

# ===== Venv: recrear si no existe, si el intérprete cambió o si <3.10 =====
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

# ===== Instalar deps solo si cambia requirements o se reconstruyó venv =====
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
( sleep 2; open -g "http://127.0.0.1:8000" ) >/dev/null 2>&1 &

# ===== Lanzar la app =====
msg "Lanzando product_research_app..."
python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
