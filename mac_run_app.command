#!/usr/bin/env bash
set -Eeuo pipefail

# ——— Localización del proyecto (puede estar en USB solo-lectura)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_ID="product_research_app"
STATE_DIR="${APP_STATE_DIR:-$HOME/Library/Application Support/$APP_ID}"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/$APP_ID}"
VENV_DIR="$STATE_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.req.sha256"
PORTABLE_DIR="$STATE_DIR/py311"         # aquí vivirá el Python portátil si hace falta
mkdir -p "$STATE_DIR" "$LOG_DIR"

# ——— Quita cuarentena si viene de Descargas
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

# ——— Detección de Python preferido (3.12→3.11→3.10; 3.9 como último recurso)
_candidates=()
[[ -n "${PYTHON_BIN:-}" ]] && _candidates+=("$PYTHON_BIN")
_candidates+=("/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3.11" "/opt/homebrew/bin/python3.10")
_candidates+=("/usr/local/bin/python3.12" "/usr/local/bin/python3.11" "/usr/local/bin/python3.10")
_candidates+=("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
              "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
              "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" "python3")

pick=""
for p in "${_candidates[@]}"; do
  if command -v "$p" >/dev/null 2>&1; then pick="$(command -v "$p")"; break; fi
done
if [[ -z "$pick" ]]; then
  cat >&2 <<'MSG'
No se encontró Python 3. Instala Homebrew y python@3.11:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install python@3.11
MSG
  exit 1
fi
PY="$pick"

# ——— Versión detectada
VER_STR="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
MAJ=$("$PY" -c 'import sys; print(sys.version_info[0])')
MIN=$("$PY" -c 'import sys; print(sys.version_info[1])')

# ——— Si existe Python portátil previo, preferirlo siempre
if [[ -x "$PORTABLE_DIR/bin/python3" ]]; then
  PY="$PORTABLE_DIR/bin/python3"
  VER_STR="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  MAJ=$("$PY" -c 'import sys; print(sys.version_info[0])')
  MIN=$("$PY" -c 'import sys; print(sys.version_info[1])')
fi

# ——— Función: descargar Python portátil 3.11 universal2 y usarlo
ensure_portable_python() {
  mkdir -p "$PORTABLE_DIR"
  local api url sum_url tar tmp
  api="https://api.github.com/repos/indygreg/python-build-standalone/releases/latest"
  # Buscar asset cpython-3.11.* macos-universal2 (tar.gz o tar.xz)
  url="$(curl -fsSL "$api" | grep -oE 'https://[^"]*cpython-3\.11\.[0-9]+[^"]*macos-universal2[^"]*\.tar\.(gz|xz)' | head -n1 || true)"
  if [[ -z "$url" ]]; then
    echo "No pude resolver el Python portátil 3.11 desde GitHub." >&2
    return 1
  fi
  sum_url="${url}.sha256"
  tmp="$(mktemp -d)"
  tar="$tmp/py311.tar"
  echo "Descargando Python portátil 3.11..." >&2
  curl -fL "$url" -o "$tar"
  # Verificación opcional
  if curl -fsLI "$sum_url" >/dev/null 2>&1; then
    curl -fsSL "$sum_url" -o "$tar.sha256"
    (cd "$tmp" && shasum -a 256 -c "$(basename "$tar").sha256") || echo "Aviso: no pude verificar checksum."
  fi
  rm -rf "$PORTABLE_DIR"
  mkdir -p "$PORTABLE_DIR"
  case "$url" in
    *.tar.gz) tar -xzf "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
    *.tar.xz) tar -xJf "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
    *)        tar -xf  "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
  esac
  rm -rf "$tmp"
  PY="$PORTABLE_DIR/bin/python3"
  VER_STR="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  MAJ=$("$PY" -c 'import sys; print(sys.version_info[0])')
  MIN=$("$PY" -c 'import sys; print(sys.version_info[1])')
}

# ——— Crear / recrear venv con el Python elegido
create_or_use_venv() {
  if [[ ! -d "$VENV_DIR" ]] || [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
}

# ——— Instalar deps sólo si cambia requirements; reintentar con Python portátil si falla en 3.9
install_deps_if_needed() {
  local need=0
  if [[ -f "$REQ_FILE" ]]; then
    local new old
    new="$(shasum -a 256 "$REQ_FILE" | awk '{print $1}')"
    old="$(cat "$HASH_FILE" 2>/dev/null || true)"
    [[ "$new" != "$old" ]] && need=1
  fi
  if [[ "$need" -eq 1 ]]; then
    python -m pip install --upgrade pip
    if ! pip install -r "$REQ_FILE"; then
      # Si estamos en 3.9, intentar con Python portátil 3.11
      if (( MAJ==3 && MIN==9 )); then
        echo "Fallo al instalar con Python $VER_STR. Intentando Python portátil 3.11..." >&2
        ensure_portable_python || { echo "No se pudo obtener Python 3.11."; exit 1; }
        "$PY" -m venv "$VENV_DIR" --clear
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
        python -m pip install --upgrade pip
        pip install -r "$REQ_FILE"
      else
        exit 1
      fi
    fi
    shasum -a 256 "$REQ_FILE" | awk '{print $1}' > "$HASH_FILE"
  fi
}

# ——— Flujo principal
create_or_use_venv
install_deps_if_needed

export PYTHONUNBUFFERED=1
( sleep 2; open -g "http://127.0.0.1:8000" ) >/dev/null 2>&1 &

# ——— Ejecutar la app (desde el directorio del proyecto)
python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
