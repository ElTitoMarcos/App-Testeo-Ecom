#!/usr/bin/env bash
set -Eeuo pipefail

# —— Proyecto (puede estar en USB solo-lectura)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_ID="product_research_app"
STATE_DIR="${APP_STATE_DIR:-$HOME/Library/Application Support/$APP_ID}"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/$APP_ID}"
VENV_DIR="$STATE_DIR/.venv"
PORTABLE_DIR="$STATE_DIR/py311"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.req.sha256"
PY_PATH_FILE="$VENV_DIR/.py.path"
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Quitar cuarentena si viene de Descargas
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

# —— Detectar Python del sistema (preferir 3.12→3.11→3.10; 3.9 como último)
_candidates=()
[[ -n "${PYTHON_BIN:-}" ]] && _candidates+=("$PYTHON_BIN")
_candidates+=("/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3.11" "/opt/homebrew/bin/python3.10")
_candidates+=("/usr/local/bin/python3.12" "/usr/local/bin/python3.11" "/usr/local/bin/python3.10")
_candidates+=("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
              "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
              "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"
              "python3")

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
SYS_PY="$pick"
SYS_VER_MIN="$("$SYS_PY" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"

# —— Si ya existe Python portátil, preferirlo SIEMPRE
if [[ -x "$PORTABLE_DIR/bin/python3" ]]; then
  PY="$PORTABLE_DIR/bin/python3"
else
  PY="$SYS_PY"
fi

# —— Forzar Python >=3.10: si versión <3.10, descargar portátil 3.11 y usarlo
need_portable="$("$PY" - <<'PY'
import sys; import os
maj, min = sys.version_info[:2]
print("yes" if (maj, min) < (3,10) or os.getenv("USE_PORTABLE")=="1" else "no")
PY
)"
ensure_portable_python() {
  mkdir -p "$PORTABLE_DIR"
  local api url sum_url tmp tar
  api="https://api.github.com/repos/indygreg/python-build-standalone/releases/latest"
  url="$(curl -fsSL "$api" | grep -oE 'https://[^"]*cpython-3\.11\.[0-9]+[^"]*macos-universal2[^"]*\.tar\.(gz|xz)' | head -n1 || true)"
  if [[ -z "$url" ]]; then echo "No pude localizar Python portátil 3.11." >&2; exit 1; fi
  sum_url="${url}.sha256"
  tmp="$(mktemp -d)"; tar="$tmp/py311.tar"
  echo "Descargando Python portátil 3.11..." >&2
  curl -fL "$url" -o "$tar"
  if curl -fsLI "$sum_url" >/dev/null 2>&1; then
    curl -fsSL "$sum_url" -o "$tar.sha256" || true
    (cd "$tmp" && shasum -a 256 -c "$(basename "$tar").sha256") || echo "Aviso: no verificado."
  fi
  rm -rf "$PORTABLE_DIR"; mkdir -p "$PORTABLE_DIR"
  case "$url" in
    *.tar.gz) tar -xzf "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
    *.tar.xz) tar -xJf "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
    *)        tar -xf  "$tar" -C "$PORTABLE_DIR" --strip-components=1 ;;
  esac
  rm -rf "$tmp"
  PY="$PORTABLE_DIR/bin/python3"
}
if [[ "$need_portable" == "yes" ]]; then
  ensure_portable_python
fi

# —— Crear/Recrear venv si:
#    (a) no existe, (b) no hay binario, (c) el intérprete guardado difiere, (d) el venv es <3.10
rebuild=0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  rebuild=1
elif [[ -f "$PY_PATH_FILE" && "$(cat "$PY_PATH_FILE")" != "$PY" ]]; then
  rebuild=1
else
  "$VENV_DIR/bin/python" - <<'PY' || rebuild=1
import sys; import sysconfig
import os; assert sys.version_info[:2] >= (3,10)
PY
fi
if [[ "$rebuild" -eq 1 ]]; then
  rm -rf "$VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

# —— Activar venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
printf "%s" "$PY" > "$PY_PATH_FILE"

# —— Instalar deps sólo si cambia requirements.txt o se reconstruyó el venv
calc_hash() { [[ -f "$REQ_FILE" ]] && shasum -a 256 "$REQ_FILE" | awk '{print $1}'; }
old="$(cat "$HASH_FILE" 2>/dev/null || true)"
new="$(calc_hash || true)"
if [[ "$rebuild" -eq 1 || "$new" != "$old" ]]; then
  python -m pip install --upgrade pip wheel setuptools
  if [[ -n "$new" ]]; then
    pip install -r "$REQ_FILE"
    printf "%s" "$new" > "$HASH_FILE"
  fi
fi

export PYTHONUNBUFFERED=1

# Abrir navegador en segundo plano
( sleep 2; open -g "http://127.0.0.1:8000" ) >/dev/null 2>&1 &

# Ejecutar la app
python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
