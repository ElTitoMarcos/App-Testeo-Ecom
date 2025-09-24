#!/usr/bin/env bash
set -Eeuo pipefail

# — Proyecto (puede estar en USB solo-lectura)
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

xattr -d com.apple.quarantine "$0" 2>/dev/null || true

# — Detectar Python del sistema (pref. 3.12→3.11→3.10; 3.9 como último)
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
[[ -z "$pick" ]] && { echo "No se encontró Python 3 en el sistema." >&2; exit 1; }
SYS_PY="$pick"

# — Preferir portátil si ya existe
if [[ -x "$PORTABLE_DIR/bin/python3" ]]; then PY="$PORTABLE_DIR/bin/python3"; else PY="$SYS_PY"; fi

# — ¿Necesitamos ≥3.10?
need_new_py="$("$PY" - <<'PY'
import sys, os
maj, min = sys.version_info[:2]
print("yes" if (maj, min) < (3,10) or os.getenv("USE_PORTABLE")=="1" else "no")
PY
)"

# — Función: descargar Python 3.11 portátil (multi-fuente) y usarlo
ensure_portable_python() {
  mkdir -p "$PORTABLE_DIR"
  local url="" sum_url="" tmp tar
  tmp="$(mktemp -d)"; tar="$tmp/py311.tar"
  # 1) API JSON
  url="$(curl -fsSL -H 'User-Agent: curl' \
        https://api.github.com/repos/indygreg/python-build-standalone/releases/latest \
        | grep -oE 'https://[^"]*cpython-3\.11\.[0-9]+[^"]*macos-universal2[^"]*\.tar\.(gz|xz)' \
        | head -n1 || true)"
  # 2) HTML latest
  if [[ -z "$url" ]]; then
    url="$(curl -fsSL -H 'User-Agent: curl' \
          https://github.com/indygreg/python-build-standalone/releases/latest \
          | grep -oE '/indygreg/python-build-standalone/releases/download/[^"]*cpython-3\.11\.[0-9]+[^"]*macos-universal2[^"]*\.tar\.(gz|xz)' \
          | head -n1 | awk '{print "https://github.com"$0}' || true)"
  fi
  # 3) HTML releases (por si latest no aparece)
  if [[ -z "$url" ]]; then
    url="$(curl -fsSL -H 'User-Agent: curl' \
          https://github.com/indygreg/python-build-standalone/releases \
          | grep -oE '/indygreg/python-build-standalone/releases/download/[^"]*cpython-3\.11\.[0-9]+[^"]*macos-universal2[^"]*\.tar\.(gz|xz)' \
          | head -n1 | awk '{print "https://github.com"$0}' || true)"
  fi

  if [[ -z "$url" ]]; then
    echo "No pude localizar Python portátil 3.11 desde GitHub." >&2
    return 1
  fi

  sum_url="${url}.sha256"
  echo "Descargando Python portátil 3.11..." >&2
  curl -fL --retry 3 --connect-timeout 10 "$url" -o "$tar"
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

# — Si hace falta, intenta portátil; si también falla y hay brew, usa Homebrew
if [[ "$need_new_py" == "yes" ]]; then
  if ! ensure_portable_python; then
    if command -v brew >/dev/null 2>&1; then
      echo "Instalando python@3.11 con Homebrew..." >&2
      brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
      # Detectar ruta de brew
      for b in /opt/homebrew /usr/local; do
        if [[ -x "$b/bin/python3.11" ]]; then PY="$b/bin/python3.11"; break; fi
      done
    else
      echo "No hay Python ≥3.10 ni portátil ni Homebrew. Instálalo y reintenta." >&2
      exit 1
    fi
  fi
fi

# — Re-crear venv si: no existe, no hay binario, cambió el intérprete o es <3.10
rebuild=0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  rebuild=1
elif [[ -f "$PY_PATH_FILE" && "$(cat "$PY_PATH_FILE")" != "$PY" ]]; then
  rebuild=1
else
  "$VENV_DIR/bin/python" - <<'PY' || rebuild=1
import sys
assert sys.version_info[:2] >= (3,10)
PY
fi
if [[ "$rebuild" -eq 1 ]]; then
  rm -rf "$VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

# — Activar venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
printf "%s" "$PY" > "$PY_PATH_FILE"

# — Instalar deps solo si cambia requirements.txt o se reconstruyó el venv
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
( sleep 2; open -g "http://127.0.0.1:8000" ) >/dev/null 2>&1 &

python -u -m product_research_app 2>&1 | tee "$LOG_DIR/session.log"
