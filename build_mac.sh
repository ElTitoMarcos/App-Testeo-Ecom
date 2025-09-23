#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENTRYPOINT="product_research_app/web_app.py"
NAME="EcomTestingApp"
DATA_STATIC="product_research_app/static:product_research_app/static"
DATA_TEMPLATES="product_research_app/templates:product_research_app/templates"

if [[ ! -d "product_research_app/static" ]]; then
  echo "No se encontró el directorio requerido product_research_app/static." >&2
  exit 1
fi

EXTRA_ARGS=("--add-data" "$DATA_STATIC")
if [[ -d "product_research_app/templates" ]]; then
  EXTRA_ARGS+=("--add-data" "$DATA_TEMPLATES")
fi

python3 -m pip install --upgrade pyinstaller

python3 -m PyInstaller --noconfirm --clean \
  --name "$NAME" \
  --windowed \
  "${EXTRA_ARGS[@]}" \
  "$ENTRYPOINT"
