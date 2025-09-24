#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-mac-build"
DIST_DIR="$ROOT_DIR/dist"
OUTPUT_DIR="$DIST_DIR/macos"
APP_NAME="ProductResearchCopilot.app"
ZIP_NAME="product_research_app-macos.zip"

CONFIG_PATH="$ROOT_DIR/product_research_app/update_config.json"
CONFIG_BACKUP="$(mktemp)"
cp "$CONFIG_PATH" "$CONFIG_BACKUP"

restore_config() {
  if [[ -f "$CONFIG_BACKUP" ]]; then
    mv "$CONFIG_BACKUP" "$CONFIG_PATH"
  fi
}
trap restore_config EXIT

if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then
  python - "$CONFIG_PATH" <<'PYCODE'
import json
import os
import sys

path = sys.argv[1]
repo = os.environ.get('GITHUB_REPOSITORY')
if not repo:
    sys.exit(0)
data = {}
with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
data['repository'] = repo
with open(path, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
PYCODE
fi

echo "[build] Root: $ROOT_DIR"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$ROOT_DIR/requirements.txt" pyinstaller

rm -rf "$DIST_DIR/$APP_NAME" "$OUTPUT_DIR"
pyinstaller "$ROOT_DIR/product_research_app/mac_app.spec" --noconfirm

mkdir -p "$OUTPUT_DIR"

APP_PATH="$DIST_DIR/$APP_NAME"

if [[ -n "${MAC_CODESIGN_IDENTITY:-}" ]]; then
  echo "[sign] codesigning bundle with identity '$MAC_CODESIGN_IDENTITY'"
  CODESIGN_ARGS=(--deep --force --options runtime --sign "$MAC_CODESIGN_IDENTITY")
  if [[ -n "${MAC_CODESIGN_ENTITLEMENTS:-}" ]]; then
    CODESIGN_ARGS+=(--entitlements "$MAC_CODESIGN_ENTITLEMENTS")
  fi
  CODESIGN_ARGS+=(--timestamp)
  CODESIGN_ARGS+=("$APP_PATH")
  /usr/bin/codesign "${CODESIGN_ARGS[@]}"
fi

create_archive() {
  rm -f "$OUTPUT_DIR/$ZIP_NAME" "$OUTPUT_DIR/$ZIP_NAME.sha256"
  if command -v ditto >/dev/null 2>&1; then
    echo "[archive] using ditto to compress bundle"
    ditto -c -k --keepParent "$APP_PATH" "$OUTPUT_DIR/$ZIP_NAME"
  else
    echo "[archive] ditto not available; falling back to zip"
    pushd "$DIST_DIR" >/dev/null
    zip -r "$OUTPUT_DIR/$ZIP_NAME" "$APP_NAME"
    popd >/dev/null
  fi
  shasum -a 256 "$OUTPUT_DIR/$ZIP_NAME" > "$OUTPUT_DIR/$ZIP_NAME.sha256"
}

create_archive

notarize_with_profile() {
  echo "[notary] submitting archive with keychain profile '$MAC_NOTARIZE_PROFILE'"
  xcrun notarytool submit "$OUTPUT_DIR/$ZIP_NAME" --keychain-profile "$MAC_NOTARIZE_PROFILE" --wait
}

notarize_with_credentials() {
  echo "[notary] submitting archive with Apple ID credentials"
  xcrun notarytool submit "$OUTPUT_DIR/$ZIP_NAME" --apple-id "$MAC_NOTARIZE_APPLE_ID" --team-id "$MAC_NOTARIZE_TEAM_ID" --password "$MAC_NOTARIZE_PASSWORD" --wait
}

if [[ -n "${MAC_NOTARIZE_PROFILE:-}" || ( -n "${MAC_NOTARIZE_APPLE_ID:-}" && -n "${MAC_NOTARIZE_TEAM_ID:-}" && -n "${MAC_NOTARIZE_PASSWORD:-}" ) ]]; then
  if [[ -n "${MAC_NOTARIZE_PROFILE:-}" ]]; then
    notarize_with_profile
  else
    notarize_with_credentials
  fi
  echo "[notary] stapling ticket"
  xcrun stapler staple "$APP_PATH"
  create_archive
fi

echo "[build] macOS bundle ready: $OUTPUT_DIR/$ZIP_NAME"
