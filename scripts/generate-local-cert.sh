#!/usr/bin/env bash
set -e
CERT_DIR="$(dirname "$0")/../certs"
mkdir -p "$CERT_DIR"
if command -v mkcert >/dev/null 2>&1; then
  mkcert -cert-file "$CERT_DIR/dev-cert.pem" -key-file "$CERT_DIR/dev-key.pem" 127.0.0.1 localhost
else
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$CERT_DIR/dev-key.pem" -out "$CERT_DIR/dev-cert.pem" -days 365 -subj "/CN=localhost"
fi
echo "Certificates generated in $CERT_DIR"
