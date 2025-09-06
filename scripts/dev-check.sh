#!/usr/bin/env bash
set -e
BACK_PORT=${BACK_PORT:-8000}
FRONT_PORT=${FRONT_PORT:-$BACK_PORT}
HTTPS_PORT=$((BACK_PORT+1))

echo "Checking backend over HTTP..."
curl -v http://127.0.0.1:${BACK_PORT}/health || exit 1

if [ "${DEV_HTTPS}" = "true" ]; then
  echo "Checking backend over HTTPS..."
  curl -vk https://127.0.0.1:${HTTPS_PORT}/health || exit 1
fi

echo "Checking products endpoint..."
curl -v http://127.0.0.1:${FRONT_PORT}/products || exit 1
