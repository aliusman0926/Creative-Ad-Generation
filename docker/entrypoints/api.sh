#!/usr/bin/env bash
set -euo pipefail

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

exec uvicorn src.api.main:app --host "${HOST}" --port "${PORT}" --proxy-headers
