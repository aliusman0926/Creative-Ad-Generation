#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "No training command provided. The environment is ready for custom MLflow runs or scripts."
  exec bash
else
  exec "$@"
fi
