#!/bin/bash
# Load repo .env into the shell. Sourced by scripts/_env.sh and setup scripts.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${_ROOT}/.env"
  set +a
fi
