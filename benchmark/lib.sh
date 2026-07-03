#!/usr/bin/env bash
# Shared helpers for the benchmark scripts.
set -euo pipefail

log() { printf '[bench] %s\n' "$*" >&2; }

require_file() { [ -f "$1" ] || { log "missing file: $1"; exit 1; }; }
