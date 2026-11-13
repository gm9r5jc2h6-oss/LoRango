#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CONFIG_JSON [LOG_FILE]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
LOG="${2:-lorango_train_$(date -u +%Y%m%dT%H%M%SZ).log}"

cd "$ROOT"
set -o pipefail
python3 train_pair.py --config "$CONFIG" 2>&1 | tee "$LOG"
