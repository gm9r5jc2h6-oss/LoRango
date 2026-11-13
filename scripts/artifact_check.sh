#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

python3 scripts/verify_release.py
python3 scripts/self_test.py
python3 -m unittest discover -s tests -v
python3 scripts/verify_reference.py

echo "ARTIFACT_CHECK=PASS"
