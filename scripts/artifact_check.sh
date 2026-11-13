#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

python scripts/verify_release.py
python scripts/self_test.py
python -m unittest discover -s tests -v
python scripts/verify_reference.py

echo "ARTIFACT_CHECK=PASS"
