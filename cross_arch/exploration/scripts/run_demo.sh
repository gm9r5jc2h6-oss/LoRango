#!/usr/bin/env bash
set -euo pipefail

rm -rf artifacts/demo_run
python3 -m xarch demo --output-dir artifacts/demo_run
python3 -m unittest discover -s tests -v

