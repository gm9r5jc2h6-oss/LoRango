#!/usr/bin/env bash
set -euo pipefail

: "${ARCH_CONFIG:?set ARCH_CONFIG to an architecture JSON file}"
: "${MODULE_INVENTORY:?set MODULE_INVENTORY to module_inventory.json}"
: "${PROBE_METRICS:?set PROBE_METRICS to probe_metrics.jsonl}"
: "${OUT_DIR:?set OUT_DIR to the destination directory}"

POLICY_CONFIG="${POLICY_CONFIG:-policies/controlled_v1.json}"
DATA_CONFIG="${DATA_CONFIG:-data/harmless_pairs.json}"

mkdir -p "$OUT_DIR"

python3 -m xarch plan \
  --architecture "$ARCH_CONFIG" \
  --policy "$POLICY_CONFIG" \
  --data "$DATA_CONFIG" \
  --output "$OUT_DIR/run_plan.json"

python3 -m xarch generate-candidates \
  --architecture "$ARCH_CONFIG" \
  --inventory "$MODULE_INVENTORY" \
  --output "$OUT_DIR/candidates.json"

python3 -m xarch rank \
  --policy "$POLICY_CONFIG" \
  --candidates "$OUT_DIR/candidates.json" \
  --metrics "$PROBE_METRICS" \
  --output "$OUT_DIR/ranked_candidates.json"

python3 -m xarch select \
  --policy "$POLICY_CONFIG" \
  --ranked "$OUT_DIR/ranked_candidates.json" \
  --output "$OUT_DIR/selected_topology.json"

python3 -c 'import json,os,pathlib; p=pathlib.Path(os.environ["OUT_DIR"])/"selected_topology.json"; print(json.dumps(json.loads(p.read_text()), indent=2))'

