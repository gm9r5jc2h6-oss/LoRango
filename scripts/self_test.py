#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorango_sdxl.carrier import BLOCK_SPECS, select_carriers  # noqa: E402
from lorango_sdxl.checkpoint import load_checkpoint  # noqa: E402
from lorango_sdxl.config import load_data  # noqa: E402


def main() -> None:
    assets = ROOT / "assets"
    scaffold_path = assets / "scaffold_init.safetensors"
    catalog_path = assets / "carrier_catalog.json"
    if not scaffold_path.is_file() or not catalog_path.is_file():
        raise FileNotFoundError("required package assets are missing")
    scaffold = load_checkpoint(scaffold_path)
    names = {row["name"] for row in scaffold["layers"]}
    expected = {name for name, _block in BLOCK_SPECS}
    if names != expected:
        raise AssertionError({"missing": sorted(expected - names), "extra": sorted(names - expected)})
    data = load_data(ROOT / "examples" / "train_data.example.json")
    for group in range(4):
        config = {
            "carrier_catalog": str(catalog_path),
            "carrier": {"group_index": group},
        }
        selected = select_carriers(config, scaffold)
        if len(selected) != 5 or any(len(row["channels"]) != 4 for row in selected):
            raise AssertionError(f"invalid carrier group {group}")
    result = {
        "status": "PASS",
        "scaffold_layers": len(scaffold["layers"]),
        "carrier_groups": 4,
        "eligible_layers_per_group": 5,
        "example_train_rows": len(data["train"]),
        "example_calibration_rows": len(data["calibration"]),
        "catalog_bytes": catalog_path.stat().st_size,
        "scaffold_bytes": scaffold_path.stat().st_size,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
