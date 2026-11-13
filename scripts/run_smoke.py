#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorango_sdxl.cli import run_training, write_evidence_manifest  # noqa: E402
from lorango_sdxl.config import load_config, load_data  # noqa: E402
from lorango_sdxl.runtime import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a one-step end-to-end functionality test; not a quality evaluation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # The reference configuration may point to an already completed quality run.
    # Smoke output is replaced below with its own fresh directory, so permit that
    # source configuration to be loaded without treating the completed run as a
    # collision.
    config = load_config(args.config, allow_completed_output=True)
    data = load_data(Path(config["data_json"]))
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"smoke output must not already exist: {output}")
    output.mkdir(parents=True)
    smoke_data = {
        "train": [data["train"][0]],
        "calibration": [data["calibration"][0]],
    }
    data_path = output / "SMOKE_DATA.json"
    write_json(data_path, smoke_data)
    config["data_json"] = str(data_path)
    config["output_dir"] = str(output)
    config["runtime"].update(
        {"height": 256, "width": 256, "inference_steps": 1, "record_steps": [0]}
    )
    config["training"].update(
        {
            "optimization_steps": 1,
            "train_seeds": [101],
            "calibration_seeds": [102],
            "checkpoint_every": 0,
            "fixed_seed": 103,
        }
    )
    config["gates"].update(
        {
            "mean_cosine_min": -1.0,
            "mean_nrmse_max": 1e9,
            "single_relative_drift_max": 1e9,
            "compile_on_failure": True,
        }
    )
    config["_config_path"] = str(output / "SMOKE_CONFIG.json")
    write_json(
        output / "SMOKE_CONFIG.json",
        {key: value for key, value in config.items() if not key.startswith("_")},
    )
    result = run_training(config, smoke_data)
    pipeline_status = result["status"]
    expected_gates = {
        "mean_cosine",
        "mean_nrmse",
        "signature_only_drift",
        "payload_only_drift",
        "loader_scaling",
        "hook_export_equivalence",
    }
    post_gates = result.get("post_export_gates") or {}
    smoke_passed = (
        pipeline_status == "PASS"
        and set(post_gates) == expected_gates
        and all(value is True for value in post_gates.values())
    )
    result["status"] = "FUNCTIONAL_SMOKE_PASS" if smoke_passed else "FUNCTIONAL_SMOKE_FAIL"
    result["quality_claim"] = False
    result["note"] = (
        "Thresholds were relaxed intentionally; this run verifies execution, "
        "export, reload, and serialization only."
    )
    write_json(output / "MANIFEST.json", result)
    write_evidence_manifest(output)
    print(
        json.dumps(
            {
                "smoke_test": "PASS" if smoke_passed else "FAIL",
                "pipeline_status_before_smoke_relabel": pipeline_status,
                "manifest_status": result["status"],
            },
            indent=2,
        )
    )
    if not smoke_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
