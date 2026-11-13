#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = ROOT / "reference" / "v1_2_run"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompile the migrated reference checkpoint and repeat standard-loader calibration."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    model_dir = args.model_dir.expanduser().resolve()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_model.py"),
            "--model-dir",
            str(model_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    output.mkdir(parents=True)
    config = json.loads((ROOT / "configs" / "reference_pair2.json").read_text())
    config.update(
        {
            "base_model": str(model_dir),
            "signature_cover_lora": str(
                ROOT / "reference" / "inputs" / "utility_s.safetensors"
            ),
            "payload_cover_lora": str(
                ROOT / "reference" / "inputs" / "utility_p.safetensors"
            ),
            "data_json": str(ROOT / "reference" / "train_data.json"),
            "scaffold_init": str(REFERENCE_RUN / "checkpoint_trained.safetensors"),
            "carrier_catalog": str(ROOT / "assets" / "carrier_catalog.json"),
            "output_dir": str(output),
        }
    )
    config_path = output / "REPRODUCE_CONFIG.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_pair.py"),
            "--config",
            str(config_path),
            "--compile-only",
            str(REFERENCE_RUN / "checkpoint_trained.safetensors"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output / "RUN_METADATA.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((ROOT / "reference" / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    model_identity = run_metadata["inputs"]["base_model"]
    model_identity_match = (
        model_identity.get("weights_digest") == model_manifest["weights_digest"]
        and model_identity.get("metadata_digest") == model_manifest["metadata_digest"]
    )
    comparisons = {}
    for name in ("modified_s.safetensors", "modified_p.safetensors"):
        expected = sha256(REFERENCE_RUN / name)
        actual = sha256(output / name)
        comparisons[name] = {"expected": expected, "actual": actual, "match": expected == actual}
    passed = (
        manifest["status"] == "PASS_COMPILE_ONLY_POST_EXPORT"
        and all((manifest.get("post_export_gates") or {}).values())
        and all(row["match"] for row in comparisons.values())
        and model_identity_match
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "byte_identical_deployable_outputs": comparisons,
        "post_export_gates": manifest.get("post_export_gates"),
        "model_identity_match": model_identity_match,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
