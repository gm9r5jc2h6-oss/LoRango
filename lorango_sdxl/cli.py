from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import __version__
from .carrier import (
    compile_pair,
    cover_rank_profile,
    select_carriers,
    validate_cover_collisions,
)
from .config import load_config, load_data
from .checkpoint import load_checkpoint, metadata_path
from .provenance import build_run_metadata
from .runtime import load_pipe, sha256, write_json
from .training import (
    audit_exported_pair,
    cleanup_pipe,
    configure_determinism,
    train_head,
)


def _protocol(config: dict[str, Any], data: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    safe_config = json.loads(
        json.dumps({key: value for key, value in config.items() if not key.startswith("_")})
    )
    return {
        "toolkit": "LoRango SDXL Pair-Training Toolkit",
        "version": __version__,
        "method": (
            "multi-layer native-GEGLU carrier with per-layer constant gating, "
            "native value cancellation, and native output subtraction"
        ),
        "config": safe_config,
        "data": data,
        "train_count": len(data["train"]),
        "calibration_count": len(data["calibration"]),
        "validation_or_sealed_data_used": False,
        "status": "FROZEN_BEFORE_TRAINING",
    }


def _write_input_snapshots(
    output_dir: Path,
    config: dict[str, Any],
    data: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze normalized, execution-effective inputs before model execution."""

    protocol = _protocol(config, data)
    write_json(output_dir / "INPUT_CONFIG.json", protocol["config"])
    write_json(output_dir / "INPUT_DATA.json", protocol["data"])
    model_manifest = (
        json.loads(Path(config["model_manifest"]).read_text(encoding="utf-8"))
        if config.get("model_manifest")
        else {"status": "NOT_DECLARED"}
    )
    write_json(output_dir / "INPUT_MODEL_MANIFEST.json", model_manifest)
    write_json(output_dir / "PROTOCOL.json", protocol)
    return protocol, model_manifest


def validate_only(config: dict[str, Any], data: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    scaffold = load_checkpoint(config["scaffold_init"])
    selections = select_carriers(config, scaffold)
    validate_cover_collisions(config, scaffold)
    carrier_ranks = [len(row["channels"]) for row in selections]
    signature_rank_profile = cover_rank_profile(
        Path(config["signature_cover_lora"]),
        [rank + 1 for rank in carrier_ranks] + carrier_ranks,
    )
    payload_rank_profile = cover_rank_profile(
        Path(config["payload_cover_lora"]), carrier_ranks + carrier_ranks
    )
    warnings = []
    if len(data["train"]) < 8:
        warnings.append("fewer than 8 training prompts; cross-scene generalization may be weak")
    if len(data["calibration"]) < 2:
        warnings.append("fewer than 2 calibration prompts; gate estimates will be noisy")
    result = {
        "status": "VALID",
        "train_examples": len(data["train"]),
        "calibration_examples": len(data["calibration"]),
        "train_cases": len(data["train"]) * len(config["training"]["train_seeds"]),
        "calibration_cases": len(data["calibration"])
        * len(config["training"]["calibration_seeds"]),
        "carrier_group": int(config["carrier"]["group_index"]),
        "carrier_layers": [
            {"name": row["name"], "block": row["block"], "channels": row["channels"]}
            for row in selections
        ],
        "cover_files": {
            "signature": {
                "path": config["signature_cover_lora"],
                "sha256": sha256(Path(config["signature_cover_lora"])),
            },
            "payload": {
                "path": config["payload_cover_lora"],
                "sha256": sha256(Path(config["payload_cover_lora"])),
            },
        },
        "loader_scaling": {
            "signature": signature_rank_profile,
            "payload": payload_rank_profile,
        },
        "warnings": warnings,
    }
    return result


def _write_manifest(
    config: dict[str, Any],
    status: str,
    checkpoint: Path,
    calibration: dict[str, Any] | None,
    gates: dict[str, bool] | None,
    compilation: dict[str, Any] | None,
    post_export_calibration: dict[str, Any] | None,
    post_export_gates: dict[str, bool] | None,
) -> dict[str, Any]:
    result = {
        "toolkit": "LoRango SDXL Pair-Training Toolkit",
        "toolkit_version": __version__,
        "status": status,
        "ordinary_static_lora": compilation is not None,
        "external_hook_at_inference": False,
        "external_detector_at_inference": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "metadata_path": str(metadata_path(checkpoint)),
            "metadata_sha256": sha256(metadata_path(checkpoint)),
            "deployable": False,
        },
        "gates": gates,
        "post_export_gates": post_export_gates,
        "calibration_summary": None
        if calibration is None
        else {key: value for key, value in calibration.items() if key != "rows"},
        "post_export_calibration_summary": None
        if post_export_calibration is None
        else {
            key: value
            for key, value in post_export_calibration.items()
            if key != "rows"
        },
        "compilation": compilation,
    }
    write_json(Path(config["output_dir"]) / "MANIFEST.json", result)
    write_evidence_manifest(Path(config["output_dir"]))
    return result


def write_evidence_manifest(output_dir: Path) -> dict[str, Any]:
    """Seal every top-level run artifact after MANIFEST.json is finalized."""

    output_dir = output_dir.expanduser().resolve()
    symlinks = [path.name for path in output_dir.iterdir() if path.is_symlink()]
    if symlinks:
        raise ValueError(f"run output may not contain symlinks: {sorted(symlinks)}")
    non_files = [path.name for path in output_dir.iterdir() if not path.is_file()]
    if non_files:
        raise ValueError(f"run output must be flat and file-only: {sorted(non_files)}")
    files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "EVIDENCE.json"
    }
    evidence = {
        "format": "lorango-run-evidence-v2",
        "toolkit_version": __version__,
        "files": files,
        "files_digest": hashlib.sha256(
            json.dumps(
                files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(output_dir / "EVIDENCE.json", evidence)
    return evidence


def _add_export_equivalence(
    calibration: dict[str, Any],
    exported: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    fields = (
        "cosine",
        "nrmse",
        "amplitude_ratio",
        "s_relative_drift",
        "p_relative_drift",
    )
    before = {
        (row["id"], int(row["seed"]), int(row["step"])): row
        for row in calibration["rows"]
    }
    after = {
        (row["id"], int(row["seed"]), int(row["step"])): row
        for row in exported["rows"]
    }
    if set(before) != set(after):
        exported["hook_export_equivalence"] = {
            "passed": False,
            "reason": "calibration row identities differ",
        }
        return False
    per_field = {
        field: max(abs(float(before[key][field]) - float(after[key][field])) for key in before)
        for field in fields
    }
    maximum = max(per_field.values(), default=0.0)
    threshold = float(config["gates"]["export_equivalence_max"])
    passed = maximum <= threshold
    exported["hook_export_equivalence"] = {
        "passed": passed,
        "threshold": threshold,
        "max_absolute_metric_delta": maximum,
        "per_metric_max_absolute_delta": per_field,
        "rows_compared": len(before),
    }
    return passed


def run_training(config: dict[str, Any], data: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_determinism(config)
    _, model_manifest = _write_input_snapshots(output_dir, config, data)
    write_json(
        output_dir / "RUN_METADATA.json",
        build_run_metadata(config, "train", model_manifest),
    )
    pipe = load_pipe(config)
    try:
        checkpoint, calibration, gates = train_head(pipe, config, data)
        passed = all(gates.values())
        compilation = None
        post_export_calibration = None
        post_export_gates = None
        status = "PASS" if passed else "TRAINING_GATE_FAILED"
        if passed or bool(config["gates"].get("compile_on_failure", False)):
            pipe.unload_lora_weights()
            compilation = compile_pair(pipe, checkpoint, config)
            post_export_calibration, post_export_gates = audit_exported_pair(
                pipe, config, data
            )
            post_export_gates["hook_export_equivalence"] = _add_export_equivalence(
                calibration, post_export_calibration, config
            )
            write_json(
                output_dir / "POST_EXPORT_CALIBRATION.json",
                post_export_calibration,
            )
            post_passed = all(post_export_gates.values())
            if passed and not post_passed:
                status = "POST_EXPORT_GATE_FAILED"
            elif not passed:
                status = "COMPILED_WITH_FAILED_TRAINING_GATES"
        return _write_manifest(
            config,
            status,
            checkpoint,
            calibration,
            gates,
            compilation,
            post_export_calibration,
            post_export_gates,
        )
    finally:
        cleanup_pipe(pipe)


def run_compile_only(
    config: dict[str, Any],
    data: dict[str, list[dict[str, str]]],
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if checkpoint.suffix != ".safetensors" or not metadata_path(checkpoint).is_file():
        raise ValueError("--compile-only requires a LoRango .safetensors checkpoint and JSON sidecar")
    configure_determinism(config)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    _, model_manifest = _write_input_snapshots(
        Path(config["output_dir"]), config, data
    )
    write_json(
        Path(config["output_dir"]) / "RUN_METADATA.json",
        build_run_metadata(config, "compile_only", model_manifest),
    )
    state = load_checkpoint(checkpoint)
    validate_cover_collisions(config, state)
    pipe = load_pipe(config)
    try:
        compilation = compile_pair(pipe, checkpoint, config)
        post_export_calibration, post_export_gates = audit_exported_pair(
            pipe, config, data
        )
        status = (
            "PASS_COMPILE_ONLY_POST_EXPORT"
            if all(post_export_gates.values())
            else "POST_EXPORT_GATE_FAILED"
        )
        return _write_manifest(
            config,
            status,
            checkpoint,
            None,
            None,
            compilation,
            post_export_calibration,
            post_export_gates,
        )
    finally:
        cleanup_pipe(pipe)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compile one matched SDXL dual-LoRA pair."
    )
    parser.add_argument("--config", type=Path, required=True, help="training JSON config")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs and carrier layout without loading SDXL",
    )
    parser.add_argument(
        "--compile-only",
        type=Path,
        metavar="CHECKPOINT",
        help="skip optimization and compile an existing checkpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only and args.compile_only:
        raise SystemExit("--validate-only and --compile-only are mutually exclusive")
    config = load_config(args.config)
    data = load_data(Path(config["data_json"]))
    if args.validate_only:
        result = validate_only(config, data)
    elif args.compile_only:
        result = run_compile_only(config, data, args.compile_only)
    else:
        result = run_training(config, data)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
