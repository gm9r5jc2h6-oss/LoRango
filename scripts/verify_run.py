#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


VERSION = "1.4.1"
SUPPORTED_RUN_VERSIONS = {"1.4.0", VERSION}
COMPILED_FILES = {
    "signature.safetensors",
    "payload.safetensors",
    "modified_s.safetensors",
    "modified_p.safetensors",
}
HOOK_GATES = {
    "mean_cosine",
    "mean_nrmse",
    "signature_only_drift",
    "payload_only_drift",
}
POST_GATES = HOOK_GATES | {"loader_scaling", "hook_export_equivalence"}
REQUIRED = {
    "INPUT_CONFIG.json",
    "INPUT_DATA.json",
    "INPUT_MODEL_MANIFEST.json",
    "PROTOCOL.json",
    "RUN_METADATA.json",
    "training_trace.json",
    "checkpoint_trained.safetensors",
    "checkpoint_trained.json",
    "CALIBRATION.json",
    "POST_EXPORT_CALIBRATION.json",
    "signature.safetensors",
    "payload.safetensors",
    "modified_s.safetensors",
    "modified_p.safetensors",
    "COMPILE.json",
    "MANIFEST.json",
    "EVIDENCE.json",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECTED_LAYERS = {
    "d0_t8": "down_blocks.2.attentions.0.transformer_blocks.8",
    "d1_t1": "down_blocks.2.attentions.1.transformer_blocks.1",
    "u0a0_t1": "up_blocks.0.attentions.0.transformer_blocks.1",
    "u0a2_t1": "up_blocks.0.attentions.2.transformer_blocks.1",
    "u0a2_t7": "up_blocks.0.attentions.2.transformer_blocks.7",
}
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
MAX_SAFETENSORS_HEADER_BYTES = 100_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, failures: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid JSON {path.name}: {exc}")
        return None


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def close(actual: object, expected: float, label: str, failures: list[str]) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        failures.append(f"{label} is not numeric")
        return
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        failures.append(f"{label} mismatch: recorded={actual!r}, recomputed={expected!r}")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_safetensors(
    path: Path, failures: list[str]
) -> dict[str, dict[str, Any]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError("truncated length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if (
                header_length < 2
                or header_length > MAX_SAFETENSORS_HEADER_BYTES
                or 8 + header_length > size
            ):
                raise ValueError("invalid header length")
            header = json.loads(
                stream.read(header_length).decode("utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("invalid safetensors metadata")
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if not tensors:
            raise ValueError("contains no tensors")
        data_size = size - 8 - header_length
        intervals = []
        for name, record in tensors.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ValueError("invalid tensor entry")
            if set(record) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"invalid fields for tensor {name}")
            offsets = record["data_offsets"]
            shape = record["shape"]
            dtype = record["dtype"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[0] > offsets[1]
                or offsets[1] > data_size
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or dtype not in DTYPE_BYTES
            ):
                raise ValueError(f"invalid tensor bounds for {name}")
            expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
            if offsets[1] - offsets[0] != expected_bytes:
                raise ValueError(f"tensor byte length does not match dtype/shape: {name}")
            intervals.append((offsets[0], offsets[1], name))
        ordered = sorted(intervals)
        expected_offset = 0
        for start, end, name in ordered:
            if start != expected_offset:
                raise ValueError(f"non-contiguous tensor data before {name}")
            expected_offset = end
        if expected_offset != data_size:
            raise ValueError("tensor data does not cover the complete file")
        return tensors
    except Exception as exc:
        failures.append(f"invalid safetensors {path.name}: {exc}")
        return {}


def verify_checkpoint(
    run_dir: Path, manifest: dict[str, Any], failures: list[str]
) -> list[dict[str, Any]]:
    tensor_path = run_dir / "checkpoint_trained.safetensors"
    sidecar = run_dir / "checkpoint_trained.json"
    tensors = validate_safetensors(tensor_path, failures)
    keys = set(tensors)
    metadata = read_json(sidecar, failures)
    if not isinstance(metadata, dict):
        return []
    if metadata.get("format") != "lorango-checkpoint-v1":
        failures.append("checkpoint metadata has the wrong format")
    if metadata.get("tensor_file") != tensor_path.name:
        failures.append("checkpoint metadata tensor_file mismatch")
    if metadata.get("tensor_sha256") != sha256(tensor_path):
        failures.append("checkpoint metadata tensor hash mismatch")
    layers = metadata.get("layers")
    if not isinstance(layers, list) or len(layers) != 5:
        failures.append("trained checkpoint does not contain exactly five layers")
    else:
        expected = set()
        names = set()
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                failures.append(f"checkpoint layer {index} is not an object")
                continue
            name = layer.get("name")
            fields = layer.get("tensor_fields")
            block = layer.get("block")
            channels = layer.get("channels")
            normalizers = layer.get("action_normalizers")
            if not isinstance(name, str) or not name or name in names:
                failures.append(f"invalid checkpoint layer name at index {index}")
            names.add(name)
            if name not in EXPECTED_LAYERS or block != EXPECTED_LAYERS.get(name):
                failures.append(f"checkpoint layer {index} topology mismatch")
            if (
                not isinstance(channels, list)
                or len(channels) != 4
                or len(set(channels)) != 4
                or not all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value < 5120
                    for value in channels
                )
            ):
                failures.append(f"checkpoint layer {index} channels are invalid")
            if (
                not isinstance(normalizers, list)
                or len(normalizers) != 4
                or not all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in normalizers
                )
            ):
                failures.append(f"checkpoint layer {index} normalizers are invalid")
            if not isinstance(fields, list) or set(fields) != {"actions", "value_readers"}:
                failures.append(f"checkpoint layer {index} tensor_fields mismatch")
                continue
            expected.update(f"layers.{index}.{field}" for field in fields)
        if keys != expected:
            failures.append("checkpoint safetensors key set does not match metadata")
        for index in range(5):
            expected_shapes = {
                f"layers.{index}.actions": [1280, 4],
                f"layers.{index}.value_readers": [4, 1280],
            }
            for key, shape in expected_shapes.items():
                record = tensors.get(key)
                if not isinstance(record, dict) or record.get("dtype") != "F32" or record.get("shape") != shape:
                    failures.append(f"checkpoint tensor dtype/shape mismatch: {key}")
    record = manifest.get("checkpoint")
    if not isinstance(record, dict):
        failures.append("manifest checkpoint record is absent")
    else:
        if record.get("sha256") != sha256(tensor_path):
            failures.append("manifest checkpoint hash mismatch")
        if record.get("metadata_sha256") != sha256(sidecar):
            failures.append("manifest checkpoint metadata hash mismatch")
        if record.get("deployable") is not False:
            failures.append("checkpoint must be marked non-deployable")
    return layers if isinstance(layers, list) else []


def recompute_calibration(
    record: object, label: str, failures: list[str]
) -> dict[str, float] | None:
    if not isinstance(record, dict) or not isinstance(record.get("rows"), list):
        failures.append(f"{label} does not contain rows")
        return None
    rows = record["rows"]
    if not rows:
        failures.append(f"{label} rows are empty")
        return None
    fields = {
        "cosine",
        "nrmse",
        "amplitude_ratio",
        "s_relative_drift",
        "p_relative_drift",
    }
    identities = set()
    columns = {field: [] for field in fields}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"{label} row {index} is not an object")
            continue
        identity = (row.get("id"), row.get("seed"), row.get("step"))
        if identity in identities:
            failures.append(f"{label} has a duplicate row identity: {identity!r}")
        identities.add(identity)
        for field in fields:
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                failures.append(f"{label} row {index} has invalid {field}")
            else:
                columns[field].append(float(value))
    if any(len(values) != len(rows) for values in columns.values()):
        return None
    summary = {
        "items": float(len(rows)),
        "mean_cosine": sum(columns["cosine"]) / len(rows),
        "p10_cosine": quantile(columns["cosine"], 0.10),
        "mean_nrmse": sum(columns["nrmse"]) / len(rows),
        "p90_nrmse": quantile(columns["nrmse"], 0.90),
        "mean_amplitude_ratio": sum(columns["amplitude_ratio"]) / len(rows),
        "s_mean_relative_drift": sum(columns["s_relative_drift"]) / len(rows),
        "p_mean_relative_drift": sum(columns["p_relative_drift"]) / len(rows),
    }
    for key, value in summary.items():
        close(record.get(key), value, f"{label}.{key}", failures)
    return summary


def recompute_gates(
    summary: dict[str, float], thresholds: dict[str, Any]
) -> dict[str, bool]:
    drift = float(thresholds["single_relative_drift_max"])
    return {
        "mean_cosine": summary["mean_cosine"] >= float(thresholds["mean_cosine_min"]),
        "mean_nrmse": summary["mean_nrmse"] <= float(thresholds["mean_nrmse_max"]),
        "signature_only_drift": summary["s_mean_relative_drift"] <= drift,
        "payload_only_drift": summary["p_mean_relative_drift"] <= drift,
    }


def verify_equivalence(
    hook: dict[str, Any], post: dict[str, Any], threshold: float, failures: list[str]
) -> bool:
    fields = {
        "cosine",
        "nrmse",
        "amplitude_ratio",
        "s_relative_drift",
        "p_relative_drift",
    }
    before = {
        (row.get("id"), row.get("seed"), row.get("step")): row
        for row in hook.get("rows", [])
        if isinstance(row, dict)
    }
    after = {
        (row.get("id"), row.get("seed"), row.get("step")): row
        for row in post.get("rows", [])
        if isinstance(row, dict)
    }
    if set(before) != set(after) or not before:
        failures.append("hook/export calibration identities differ")
        return False
    maximum = max(
        abs(float(before[key][field]) - float(after[key][field]))
        for key in before
        for field in fields
    )
    record = post.get("hook_export_equivalence")
    if not isinstance(record, dict):
        failures.append("post calibration lacks hook_export_equivalence")
        return False
    close(
        record.get("max_absolute_metric_delta"),
        maximum,
        "hook_export_equivalence.maximum",
        failures,
    )
    if record.get("rows_compared") != len(before):
        failures.append("hook_export_equivalence rows_compared mismatch")
    close(record.get("threshold"), threshold, "hook_export_equivalence.threshold", failures)
    passed = maximum <= threshold
    if record.get("passed") is not passed:
        failures.append("hook_export_equivalence passed flag mismatch")
    return passed


def _expected_carrier_shapes(signature: bool) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for block in EXPECTED_LAYERS.values():
        prefix = f"unet.{block}"
        projection_rank = 5 if signature else 4
        result.update(
            {
                f"{prefix}.ff.net.0.proj.lora.down.weight": [projection_rank, 1280],
                f"{prefix}.ff.net.0.proj.lora.up.weight": [10240, projection_rank],
                f"{prefix}.ff.net.2.lora.down.weight": [4, 5120],
                f"{prefix}.ff.net.2.lora.up.weight": [1280, 4],
            }
        )
    return result


def verify_lora_tensors(
    filename: str,
    tensors: dict[str, dict[str, Any]],
    compilation: dict[str, Any],
    failures: list[str],
) -> None:
    signature = filename in {"signature.safetensors", "modified_s.safetensors"}
    modified = filename.startswith("modified_")
    expected_carrier = _expected_carrier_shapes(signature)
    if modified:
        missing = set(expected_carrier) - set(tensors)
        if missing:
            failures.append(f"{filename} lacks carrier tensors: {sorted(missing)}")
    elif set(tensors) != set(expected_carrier):
        failures.append(f"{filename} carrier tensor set is not exact")
    for key, shape in expected_carrier.items():
        record = tensors.get(key)
        if not isinstance(record, dict):
            continue
        if record.get("dtype") != "F32" or record.get("shape") != shape:
            failures.append(f"{filename} carrier dtype/shape mismatch: {key}")

    down_suffix = ".lora.down.weight"
    up_suffix = ".lora.up.weight"
    down = {key[: -len(down_suffix)]: value for key, value in tensors.items() if key.endswith(down_suffix)}
    up = {key[: -len(up_suffix)]: value for key, value in tensors.items() if key.endswith(up_suffix)}
    recognized = {
        key
        for key in tensors
        if key.endswith(down_suffix) or key.endswith(up_suffix)
    }
    if set(down) != set(up) or len(recognized) != len(tensors):
        failures.append(f"{filename} does not contain exact paired LoRA tensors")
        return
    if any(not key.startswith("unet.") for key in tensors):
        failures.append(f"{filename} contains non-UNet tensors")
    rank_counts: Counter[int] = Counter()
    for prefix, record in down.items():
        paired = up[prefix]
        down_shape = record.get("shape")
        up_shape = paired.get("shape")
        if (
            record.get("dtype") != "F32"
            or paired.get("dtype") != "F32"
            or not isinstance(down_shape, list)
            or len(down_shape) != 2
            or not isinstance(up_shape, list)
            or len(up_shape) != 2
            or down_shape[0] != up_shape[1]
            or down_shape[0] < 1
        ):
            failures.append(f"{filename} has an invalid LoRA pair: {prefix}")
            continue
        rank_counts[int(down_shape[0])] += 1

    if modified:
        branch = "signature" if signature else "payload"
        loader_scaling = compilation.get("loader_scaling")
        profile = loader_scaling.get(branch) if isinstance(loader_scaling, dict) else None
        if not isinstance(profile, dict):
            failures.append(f"compilation lacks {branch} loader-scaling profile")
            return
        recorded = profile.get("merged_rank_counts")
        actual = {str(key): value for key, value in sorted(rank_counts.items())}
        if recorded != actual:
            failures.append(f"{filename} merged rank profile mismatch")


def verify_loader_scaling(
    post: dict[str, Any], compilation: dict[str, Any], failures: list[str]
) -> bool:
    audit = post.get("loader_scaling_audit")
    profiles = compilation.get("loader_scaling")
    if not isinstance(audit, dict) or not isinstance(profiles, dict):
        failures.append("loader-scaling audit/profile is absent")
        return False
    expected: dict[str, float] = {}
    for branch in ("signature", "payload"):
        profile = profiles.get(branch)
        alpha = profile.get("inferred_loader_alpha") if isinstance(profile, dict) else None
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0:
            failures.append(f"invalid {branch} inferred loader alpha")
            return False
        expected[branch] = float(alpha)
    if audit.get("expected_modified_s_alpha") != expected["signature"]:
        failures.append("loader audit signature alpha does not match compilation")
    if audit.get("expected_modified_p_alpha") != expected["payload"]:
        failures.append("loader audit payload alpha does not match compilation")
    actual = audit.get("actual")
    if not isinstance(actual, dict) or set(actual) != {
        "clean_s",
        "clean_p",
        "modified_s",
        "modified_p",
    }:
        failures.append("loader audit adapter set is invalid")
        passed = False
    else:
        expected_by_adapter = {
            "clean_s": expected["signature"],
            "modified_s": expected["signature"],
            "clean_p": expected["payload"],
            "modified_p": expected["payload"],
        }
        passed = True
        for name, expected_alpha in expected_by_adapter.items():
            row = actual.get(name)
            if (
                not isinstance(row, dict)
                or isinstance(row.get("r"), bool)
                or not isinstance(row.get("r"), int)
                or row["r"] < 1
                or isinstance(row.get("lora_alpha"), bool)
                or not isinstance(row.get("lora_alpha"), (int, float))
                or float(row["lora_alpha"]) != expected_alpha
                or not isinstance(row.get("rank_pattern"), dict)
            ):
                failures.append(f"loader audit adapter metadata mismatch: {name}")
                passed = False
    if audit.get("passed") is not passed:
        failures.append("loader audit passed flag does not match recomputed result")
    return passed


def verify_evidence(
    run_dir: Path, expected_version: str, failures: list[str]
) -> None:
    evidence = read_json(run_dir / "EVIDENCE.json", failures)
    if not isinstance(evidence, dict):
        return
    if (
        evidence.get("format") != "lorango-run-evidence-v2"
        or evidence.get("toolkit_version") != expected_version
    ):
        failures.append("EVIDENCE.json format/version mismatch")
    records = evidence.get("files")
    if not isinstance(records, dict):
        failures.append("EVIDENCE.json files is not an object")
        return
    if evidence.get("files_digest") != canonical_hash(records):
        failures.append("EVIDENCE.json files_digest mismatch")
    symlinks = [path.name for path in run_dir.iterdir() if path.is_symlink()]
    if symlinks:
        failures.append(f"run directory contains symlinks: {sorted(symlinks)}")
    non_files = [path.name for path in run_dir.iterdir() if not path.is_file()]
    if non_files:
        failures.append(f"run directory is not flat/file-only: {sorted(non_files)}")
    actual_names = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "EVIDENCE.json"
    }
    if set(records) != actual_names:
        failures.append(
            "EVIDENCE.json file set mismatch: "
            f"missing={sorted(actual_names - set(records))}, "
            f"extra={sorted(set(records) - actual_names)}"
        )
    for name, record in records.items():
        path = run_dir / name
        if not path.is_file() or not isinstance(record, dict):
            failures.append(f"invalid evidence entry: {name}")
            continue
        if (
            record.get("sha256") != sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            failures.append(f"evidence hash/size mismatch: {name}")


def verify_input_snapshots(
    protocol: dict[str, Any],
    metadata: dict[str, Any],
    input_config: object,
    input_data: object,
    failures: list[str],
) -> None:
    config = protocol.get("config")
    data = protocol.get("data")
    if not isinstance(config, dict) or input_config != config:
        failures.append("INPUT_CONFIG.json does not exactly match PROTOCOL.json")
    if not isinstance(data, dict) or input_data != data:
        failures.append("INPUT_DATA.json does not exactly match PROTOCOL.json")
    if isinstance(config, dict):
        private_keys = sorted(key for key in config if key.startswith("_"))
        if private_keys:
            failures.append(f"protocol exposes private config keys: {private_keys}")
        if metadata.get("config_sha256") != canonical_hash(config):
            failures.append("RUN_METADATA config_sha256 does not match INPUT_CONFIG.json")
    if not isinstance(data, dict):
        failures.append("protocol data is not an object")
        return
    train = data.get("train")
    calibration = data.get("calibration")
    if not isinstance(train, list) or not isinstance(calibration, list):
        failures.append("protocol data split is invalid")
        return
    if protocol.get("train_count") != len(train):
        failures.append("protocol train_count mismatch")
    if protocol.get("calibration_count") != len(calibration):
        failures.append("protocol calibration_count mismatch")
    if protocol.get("validation_or_sealed_data_used") is not False:
        failures.append("protocol does not disclaim validation/sealed-data use")
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING":
        failures.append("protocol was not frozen before training")


def verify_model_manifest_binding(
    config: dict[str, Any],
    metadata: dict[str, Any],
    manifest_snapshot: object,
    failures: list[str],
) -> None:
    manifest_path = config.get("model_manifest")
    inputs = metadata.get("inputs")
    identity = inputs.get("base_model") if isinstance(inputs, dict) else None
    binding = inputs.get("model_manifest") if isinstance(inputs, dict) else None
    if not manifest_path:
        if binding is not None:
            failures.append("RUN_METADATA has an undeclared model-manifest binding")
        if manifest_snapshot != {"status": "NOT_DECLARED"}:
            failures.append("undeclared model manifest has an invalid input snapshot")
        return
    if (
        not isinstance(identity, dict)
        or not isinstance(binding, dict)
        or not isinstance(manifest_snapshot, dict)
    ):
        failures.append("declared model manifest is not bound in RUN_METADATA")
        return
    if binding.get("path") != manifest_path:
        failures.append("model-manifest path mismatch")
    if binding.get("binding_status") != "MATCH":
        failures.append("model-manifest binding status is not MATCH")
    if binding.get("canonical_sha256") != canonical_hash(manifest_snapshot):
        failures.append("model-manifest canonical snapshot hash mismatch")
    for rows_field, identity_field, digest in (
        ("weight_files", "weight_files", "weights_digest"),
        ("metadata_files", "metadata_files", "metadata_digest"),
    ):
        rows = manifest_snapshot.get(rows_field)
        if not isinstance(rows, list) or not rows:
            failures.append(f"model-manifest snapshot lacks {rows_field}")
            continue
        expected = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "relative_path",
                "sha256",
                "bytes",
            }:
                failures.append(f"model-manifest snapshot has invalid {rows_field}")
                expected = {}
                break
            if row["relative_path"] in expected:
                failures.append(f"model-manifest snapshot has duplicate {rows_field}")
                expected = {}
                break
            expected[row["relative_path"]] = {
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
        expected_digest = canonical_hash(expected)
        if manifest_snapshot.get(digest) != expected_digest:
            failures.append(f"model-manifest snapshot {digest} is inconsistent")
        if identity.get(identity_field) != expected:
            failures.append(f"model-manifest {identity_field} does not match runtime identity")
        if binding.get(digest) != expected_digest or identity.get(digest) != expected_digest:
            failures.append(f"model-manifest {digest} does not match runtime identity")


def verify_trace(trace: list[Any], protocol: dict[str, Any], failures: list[str]) -> None:
    config = protocol.get("config")
    training = config.get("training") if isinstance(config, dict) else None
    steps = training.get("optimization_steps") if isinstance(training, dict) else None
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        failures.append("protocol optimization_steps is invalid")
        return
    if len(trace) != steps:
        failures.append(f"training trace length is {len(trace)}, expected {steps}")
    data = protocol.get("data")
    runtime = config.get("runtime") if isinstance(config, dict) else None
    train_rows = data.get("train") if isinstance(data, dict) else None
    train_ids = {
        row.get("id") for row in train_rows if isinstance(row, dict)
    } if isinstance(train_rows, list) else set()
    diffusion_steps = set(runtime.get("record_steps", [])) if isinstance(runtime, dict) else set()
    required = {
        "optimization_step",
        "id",
        "diffusion_step",
        "loss",
        "nmse",
        "masked_nmse",
        "direction_loss",
        "gradient_norm",
    }
    for index, row in enumerate(trace, 1):
        if not isinstance(row, dict) or set(row) != required:
            failures.append(f"training trace row {index} has the wrong schema")
            continue
        if row.get("optimization_step") != index:
            failures.append(f"training trace row {index} has a non-contiguous step")
        if row.get("id") not in train_ids:
            failures.append(f"training trace row {index} has an unknown training id")
        if row.get("diffusion_step") not in diffusion_steps:
            failures.append(f"training trace row {index} has an invalid diffusion step")
        for field in ("loss", "nmse", "masked_nmse", "direction_loss", "gradient_norm"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                failures.append(f"training trace row {index} has invalid {field}")
            elif float(value) < 0:
                failures.append(f"training trace row {index} has negative {field}")


def verify_calibration_identities(
    protocol: dict[str, Any],
    calibration: dict[str, Any],
    post: dict[str, Any],
    failures: list[str],
) -> None:
    config = protocol.get("config")
    data = protocol.get("data")
    if not isinstance(config, dict) or not isinstance(data, dict):
        failures.append("protocol lacks config/data for calibration identity verification")
        return
    runtime = config.get("runtime")
    training = config.get("training")
    examples = data.get("calibration")
    if not isinstance(runtime, dict) or not isinstance(training, dict) or not isinstance(examples, list):
        failures.append("protocol calibration definition is invalid")
        return
    try:
        expected = {
            (row["id"], seed, step)
            for row in examples
            for seed in training["calibration_seeds"]
            for step in runtime["record_steps"]
        }
    except (KeyError, TypeError):
        failures.append("protocol calibration definition is incomplete")
        return
    for label, record in (("CALIBRATION", calibration), ("POST_EXPORT_CALIBRATION", post)):
        actual = {
            (row.get("id"), row.get("seed"), row.get("step"))
            for row in record.get("rows", [])
            if isinstance(row, dict)
        }
        if actual != expected:
            failures.append(
                f"{label} row identities do not match the protocol Cartesian product"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify a completed v1.4 LoRango run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    failures = [
        f"missing {name}" for name in sorted(REQUIRED) if not (run_dir / name).is_file()
    ]
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(1)

    manifest = read_json(run_dir / "MANIFEST.json", failures)
    protocol = read_json(run_dir / "PROTOCOL.json", failures)
    metadata = read_json(run_dir / "RUN_METADATA.json", failures)
    input_config = read_json(run_dir / "INPUT_CONFIG.json", failures)
    input_data = read_json(run_dir / "INPUT_DATA.json", failures)
    input_model_manifest = read_json(
        run_dir / "INPUT_MODEL_MANIFEST.json", failures
    )
    compilation = read_json(run_dir / "COMPILE.json", failures)
    calibration = read_json(run_dir / "CALIBRATION.json", failures)
    post = read_json(run_dir / "POST_EXPORT_CALIBRATION.json", failures)
    trace = read_json(run_dir / "training_trace.json", failures)
    if not all(
        isinstance(value, dict)
        for value in (manifest, protocol, metadata, compilation, calibration, post)
    ):
        failures.append("one or more required documents have the wrong top-level type")
    if not isinstance(trace, list) or not trace:
        failures.append("training_trace.json must be a non-empty array")
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(1)

    assert isinstance(manifest, dict) and isinstance(protocol, dict)
    assert isinstance(metadata, dict) and isinstance(compilation, dict)
    assert isinstance(calibration, dict) and isinstance(post, dict)
    run_version = manifest.get("toolkit_version")
    if (
        run_version not in SUPPORTED_RUN_VERSIONS
        or protocol.get("version") != run_version
    ):
        failures.append("manifest/protocol has an unsupported or inconsistent version")
    if metadata.get("toolkit_version") != run_version or metadata.get("mode") != "train":
        failures.append("RUN_METADATA version/mode does not match the run")
    verify_input_snapshots(protocol, metadata, input_config, input_data, failures)
    allowed_status = {"PASS", "FUNCTIONAL_SMOKE_PASS"} if args.allow_smoke else {"PASS"}
    if manifest.get("status") not in allowed_status:
        failures.append(f"manifest status is not acceptable: {manifest.get('status')!r}")
    if (
        manifest.get("status") == "FUNCTIONAL_SMOKE_PASS"
        and manifest.get("quality_claim") is not False
    ):
        failures.append("smoke run does not explicitly disclaim a quality claim")
    if manifest.get("ordinary_static_lora") is not True:
        failures.append("ordinary_static_lora is not true")
    if (
        manifest.get("external_hook_at_inference") is not False
        or manifest.get("external_detector_at_inference") is not False
    ):
        failures.append("manifest declares an external inference dependency")
    verify_trace(trace, protocol, failures)
    verify_calibration_identities(protocol, calibration, post, failures)

    config = protocol.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("gates"), dict):
        failures.append("protocol does not contain gate thresholds")
        thresholds: dict[str, Any] = {}
    else:
        thresholds = config["gates"]
        verify_model_manifest_binding(
            config, metadata, input_model_manifest, failures
        )
    required_thresholds = {
        "mean_cosine_min",
        "mean_nrmse_max",
        "single_relative_drift_max",
        "export_equivalence_max",
    }
    thresholds_valid = required_thresholds.issubset(thresholds)
    if not thresholds_valid:
        failures.append("protocol gate thresholds are incomplete")
    else:
        for name in required_thresholds:
            value = thresholds[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                failures.append(f"protocol threshold {name} is not finite numeric")
                thresholds_valid = False
        if thresholds_valid:
            if not -1 <= float(thresholds["mean_cosine_min"]) <= 1:
                failures.append("protocol mean_cosine_min is outside [-1, 1]")
                thresholds_valid = False
            for name in (
                "mean_nrmse_max",
                "single_relative_drift_max",
                "export_equivalence_max",
            ):
                if float(thresholds[name]) < 0:
                    failures.append(f"protocol threshold {name} is negative")
                    thresholds_valid = False

    hook_summary = recompute_calibration(calibration, "CALIBRATION", failures)
    post_summary = recompute_calibration(post, "POST_EXPORT_CALIBRATION", failures)
    expected_hook_summary = {key: value for key, value in calibration.items() if key != "rows"}
    expected_post_summary = {key: value for key, value in post.items() if key != "rows"}
    if manifest.get("calibration_summary") != expected_hook_summary:
        failures.append("manifest calibration_summary does not match CALIBRATION.json")
    if manifest.get("post_export_calibration_summary") != expected_post_summary:
        failures.append(
            "manifest post_export_calibration_summary does not match POST_EXPORT_CALIBRATION.json"
        )
    expected_hook_gates: dict[str, bool] = {}
    expected_post: dict[str, bool] = {}
    if hook_summary is not None and thresholds_valid:
        expected_hook_gates = recompute_gates(hook_summary, thresholds)
        if (
            set(manifest.get("gates") or {}) != HOOK_GATES
            or manifest.get("gates") != expected_hook_gates
        ):
            failures.append("manifest hook gates do not match recomputed gates")
    if post_summary is not None and thresholds_valid:
        expected_post = recompute_gates(post_summary, thresholds)
        expected_post["loader_scaling"] = verify_loader_scaling(
            post, compilation, failures
        )
        expected_post["hook_export_equivalence"] = verify_equivalence(
            calibration,
            post,
            float(thresholds["export_equivalence_max"]),
            failures,
        )
        if (
            set(manifest.get("post_export_gates") or {}) != POST_GATES
            or manifest.get("post_export_gates") != expected_post
        ):
            failures.append("manifest post-export gates do not match recomputed gates")
        if not all(expected_post.values()):
            failures.append(f"one or more recomputed post-export gates failed: {expected_post}")

    if (
        compilation.get("external_hook_at_inference") is not False
        or compilation.get("external_detector_at_inference") is not False
    ):
        failures.append("compiled output requires an external inference dependency")
    layers = compilation.get("layers")
    if not isinstance(layers, list) or len(layers) != 5:
        failures.append("compiled topology is not exactly five layers")
        compiled_by_name: dict[str, dict[str, Any]] = {}
    else:
        compiled_by_name = {
            layer.get("name"): layer for layer in layers if isinstance(layer, dict)
        }
        if set(compiled_by_name) != set(EXPECTED_LAYERS):
            failures.append("compiled layer names do not match the frozen topology")
        else:
            for name, block in EXPECTED_LAYERS.items():
                layer = compiled_by_name[name]
                channels = layer.get("channels")
                if layer.get("block") != block or not isinstance(channels, list) or len(channels) != 4:
                    failures.append(f"compiled layer topology mismatch: {name}")
    file_records = compilation.get("files")
    if not isinstance(file_records, dict) or set(file_records) != COMPILED_FILES:
        failures.append("compiled file set is not the exact four-file contract")
    else:
        for name, record in file_records.items():
            path = run_dir / name
            tensors = validate_safetensors(path, failures)
            verify_lora_tensors(name, tensors, compilation, failures)
            if (
                not isinstance(record, dict)
                or record.get("sha256") != sha256(path)
                or record.get("bytes") != path.stat().st_size
            ):
                failures.append(f"compiled file hash/size mismatch: {name}")
    if manifest.get("compilation") != compilation:
        failures.append("MANIFEST compilation does not exactly match COMPILE.json")

    checkpoint_layers = verify_checkpoint(run_dir, manifest, failures)
    if compiled_by_name and checkpoint_layers:
        checkpoint_by_name = {
            layer.get("name"): layer
            for layer in checkpoint_layers
            if isinstance(layer, dict)
        }
        for name, compiled_layer in compiled_by_name.items():
            checkpoint_layer = checkpoint_by_name.get(name)
            if (
                not isinstance(checkpoint_layer, dict)
                or checkpoint_layer.get("block") != compiled_layer.get("block")
                or checkpoint_layer.get("channels") != compiled_layer.get("channels")
            ):
                failures.append(f"checkpoint/compiled topology mismatch: {name}")
    inputs = metadata.get("inputs")
    identity = inputs.get("base_model") if isinstance(inputs, dict) else None
    if not isinstance(identity, dict) or identity.get("identity_complete") is not True:
        failures.append("base-model identity is incomplete")
    else:
        for kind, digest_name in (
            ("weight_files", "weights_digest"),
            ("metadata_files", "metadata_digest"),
        ):
            records = identity.get(kind)
            digest = identity.get(digest_name)
            if (
                not isinstance(records, dict)
                or not records
                or not isinstance(digest, str)
                or not HEX64.fullmatch(digest)
            ):
                failures.append(f"base-model {kind}/{digest_name} is absent")
            elif canonical_hash(records) != digest:
                failures.append(f"base-model {digest_name} does not match its file records")

    verify_evidence(run_dir, str(run_version), failures)
    result = {
        "status": "PASS" if not failures else "FAIL",
        "run_dir": str(run_dir),
        "recomputed_hook_gates": expected_hook_gates,
        "recomputed_post_export_gates": expected_post,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
