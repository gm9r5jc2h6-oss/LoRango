from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .checkpoint import metadata_path
from .runtime import sha256


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _base_model_identity(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    result: dict[str, Any] = {"path_or_id": path_value}
    if not path.exists():
        result["local"] = False
        return result
    result["local"] = True
    if path.is_file():
        result["sha256"] = sha256(path)
        result["bytes"] = path.stat().st_size
        return result
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    revision_metadata = path / ".cache" / "huggingface" / "download" / "model_index.json.metadata"
    if revision_metadata.is_file():
        lines = revision_metadata.read_text(encoding="utf-8").splitlines()
        if lines:
            result["huggingface_revision"] = lines[0]
    weight_files = {}
    metadata_files = {}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = str(item.relative_to(path))
        record = {"sha256": sha256(item), "bytes": item.stat().st_size}
        if item.suffix.lower() in weight_suffixes:
            weight_files[relative] = record
        elif item.suffix.lower() in {".json", ".txt", ".model"}:
            metadata_files[relative] = record
    if not weight_files:
        raise ValueError(f"base model directory contains no recognized weight files: {path}")
    result["weight_files"] = weight_files
    result["weights_bytes"] = sum(item["bytes"] for item in weight_files.values())
    result["weights_digest"] = _canonical_hash(weight_files)
    result["metadata_files"] = metadata_files
    result["metadata_digest"] = _canonical_hash(metadata_files)
    result["identity_complete"] = True
    return result


def _bind_model_manifest(
    identity: dict[str, Any],
    manifest_path_value: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require the loaded local model to match the declared reference manifest."""

    manifest_path = Path(manifest_path_value)
    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if current_manifest != manifest:
        raise ValueError("model manifest changed while input snapshots were created")
    expected: dict[str, dict[str, dict[str, Any]]] = {}
    for field, identity_field in (
        ("weight_files", "weight_files"),
        ("metadata_files", "metadata_files"),
    ):
        rows = manifest.get(field)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"model manifest {field} must be a non-empty array")
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "relative_path",
                "sha256",
                "bytes",
            }:
                raise ValueError(f"model manifest has an invalid {field} entry")
            relative = row["relative_path"]
            if not isinstance(relative, str) or not relative or relative in records:
                raise ValueError(f"model manifest has a duplicate/invalid {field} path")
            records[relative] = {"sha256": row["sha256"], "bytes": row["bytes"]}
        expected[identity_field] = records
    failures = []
    for field, digest_field in (
        ("weight_files", "weights_digest"),
        ("metadata_files", "metadata_digest"),
    ):
        expected_digest = _canonical_hash(expected[field])
        if manifest.get(digest_field) != expected_digest:
            failures.append(f"manifest {digest_field} is internally inconsistent")
        if identity.get(field) != expected[field]:
            failures.append(f"runtime {field} does not match the manifest")
        if identity.get(digest_field) != expected_digest:
            failures.append(f"runtime {digest_field} does not match the manifest")
    if failures:
        raise ValueError("base-model identity binding failed: " + "; ".join(failures))
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "canonical_sha256": _canonical_hash(manifest),
        "schema_version": manifest.get("schema_version"),
        "model_id": manifest.get("model_id"),
        "huggingface_revision": manifest.get("huggingface_revision"),
        "weights_digest": manifest["weights_digest"],
        "metadata_digest": manifest["metadata_digest"],
        "binding_status": "MATCH",
    }


def build_run_metadata(
    config: dict[str, Any],
    mode: str,
    model_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = str(config["device"])
    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    base_model = _base_model_identity(config["base_model"])
    inputs = {
        "base_model": base_model,
        "signature_cover_lora": {
            "path": config["signature_cover_lora"],
            "sha256": sha256(Path(config["signature_cover_lora"])),
        },
        "payload_cover_lora": {
            "path": config["payload_cover_lora"],
            "sha256": sha256(Path(config["payload_cover_lora"])),
        },
        "data_json": {
            "path": config["data_json"],
            "sha256": sha256(Path(config["data_json"])),
        },
        "scaffold_init": {
            "path": config["scaffold_init"],
            "sha256": sha256(Path(config["scaffold_init"])),
            "metadata_path": str(metadata_path(Path(config["scaffold_init"]))),
            "metadata_sha256": sha256(metadata_path(Path(config["scaffold_init"]))),
        },
        "carrier_catalog": {
            "path": config["carrier_catalog"],
            "sha256": sha256(Path(config["carrier_catalog"])),
        },
    }
    if config.get("model_manifest"):
        if not isinstance(model_manifest, dict) or model_manifest.get("status") == "NOT_DECLARED":
            raise ValueError("declared model manifest snapshot is absent")
        inputs["model_manifest"] = _bind_model_manifest(
            base_model, config["model_manifest"], model_manifest
        )
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "schema_version": 1,
        "toolkit": "LoRango SDXL Pair-Training Toolkit",
        "toolkit_version": __version__,
        "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config.get("_config_path"),
        "config_sha256": _canonical_hash(public_config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "diffusers": _version("diffusers"),
            "transformers": _version("transformers"),
            "safetensors": _version("safetensors"),
            "numpy": _version("numpy"),
            "accelerate": _version("accelerate"),
            "peft": _version("peft"),
            "huggingface_hub": _version("huggingface-hub"),
            "device": device,
            "gpu": gpu,
        },
        "determinism": {
            "requested": bool(config["training"].get("deterministic", True)),
            "fixed_seed": int(config["training"]["fixed_seed"]),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "note": "When requested, training disables Flash/memory-efficient SDP and enables strict deterministic algorithms.",
        },
        "inputs": inputs,
    }
