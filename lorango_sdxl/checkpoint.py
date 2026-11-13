from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .runtime import sha256, write_json


FORMAT = "lorango-checkpoint-v1"
PRIMITIVE_FIELDS = {
    "name",
    "block",
    "channels",
    "action_normalizers",
    "signals",
    "references",
}
TENSOR_FIELDS = {"amplitudes", "value_readers", "actions", "constant_reader"}


def metadata_path(tensor_path: Path) -> Path:
    tensor_path = Path(tensor_path)
    if tensor_path.suffix != ".safetensors":
        raise ValueError(f"checkpoint must use .safetensors, got: {tensor_path}")
    return tensor_path.with_suffix(".json")


def checkpoint_files(tensor_path: Path) -> tuple[Path, Path]:
    tensor_path = Path(tensor_path)
    return tensor_path, metadata_path(tensor_path)


def _validate_metadata(metadata: object, tensor_path: Path) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    allowed_top = {
        "format",
        "training_step",
        "tensor_file",
        "tensor_sha256",
        "layers",
        "source",
    }
    unknown_top = set(metadata) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown checkpoint metadata fields: {sorted(unknown_top)}")
    if metadata.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {metadata.get('format')!r}")
    if metadata.get("tensor_file") != tensor_path.name:
        raise ValueError("checkpoint metadata tensor_file does not match its file")
    if metadata.get("tensor_sha256") != sha256(tensor_path):
        raise ValueError("checkpoint tensor hash does not match metadata")
    step = metadata.get("training_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint training_step must be a non-negative integer")
    layers = metadata.get("layers")
    if not isinstance(layers, list) or len(layers) not in {5, 12}:
        raise ValueError("checkpoint must contain the frozen 5-layer state or 12-layer scaffold")
    seen = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"checkpoint layer {index} must be an object")
        unknown = set(layer) - PRIMITIVE_FIELDS - {"tensor_fields"}
        if unknown:
            raise ValueError(f"unknown checkpoint layer fields at {index}: {sorted(unknown)}")
        name = layer.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"invalid or duplicate checkpoint layer name at {index}: {name!r}")
        seen.add(name)
        fields = layer.get("tensor_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or len(fields) != len(set(fields))
            or not set(fields).issubset(TENSOR_FIELDS)
        ):
            raise ValueError(f"invalid tensor_fields for checkpoint layer {name}")
        if not {"value_readers", "actions"}.issubset(fields):
            raise ValueError(f"checkpoint layer {name} lacks trainable tensors")
    return metadata


def load_checkpoint(tensor_path: Path | str) -> dict[str, Any]:
    tensor_path = Path(tensor_path).expanduser().resolve()
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    sidecar = metadata_path(tensor_path)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    metadata = _validate_metadata(
        json.loads(sidecar.read_text(encoding="utf-8")), tensor_path
    )
    tensors = load_file(str(tensor_path), device="cpu")
    expected = {
        f"layers.{index}.{field}"
        for index, layer in enumerate(metadata["layers"])
        for field in layer["tensor_fields"]
    }
    if set(tensors) != expected:
        raise ValueError(
            "checkpoint tensor set mismatch: "
            f"missing={sorted(expected - set(tensors))}, extra={sorted(set(tensors) - expected)}"
        )
    layers = []
    for index, record in enumerate(metadata["layers"]):
        layer = {key: value for key, value in record.items() if key != "tensor_fields"}
        for field in record["tensor_fields"]:
            value = tensors[f"layers.{index}.{field}"]
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"checkpoint tensor layers.{index}.{field} must be finite F32")
            layer[field] = value
        layers.append(layer)
    return {"training_step": metadata["training_step"], "layers": layers}


def save_checkpoint(tensor_path: Path | str, state: dict[str, Any]) -> tuple[Path, Path]:
    tensor_path = Path(tensor_path)
    sidecar = metadata_path(tensor_path)
    if set(state) != {"training_step", "layers"}:
        raise ValueError("checkpoint state must contain exactly training_step and layers")
    layers = state["layers"]
    if not isinstance(layers, list) or len(layers) != 5:
        raise ValueError("trained checkpoint must contain exactly five layers")
    tensor_state: dict[str, torch.Tensor] = {}
    metadata_layers = []
    seen = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"checkpoint layer {index} must be an object")
        unknown = set(layer) - PRIMITIVE_FIELDS - TENSOR_FIELDS
        if unknown:
            raise ValueError(f"unsupported checkpoint state fields: {sorted(unknown)}")
        name = layer.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"invalid or duplicate layer name: {name!r}")
        seen.add(name)
        metadata_layer = {
            key: value for key, value in layer.items() if key in PRIMITIVE_FIELDS
        }
        fields = []
        for field in sorted(TENSOR_FIELDS.intersection(layer)):
            value = layer[field]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"checkpoint {name}.{field} must be a tensor")
            value = value.detach().cpu().to(torch.float32).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"checkpoint {name}.{field} contains non-finite values")
            tensor_state[f"layers.{index}.{field}"] = value
            fields.append(field)
        if not {"value_readers", "actions"}.issubset(fields):
            raise ValueError(f"checkpoint layer {name} lacks trainable tensors")
        metadata_layer["tensor_fields"] = fields
        metadata_layers.append(metadata_layer)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    save_file(tensor_state, str(temporary), metadata={"format": FORMAT})
    temporary.replace(tensor_path)
    metadata = {
        "format": FORMAT,
        "training_step": int(state["training_step"]),
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256(tensor_path),
        "layers": metadata_layers,
    }
    write_json(sidecar, metadata)
    return tensor_path, sidecar
