from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from safetensors.torch import safe_open

from .checkpoint import metadata_path


PACKAGE_ROOT = Path(__file__).resolve().parent

DEFAULTS: dict[str, Any] = {
    "device": "cuda:0",
    "local_files_only": True,
    "runtime": {
        "height": 512,
        "width": 512,
        "inference_steps": 30,
        "guidance_scale": 7.5,
        "record_steps": [0, 3, 7, 12, 18, 20],
    },
    "training": {
        "optimization_steps": 3000,
        "train_seeds": [2026087701, 2026087702],
        "calibration_seeds": [2026087791],
        "action_lr": 8e-4,
        "reader_lr": 2e-4,
        "train_readers": True,
        "use_warm_start_actions": True,
        "checkpoint_every": 500,
        "fixed_seed": 2026087714,
        "deterministic": True,
    },
    "carrier": {
        "group_index": 1,
        "gate_scale": 35.0,
        "reader_scale": 0.06,
        "action_scale": 0.45,
        "value_cancel_scale": 0.90,
    },
    "gates": {
        "mean_cosine_min": 0.35,
        "mean_nrmse_max": 1.10,
        "single_relative_drift_max": 0.020,
        "export_equivalence_max": 0.010,
        "compile_on_failure": False,
    },
}

REQUIRED_PATHS = (
    "base_model",
    "signature_cover_lora",
    "payload_cover_lora",
    "data_json",
    "output_dir",
)


def _reject_unknown_keys(user: dict[str, Any]) -> None:
    allowed_top = set(DEFAULTS) | set(REQUIRED_PATHS) | {
        "scaffold_init",
        "carrier_catalog",
        "model_manifest",
    }
    unknown_top = sorted(set(user) - allowed_top)
    if unknown_top:
        raise ValueError(f"unknown config keys: {unknown_top}")
    for section in ("runtime", "training", "carrier", "gates"):
        if section in user:
            if not isinstance(user[section], dict):
                raise TypeError(f"{section} must be a JSON object")
            unknown = sorted(set(user[section]) - set(DEFAULTS[section]))
            if unknown:
                raise ValueError(f"unknown {section} keys: {unknown}")


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _absolute(value: str, parent: Path) -> str:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return str(path if path.is_absolute() else (parent / path).resolve())


def load_config(path: Path, allow_completed_output: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    user = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(user, dict):
        raise TypeError("top-level config must be a JSON object")
    _reject_unknown_keys(user)
    config = _deep_update(deepcopy(DEFAULTS), user)
    config["_config_path"] = str(path)
    missing = [key for key in REQUIRED_PATHS if not config.get(key)]
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
    for key in ("base_model", "signature_cover_lora", "payload_cover_lora", "data_json", "output_dir"):
        config[key] = _absolute(str(config[key]), path.parent)
    config["scaffold_init"] = _absolute(
        str(
            config.get(
                "scaffold_init", PACKAGE_ROOT / "assets" / "scaffold_init.safetensors"
            )
        ),
        path.parent,
    )
    config["carrier_catalog"] = _absolute(
        str(config.get("carrier_catalog", PACKAGE_ROOT / "assets" / "carrier_catalog.json")),
        path.parent,
    )
    if config.get("model_manifest"):
        config["model_manifest"] = _absolute(str(config["model_manifest"]), path.parent)
    validate_config(config, allow_completed_output=allow_completed_output)
    return config


def _adapter_keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _validate_cover(path: Path, label: str) -> None:
    keys = _adapter_keys(path)
    if not keys:
        raise ValueError(f"{label} is empty: {path}")
    forbidden = sorted(
        key
        for key in keys
        if "lora_magnitude_vector" in key
        or key.endswith(".alpha")
        or not key.startswith("unet.")
    )
    if forbidden:
        raise ValueError(
            f"{label} must be a canonical UNet-only, non-DoRA Diffusers LoRA; "
            f"unsupported keys: {forbidden[:4]}"
        )
    down = {key[: -len(".lora.down.weight")] for key in keys if key.endswith(".lora.down.weight")}
    up = {key[: -len(".lora.up.weight")] for key in keys if key.endswith(".lora.up.weight")}
    recognized = {
        key
        for key in keys
        if key.endswith(".lora.down.weight") or key.endswith(".lora.up.weight")
    }
    if not down or down != up or len(recognized) != len(keys):
        raise ValueError(
            f"{label} must contain exactly paired Diffusers lora.down/lora.up tensors"
        )


def validate_config(config: dict[str, Any], allow_completed_output: bool = False) -> None:
    if not _boolean(config.get("local_files_only", True), "local_files_only"):
        raise ValueError(
            "v1.4 requires local_files_only=true; download and verify model assets "
            "before running the artifact"
        )
    for key in ("base_model", "signature_cover_lora", "payload_cover_lora", "data_json", "scaffold_init", "carrier_catalog"):
        path = Path(config[key])
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
    if config.get("model_manifest") and not Path(config["model_manifest"]).is_file():
        raise FileNotFoundError(f"model_manifest: {config['model_manifest']}")
    if config.get("model_manifest"):
        manifest = json.loads(Path(config["model_manifest"]).read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("weight_files"), list)
            or not isinstance(manifest.get("metadata_files"), list)
            or not isinstance(manifest.get("weights_digest"), str)
            or not isinstance(manifest.get("metadata_digest"), str)
        ):
            raise ValueError("model_manifest does not contain the required identity fields")
    if not metadata_path(Path(config["scaffold_init"])).is_file():
        raise FileNotFoundError(
            f"scaffold checkpoint metadata: {metadata_path(Path(config['scaffold_init']))}"
        )
    output = Path(config["output_dir"])
    if (
        not allow_completed_output
        and output.exists()
        and (output / "MANIFEST.json").exists()
    ):
        raise FileExistsError(f"completed output already exists: {output}")
    if Path(config["signature_cover_lora"]).resolve() == Path(
        config["payload_cover_lora"]
    ).resolve():
        raise ValueError("Signature and Payload cover LoRAs must be different files")
    runtime = config["runtime"]
    steps = _integer(runtime["inference_steps"], "runtime.inference_steps")
    if not isinstance(runtime["record_steps"], list):
        raise ValueError("runtime.record_steps must be an array")
    record = [
        _integer(value, f"runtime.record_steps[{index}]")
        for index, value in enumerate(runtime["record_steps"])
    ]
    if not record or record != sorted(set(record)) or record[0] < 0 or record[-1] >= steps:
        raise ValueError("runtime.record_steps must be sorted, unique, and within inference_steps")
    if steps < 1:
        raise ValueError("runtime.inference_steps must be >= 1")
    height = _integer(runtime["height"], "runtime.height")
    width = _integer(runtime["width"], "runtime.width")
    if height < 128 or width < 128 or height % 8 or width % 8:
        raise ValueError("runtime height and width must be >=128 and divisible by 8")
    if not re.fullmatch(r"(?:cuda(?::[0-9]+)?|cpu)", str(config["device"])):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if _finite_number(runtime["guidance_scale"], "runtime.guidance_scale") < 0:
        raise ValueError("runtime.guidance_scale must be non-negative")
    group = _integer(config["carrier"]["group_index"], "carrier.group_index")
    if group not in (0, 1, 2, 3):
        raise ValueError("carrier.group_index must be one of 0, 1, 2, 3")
    for name in ("train_seeds", "calibration_seeds"):
        if not isinstance(config["training"].get(name), list) or not config["training"].get(name):
            raise ValueError(f"training.{name} cannot be empty")
    train_values = [
        _integer(value, f"training.train_seeds[{index}]")
        for index, value in enumerate(config["training"]["train_seeds"])
    ]
    calibration_values = [
        _integer(value, f"training.calibration_seeds[{index}]")
        for index, value in enumerate(config["training"]["calibration_seeds"])
    ]
    if len(train_values) != len(set(train_values)):
        raise ValueError("training.train_seeds must be unique")
    if len(calibration_values) != len(set(calibration_values)):
        raise ValueError("training.calibration_seeds must be unique")
    train_seeds = set(train_values)
    calibration_seeds = set(calibration_values)
    if train_seeds.intersection(calibration_seeds):
        raise ValueError("train_seeds and calibration_seeds must be disjoint")
    if _integer(config["training"]["optimization_steps"], "training.optimization_steps") < 1:
        raise ValueError("training.optimization_steps must be >= 1")
    if _integer(config["training"]["checkpoint_every"], "training.checkpoint_every") < 0:
        raise ValueError("training.checkpoint_every must be >= 0")
    _integer(config["training"]["fixed_seed"], "training.fixed_seed")
    for key in ("train_readers", "use_warm_start_actions", "deterministic"):
        _boolean(config["training"][key], f"training.{key}")
    for key in ("action_lr", "reader_lr"):
        if _finite_number(config["training"][key], f"training.{key}") <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("gate_scale", "reader_scale", "action_scale"):
        if _finite_number(config["carrier"][key], f"carrier.{key}") <= 0:
            raise ValueError(f"carrier.{key} must be positive")
    if _finite_number(
        config["carrier"]["value_cancel_scale"], "carrier.value_cancel_scale"
    ) < 0:
        raise ValueError("carrier.value_cancel_scale must be non-negative")
    mean_cosine = _finite_number(config["gates"]["mean_cosine_min"], "gates.mean_cosine_min")
    if not -1 <= mean_cosine <= 1:
        raise ValueError("gates.mean_cosine_min must be within [-1, 1]")
    for key in ("mean_nrmse_max", "single_relative_drift_max", "export_equivalence_max"):
        if _finite_number(config["gates"][key], f"gates.{key}") < 0:
            raise ValueError(f"gates.{key} must be non-negative")
    _boolean(config["gates"]["compile_on_failure"], "gates.compile_on_failure")
    for key in ("signature_cover_lora", "payload_cover_lora"):
        _validate_cover(Path(config[key]), key)


def load_data(path: Path) -> dict[str, list[dict[str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"train", "calibration"}:
        raise ValueError("data JSON must contain exactly 'train' and 'calibration'")
    result: dict[str, list[dict[str, str]]] = {}
    seen: set[str] = set()
    normalized_by_split: dict[str, dict[str, set[Any]]] = {}
    for split in ("train", "calibration"):
        rows = []
        for index, row in enumerate(raw[split]):
            if isinstance(row, list) and len(row) == 3:
                item_id, base, target = row
            elif isinstance(row, dict):
                item_id = row.get("id")
                base = row.get("base_prompt", row.get("base"))
                target = row.get("target_prompt", row.get("teacher"))
            else:
                raise ValueError(f"invalid {split} row {index}: {row!r}")
            if not all(isinstance(value, str) and value.strip() for value in (item_id, base, target)):
                raise ValueError(f"empty or non-string field in {split} row {index}")
            if item_id in seen:
                raise ValueError(f"duplicate example id: {item_id}")
            seen.add(item_id)
            rows.append({"id": item_id, "base": base, "target": target})
        result[split] = rows
        normalize = lambda value: " ".join(value.split()).casefold()
        normalized_by_split[split] = {
            "base": {normalize(row["base"]) for row in rows},
            "target": {normalize(row["target"]) for row in rows},
            "pair": {(normalize(row["base"]), normalize(row["target"])) for row in rows},
        }
    if not result["train"] or not result["calibration"]:
        raise ValueError("both train and calibration splits must be non-empty")
    for field in ("base", "target", "pair"):
        overlap = normalized_by_split["train"][field].intersection(
            normalized_by_split["calibration"][field]
        )
        if overlap:
            raise ValueError(
                f"train/calibration {field} prompts must be disjoint; "
                f"overlap={sorted(overlap, key=str)[:2]}"
            )
    return result
