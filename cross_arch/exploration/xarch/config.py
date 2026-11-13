from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import load_json


def require_keys(value: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def load_architecture(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require_keys(
        value,
        ["schema_version", "architecture_id", "model", "candidate_templates"],
        "architecture config",
    )
    if value["schema_version"] != 1:
        raise ValueError("only architecture schema_version=1 is supported")
    if not value["candidate_templates"]:
        raise ValueError("architecture must declare at least one candidate template")
    for template in value["candidate_templates"]:
        require_keys(template, ["id", "family", "block_ids", "roles"], "candidate template")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require_keys(
        value,
        ["schema_version", "policy_id", "probe", "score", "thresholds", "selection"],
        "exploration policy",
    )
    if value["schema_version"] != 1:
        raise ValueError("only policy schema_version=1 is supported")
    return value

