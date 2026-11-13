from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import canonical_sha256, sha256_file


def build_plan(
    architecture: dict[str, Any],
    policy: dict[str, Any],
    architecture_path: Path,
    policy_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    stages = [
        ("00_loader_contract", [], ["loader_contract.json"]),
        ("01_module_inventory", ["loader_contract.json"], ["module_inventory.json"]),
        ("02_candidate_generation", ["module_inventory.json"], ["candidates.json"]),
        ("03_activation_capture", ["candidates.json"], ["activation_manifest.json"]),
        ("04_short_probes", ["activation_manifest.json"], ["probe_metrics.jsonl"]),
        ("05_rank", ["candidates.json", "probe_metrics.jsonl"], ["ranked_candidates.json"]),
        ("06_select", ["ranked_candidates.json"], ["selected_topology.json"]),
        ("07_full_cache", ["selected_topology.json"], ["state_cache.pt", "cache_manifest.json"]),
        ("08_train", ["state_cache.pt"], ["training_state.pt", "training_trace.jsonl"]),
        ("09_export", ["training_state.pt"], ["signature.safetensors", "payload.safetensors"]),
        ("10_fresh_process_audit", ["signature.safetensors", "payload.safetensors"], ["four_condition_audit.json"]),
    ]
    return {
        "schema_version": 1,
        "architecture_id": architecture["architecture_id"],
        "policy_id": policy["policy_id"],
        "inputs": {
            "architecture": {"path": str(architecture_path), "sha256": sha256_file(architecture_path)},
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
        },
        "policy_canonical_sha256": canonical_sha256(policy),
        "stages": [
            {"index": index, "name": name, "inputs": inputs, "outputs": outputs}
            for index, (name, inputs, outputs) in enumerate(stages)
        ],
        "negative_result_is_valid": True,
    }

