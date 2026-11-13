from __future__ import annotations

import math
from typing import Any


def _metric_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["candidate_id"]), int(row["rank"])


def _passes(row: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    checks = [
        (bool(row.get("finite", False)), "non_finite"),
        (bool(row.get("exportable", False)), "not_exportable"),
        (bool(row.get("loader_supported", False)), "loader_unsupported"),
        (
            float(row["interaction_cosine"]) >= float(thresholds["minimum_interaction_cosine"]),
            "interaction_cosine_below_minimum",
        ),
        (
            float(row["s_drift_ratio"]) <= float(thresholds["maximum_single_drift_ratio"]),
            "signature_drift_above_maximum",
        ),
        (
            float(row["p_drift_ratio"]) <= float(thresholds["maximum_single_drift_ratio"]),
            "payload_drift_above_maximum",
        ),
        (
            float(thresholds["minimum_sp_amplitude"])
            <= float(row["sp_amplitude"])
            <= float(thresholds["maximum_sp_amplitude"]),
            "sp_amplitude_outside_range",
        ),
    ]
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    return not reasons, reasons


def score_row(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    weights = policy["score"]
    maximum_rank = max(int(value) for value in policy["probe"]["ranks"])
    amplitude_error = abs(math.log(max(float(row["sp_amplitude"]), 1e-8)))
    score = (
        float(weights["interaction_cosine_weight"]) * float(row["interaction_cosine"])
        + float(weights["sp_cosine_weight"]) * float(row["sp_cosine"])
        - float(weights["signature_drift_weight"]) * float(row["s_drift_ratio"])
        - float(weights["payload_drift_weight"]) * float(row["p_drift_ratio"])
        - float(weights["amplitude_error_weight"]) * amplitude_error
        - float(weights["rank_cost_weight"]) * int(row["rank"]) / maximum_rank
    )
    passed, reasons = _passes(row, policy["thresholds"])
    result = dict(row)
    result.update(
        {
            "score": score,
            "passed_thresholds": passed,
            "rejection_reasons": reasons,
        }
    )
    return result


def rank_candidates(
    candidates_document: dict[str, Any], metrics: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    candidates = {row["candidate_id"]: row for row in candidates_document["candidates"]}
    expected_ranks = {int(value) for value in policy["probe"]["ranks"]}
    seen: set[tuple[str, int]] = set()
    rows = []
    for metric in metrics:
        key = _metric_key(metric)
        if key in seen:
            raise ValueError(f"duplicate probe metric for {key}")
        seen.add(key)
        candidate_id, rank = key
        if candidate_id not in candidates:
            raise KeyError(f"probe metric refers to unknown candidate {candidate_id}")
        if rank not in expected_ranks:
            raise ValueError(f"probe metric rank {rank} is outside policy ranks")
        combined = {**candidates[candidate_id], **metric}
        combined["parameter_count"] = int(combined["parameter_width_per_rank"]) * rank
        rows.append(score_row(combined, policy))

    rows.sort(
        key=lambda row: (
            not bool(row["passed_thresholds"]),
            -float(row["score"]),
            int(row["rank"]),
            str(row["candidate_id"]),
        )
    )
    for index, row in enumerate(rows, 1):
        row["global_rank"] = index
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "candidate_rows": len(rows),
        "rows": rows,
    }


def select_candidates(ranked: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    maximum = int(policy["selection"]["maximum_selected_candidates"])
    maximum_per_family = int(policy["selection"]["maximum_per_family"])
    selected = []
    family_counts: dict[str, int] = {}
    used_module_sets: set[tuple[str, ...]] = set()
    for row in ranked["rows"]:
        if len(selected) >= maximum:
            break
        if not row["passed_thresholds"]:
            continue
        family = str(row["family"])
        if family_counts.get(family, 0) >= maximum_per_family:
            continue
        module_set = tuple(sorted(row["roles"].values()))
        if module_set in used_module_sets:
            continue
        selected.append(row)
        used_module_sets.add(module_set)
        family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "status": "selected" if selected else "no_supported_construction_found",
        "selected": selected,
        "stopping_rule": policy["selection"]["stopping_rule"],
    }

