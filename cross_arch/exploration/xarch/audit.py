from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .util import sha256_file


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    training = record.get("training", {})
    check("training_reached_final_update", training.get("completed_updates") == training.get("configured_updates"))
    check("training_finite", bool(training.get("finite", False)))

    files = record.get("files", {})
    for branch in ["signature", "payload"]:
        row = files.get(branch, {})
        path_value = row.get("path")
        path = Path(path_value) if path_value else None
        exists = bool(path and path.is_file())
        check(f"{branch}_file_exists", exists, path_value)
        if exists:
            observed = sha256_file(path)
            expected = row.get("sha256")
            check(f"{branch}_sha256", expected is None or observed == expected, observed)
        check(f"{branch}_loader_registered", bool(row.get("loader_registered", False)))

    check("hooks_after_load_zero", record.get("hooks_after_load") == 0)
    conditions = record.get("conditions", {})
    reference_shape = None
    for condition in ["Base", "S", "P", "SP"]:
        row = conditions.get(condition, {})
        shape = row.get("shape")
        finite_values = bool(row.get("finite", False))
        check(f"{condition}_forward_completed", bool(row.get("completed", False)))
        check(f"{condition}_finite", finite_values)
        if reference_shape is None and shape is not None:
            reference_shape = shape
        check(f"{condition}_shape_matches", shape is not None and shape == reference_shape, shape)
        for key, value in row.get("metrics", {}).items():
            check(f"{condition}_{key}_finite", isinstance(value, (int, float)) and math.isfinite(value), value)

    passed = all(row["passed"] for row in checks)
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "success_definition": "execution and two-file loader contract only; no quality threshold",
        "checks": checks,
    }

