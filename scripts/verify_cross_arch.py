#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "cross_arch" / "reproducibility"
MANIFEST = CAPSULE / "MANIFEST.sha256"
EXPLORATION = ROOT / "cross_arch" / "exploration"
EXPLORATION_MANIFEST = EXPLORATION / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures = []
    checked = 0
    recorded_capsule_paths = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if relative in recorded_capsule_paths:
            failures.append({"path": relative, "error": "duplicate manifest entry"})
            continue
        recorded_capsule_paths.add(relative)
        path = (CAPSULE / relative).resolve()
        try:
            path.relative_to(CAPSULE)
        except ValueError:
            failures.append({"path": relative, "error": "path escapes capsule"})
            continue
        checked += 1
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    actual_capsule_paths = {
        str(path.relative_to(CAPSULE))
        for path in CAPSULE.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if recorded_capsule_paths != actual_capsule_paths:
        failures.append(
            {
                "manifest": "cross_arch/reproducibility/MANIFEST.sha256",
                "error": "file set mismatch",
                "missing": sorted(actual_capsule_paths - recorded_capsule_paths),
                "extra": sorted(recorded_capsule_paths - actual_capsule_paths),
            }
        )

    exploration_manifest = json.loads(
        EXPLORATION_MANIFEST.read_text(encoding="utf-8")
    )
    rows = exploration_manifest.get("files")
    if exploration_manifest.get("schema_version") != 1 or not isinstance(rows, list):
        failures.append({"manifest": "cross_arch/exploration/MANIFEST.json", "error": "invalid structure"})
        rows = []
    recorded_exploration_paths = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            failures.append({"manifest": "cross_arch/exploration/MANIFEST.json", "error": "invalid entry"})
            continue
        relative = row["path"]
        if not isinstance(relative, str) or relative in recorded_exploration_paths:
            failures.append({"path": relative, "error": "duplicate/invalid exploration entry"})
            continue
        recorded_exploration_paths.add(relative)
        path = (EXPLORATION / relative).resolve()
        try:
            path.relative_to(EXPLORATION)
        except ValueError:
            failures.append({"path": relative, "error": "path escapes exploration tree"})
            continue
        checked += 1
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            failures.append({"path": relative, "error": "hash/size mismatch"})
    ignored = {"artifacts", "__pycache__", ".git"}
    actual_exploration_paths = {
        str(path.relative_to(EXPLORATION))
        for path in EXPLORATION.rglob("*")
        if path.is_file()
        and path != EXPLORATION_MANIFEST
        and path.suffix != ".pyc"
        and not ignored.intersection(path.relative_to(EXPLORATION).parts)
    }
    if recorded_exploration_paths != actual_exploration_paths:
        failures.append(
            {
                "manifest": "cross_arch/exploration/MANIFEST.json",
                "error": "file set mismatch",
                "missing": sorted(actual_exploration_paths - recorded_exploration_paths),
                "extra": sorted(recorded_exploration_paths - actual_exploration_paths),
            }
        )
    result = {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": checked,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
