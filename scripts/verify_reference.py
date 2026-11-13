#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reference" / "REFERENCE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    expected_files = set(manifest["files"])
    actual_files = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "reference").rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if actual_files != expected_files:
        failures.append(
            {
                "error": "reference file set mismatch",
                "missing": sorted(expected_files - actual_files),
                "extra": sorted(actual_files - expected_files),
            }
        )
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        actual = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    result = {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": len(manifest["files"]),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
