#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPSULE = ROOT / "cross_arch" / "reproducibility"
MANIFEST = CAPSULE / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
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
