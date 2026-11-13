#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
DESTINATION = REFERENCE / "REFERENCE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(REFERENCE.rglob("*")):
        if path.is_file() and path != DESTINATION:
            relative = str(path.relative_to(ROOT))
            files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    payload = {
        "schema_version": 2,
        "toolkit_version": "1.4.0",
        "files": files,
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "files": len(files), "path": str(DESTINATION)}))


if __name__ == "__main__":
    main()
