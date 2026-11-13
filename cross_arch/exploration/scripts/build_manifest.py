#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "MANIFEST.json"
    ignored_prefixes = {"artifacts", "__pycache__", ".git"}
    files = []
    for path in sorted(args.root.rglob("*")):
        relative = path.relative_to(args.root)
        if not path.is_file() or path == output:
            continue
        if any(part in ignored_prefixes for part in relative.parts):
            continue
        if path.suffix == ".pyc":
            continue
        files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path)})
    output.write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

