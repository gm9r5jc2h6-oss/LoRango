#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "SHA256SUMS"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "runs",
    "build",
    "dist",
}


def excluded(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return bool(EXCLUDED_PARTS.intersection(parts)) or any(
        part.endswith(".egg-info") for part in parts
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path != DESTINATION
        and not excluded(path)
    ]
    payload = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in paths
    )
    DESTINATION.write_text(payload, encoding="utf-8")
    print(f"files={len(paths)}")
    print(f"path={DESTINATION}")


if __name__ == "__main__":
    main()
