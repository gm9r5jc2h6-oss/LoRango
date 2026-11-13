#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "LARGE_ASSETS.json"


def safe_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes review tree: {relative}") from error
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_chunk(row: dict[str, Any]) -> Path:
    path = safe_path(str(row["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["bytes"]):
        raise ValueError(f"chunk size mismatch: {path}")
    if sha256(path) != row["sha256"]:
        raise ValueError(f"chunk hash mismatch: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore and verify large assets from anonymous-review chunks."
    )
    parser.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="remove chunk files and LARGE_ASSETS.json after every asset passes",
    )
    args = parser.parse_args()
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("format") != "lorango-anonymous-large-assets-v1":
        raise ValueError("unsupported large-asset manifest format")
    restored = []
    validated_chunks: list[Path] = []
    for relative, spec in sorted(manifest["files"].items()):
        destination = safe_path(relative)
        chunks = [validate_chunk(row) for row in spec["chunks"]]
        validated_chunks.extend(chunks)
        if destination.exists():
            if destination.stat().st_size != int(spec["bytes"]) or sha256(destination) != spec["sha256"]:
                raise FileExistsError(f"existing destination does not match manifest: {destination}")
            restored.append(relative)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".reassemble.tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        digest = hashlib.sha256()
        total = 0
        with temporary.open("xb") as output:
            for chunk_path in chunks:
                with chunk_path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        total += len(block)
        if total != int(spec["bytes"]) or digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"reconstructed asset mismatch: {relative}")
        os.replace(temporary, destination)
        restored.append(relative)
    if args.cleanup_parts:
        for path in validated_chunks:
            path.unlink()
        MANIFEST.unlink()
    print(
        json.dumps(
            {
                "status": "PASS",
                "restored_files": restored,
                "cleanup_parts": args.cleanup_parts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
