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
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"invalid relative path: {relative!r}")
    candidate = ROOT / relative
    path = candidate.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes review tree: {relative}") from error
    current = candidate
    while current != ROOT:
        if current.is_symlink():
            raise ValueError(f"symbolic links are not allowed: {relative}")
        current = current.parent
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_chunk(row: dict[str, Any]) -> Path:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError("invalid chunk manifest entry")
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
    if not isinstance(manifest, dict) or set(manifest) != {
        "format",
        "chunk_bytes",
        "files",
    }:
        raise ValueError("invalid large-asset manifest structure")
    if manifest.get("format") != "lorango-anonymous-large-assets-v1":
        raise ValueError("unsupported large-asset manifest format")
    if (
        isinstance(manifest.get("chunk_bytes"), bool)
        or not isinstance(manifest.get("chunk_bytes"), int)
        or manifest["chunk_bytes"] < 1024 * 1024
        or manifest["chunk_bytes"] >= 8 * 1024 * 1024
    ):
        raise ValueError("invalid chunk_bytes in large-asset manifest")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("large-asset manifest contains no files")
    restored = []
    validated_chunks: list[Path] = []
    for relative, spec in sorted(manifest["files"].items()):
        if not isinstance(spec, dict) or set(spec) != {"bytes", "sha256", "chunks"}:
            raise ValueError(f"invalid asset manifest entry: {relative}")
        if (
            isinstance(spec["bytes"], bool)
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] < 1
            or not isinstance(spec["sha256"], str)
            or len(spec["sha256"]) != 64
            or not isinstance(spec["chunks"], list)
            or not spec["chunks"]
        ):
            raise ValueError(f"invalid asset metadata: {relative}")
        destination = safe_path(relative)
        chunks = [validate_chunk(row) for row in spec["chunks"]]
        if len(chunks) != len(set(chunks)):
            raise ValueError(f"duplicate chunks for asset: {relative}")
        if sum(path.stat().st_size for path in chunks) != spec["bytes"]:
            raise ValueError(f"chunk sizes do not match asset size: {relative}")
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
