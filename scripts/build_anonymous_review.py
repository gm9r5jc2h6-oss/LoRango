#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "LoRango_SDXL_Reproducibility_Bundle_v1.4.0_anonymous"
DEFAULT_CHUNK_BYTES = 7 * 1024 * 1024
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


def normalized_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=f"{ARCHIVE_ROOT}/{name}")
    info.size = size
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    return info


def run_preflight() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "submission_check.py")],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic anonymous-review archive with large files split."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    args = parser.parse_args()
    if args.chunk_bytes < 1024 * 1024 or args.chunk_bytes >= 8 * 1024 * 1024:
        raise ValueError("chunk-bytes must be at least 1 MiB and smaller than 8 MiB")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("review archive output must be outside the source tree")
    if output.exists():
        raise FileExistsError(output)
    symlinks = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_symlink()
    ]
    if symlinks:
        raise ValueError(f"release source may not contain symlinks: {symlinks}")
    run_preflight()
    files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and not excluded(path) and path.resolve() != output
    ]
    large_assets: dict[str, dict[str, object]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    relative = str(path.relative_to(ROOT))
                    size = path.stat().st_size
                    if size <= args.chunk_bytes:
                        info = normalized_info(relative, size, path.stat().st_mode & 0o777)
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                        continue
                    chunks = []
                    with path.open("rb") as stream:
                        index = 0
                        while True:
                            payload = stream.read(args.chunk_bytes)
                            if not payload:
                                break
                            chunk_name = f"{relative}.part-{index:03d}"
                            archive.addfile(
                                normalized_info(chunk_name, len(payload)),
                                io.BytesIO(payload),
                            )
                            chunks.append(
                                {
                                    "path": chunk_name,
                                    "bytes": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                }
                            )
                            index += 1
                    large_assets[relative] = {
                        "bytes": size,
                        "sha256": sha256(path),
                        "chunks": chunks,
                    }
                manifest = {
                    "format": "lorango-anonymous-large-assets-v1",
                    "chunk_bytes": args.chunk_bytes,
                    "files": large_assets,
                }
                payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
                archive.addfile(
                    normalized_info("LARGE_ASSETS.json", len(payload)),
                    io.BytesIO(payload),
                )
    print(f"archive={output}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    print(f"split_files={len(large_assets)}")
    print(f"maximum_chunk_bytes={args.chunk_bytes}")


if __name__ == "__main__":
    main()
