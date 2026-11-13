#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "LoRango_SDXL_Reproducibility_Bundle_v1.4.0"
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
    parser = argparse.ArgumentParser(
        description="Build the immutable SDXL reproducibility bundle."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("release output must be outside the artifact source tree")
    if output.exists():
        raise FileExistsError(output)
    symlinks = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise ValueError(f"release source may not contain symlinks: {symlinks}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not excluded(path)
        and path.resolve() != output
    ]
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    info = archive.gettarinfo(
                        str(path),
                        arcname=f"{ARCHIVE_ROOT}/{path.relative_to(ROOT)}",
                    )
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    print(f"archive={output}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
