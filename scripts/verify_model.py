#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify every SDXL weight loaded by the reference run."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reference" / "MODEL_MANIFEST.json",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []

    def verify_group(field: str) -> tuple[dict[str, dict[str, object]], str]:
        verified = {}
        for row in manifest[field]:
            relative = row["relative_path"]
            path = model_dir / relative
            if not path.is_file():
                failures.append({"path": relative, "error": "missing"})
                continue
            actual = sha256(path)
            if actual != row["sha256"] or path.stat().st_size != row["bytes"]:
                failures.append(
                    {
                        "path": relative,
                        "expected_sha256": row["sha256"],
                        "actual_sha256": actual,
                        "expected_bytes": row["bytes"],
                        "actual_bytes": path.stat().st_size,
                    }
                )
            verified[relative] = {"sha256": actual, "bytes": path.stat().st_size}
        return verified, canonical_hash(verified)

    verified_weights, weights_digest = verify_group("weight_files")
    verified_metadata, metadata_digest = verify_group("metadata_files")
    expected_files = {
        row["relative_path"]
        for field in ("weight_files", "metadata_files")
        for row in manifest[field]
    }
    actual_files = {
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(model_dir).parts
        and path.suffix.lower() in {".safetensors", ".json", ".txt", ".model"}
    }
    if actual_files != expected_files:
        failures.append(
            {
                "error": "model runtime file set mismatch",
                "missing": sorted(expected_files - actual_files),
                "extra": sorted(actual_files - expected_files),
            }
        )
    result = {
        "status": "PASS" if not failures else "FAIL",
        "weight_files_checked": len(verified_weights),
        "metadata_files_checked": len(verified_metadata),
        "weights_digest": weights_digest,
        "expected_weights_digest": manifest["weights_digest"],
        "metadata_digest": metadata_digest,
        "expected_metadata_digest": manifest["metadata_digest"],
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if (
        failures
        or weights_digest != manifest["weights_digest"]
        or metadata_digest != manifest["metadata_digest"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
