#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.4.0"
REQUIRED = [
    "README.md",
    "SHA256SUMS",
    "ARTIFACT_EVALUATION.md",
    "SUBMISSION_CHECKLIST.md",
    "CLAIMS_AND_EVIDENCE.md",
    "REPRODUCIBILITY.md",
    "SUBMISSION_STATUS.json",
    "MERGE_VERIFICATION.json",
    "CROSS_ARCH_RELEASE_STATUS.json",
    "schemas/run-evidence.schema.json",
    "SECURITY_AND_ETHICS.md",
    "CITATION.cff",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "configs/train_pair.example.json",
    "configs/train_pair.schema.json",
    "examples/train_data.example.json",
    "assets/scaffold_init.safetensors",
    "assets/scaffold_init.json",
    "assets/carrier_catalog.json",
    "assets/ASSET_MANIFEST.json",
    "security/sbom.cdx.json",
    "security/OSV_SCAN_2026-11-07.json",
    "security/REACHABILITY.md",
    "reference/MODEL_MANIFEST.json",
    "reference/REFERENCE_MANIFEST.json",
    "reference/inputs/utility_s.safetensors",
    "reference/inputs/utility_p.safetensors",
    "reference/outputs/modified_s.safetensors",
    "reference/outputs/modified_p.safetensors",
    "reference/images/overview.png",
    "lorango_sdxl/assets/scaffold_init.safetensors",
    "lorango_sdxl/assets/scaffold_init.json",
    "lorango_sdxl/assets/carrier_catalog.json",
    "train_pair.py",
    "scripts/submission_check.py",
    "scripts/build_anonymous_review.py",
    "scripts/reassemble_large_assets.py",
    "scripts/verify_cross_arch.py",
    "docs/ANONYMOUS_REVIEW.md",
    "evidence/README.md",
    "evidence/V1_4_GPU_EVIDENCE_SUMMARY.json",
]


FORBIDDEN_IDENTITY_MARKERS = tuple(
    bytes.fromhex(value)
    for value in (
        "2f686f6d652f6c7264",
        "2f55736572732f6c6972756e646f6e67",
        "6c6972756e646f6e67",
        "7072697661746572656c61792e6170706c6569642e636f6d",
        "31302e3137362e35362e313136",
    )
)
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
    symlinks = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise SystemExit(f"release contains symlinks: {symlinks}")
    identity_leaks = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or excluded(path):
            continue
        relative = str(path.relative_to(ROOT))
        with path.open("rb") as stream:
            payload = stream.read()
        for marker in FORBIDDEN_IDENTITY_MARKERS:
            if marker.lower() in payload.lower():
                identity_leaks.append(
                    {"path": relative, "marker_sha256": hashlib.sha256(marker).hexdigest()}
                )
    if identity_leaks:
        raise SystemExit(f"release contains private identity markers: {identity_leaks}")
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    manifest = json.loads((ROOT / "assets" / "ASSET_MANIFEST.json").read_text())
    for name, spec in manifest["assets"].items():
        actual = sha256(ROOT / "assets" / name)
        if actual != spec["sha256"]:
            raise SystemExit(f"asset checksum mismatch: {name}: {actual}")
    checksums = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    release_paths = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and not any(
            part.endswith(".egg-info") for part in path.relative_to(ROOT).parts
        )
    }
    if set(checksums) != release_paths:
        raise SystemExit(
            "SHA256SUMS file set mismatch; run scripts/update_release_checksums.py"
        )
    for relative, expected in checksums.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"release checksum mismatch: {relative}: {actual}")
    for path in sorted(ROOT.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted(ROOT.rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    version_documents = {
        "SUBMISSION_STATUS.json": json.loads(
            (ROOT / "SUBMISSION_STATUS.json").read_text(encoding="utf-8")
        ).get("toolkit_version"),
        ".zenodo.json": json.loads(
            (ROOT / ".zenodo.json").read_text(encoding="utf-8")
        ).get("version"),
        "security/sbom.cdx.json": json.loads(
            (ROOT / "security" / "sbom.cdx.json").read_text(encoding="utf-8")
        ).get("metadata", {}).get("component", {}).get("version"),
        "VERIFICATION.json": json.loads(
            (ROOT / "VERIFICATION.json").read_text(encoding="utf-8")
        ).get("toolkit_version"),
        "reference/REFERENCE_MANIFEST.json": json.loads(
            (ROOT / "reference" / "REFERENCE_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        ).get("toolkit_version"),
    }
    mismatched_versions = {
        name: value for name, value in version_documents.items() if value != VERSION
    }
    if mismatched_versions:
        raise SystemExit(f"version metadata mismatch: {mismatched_versions}")
    text_versions = {
        "pyproject.toml": f'version = "{VERSION}"',
        "lorango_sdxl/__init__.py": f'__version__ = "{VERSION}"',
        "CITATION.cff": f"version: {VERSION}",
        "README.md": f"Artifact release v{VERSION}",
    }
    for relative, marker in text_versions.items():
        if marker not in (ROOT / relative).read_text(encoding="utf-8"):
            raise SystemExit(f"version marker mismatch: {relative}")
    unsafe_checkpoints = [
        str(path.relative_to(ROOT))
        for suffix in ("*.pt", "*.pth", "*.ckpt")
        for path in ROOT.rglob(suffix)
    ]
    if unsafe_checkpoints:
        raise SystemExit(f"pickle-based checkpoints remain in release: {unsafe_checkpoints}")
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "lorango_sdxl").glob("*.py")
    )
    if "torch.load(" in runtime_sources or "torch.save(" in runtime_sources:
        raise SystemExit("runtime still contains pickle-based torch serialization")
    print(
        json.dumps(
            {
                "status": "PASS",
                "required_files": len(REQUIRED),
                "verified_assets": len(manifest["assets"]),
                "python_sources": len(list(ROOT.rglob("*.py"))),
                "json_documents": len(list(ROOT.rglob("*.json"))),
                "identity_markers_found": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
