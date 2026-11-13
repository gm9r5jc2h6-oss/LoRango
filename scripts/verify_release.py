#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.4.1"
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
    "security/OSV_SCAN_2026-11-14.json",
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


def source_paths() -> list[Path]:
    if (ROOT / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return [
            ROOT / value.decode("utf-8")
            for value in completed.stdout.split(b"\0")
            if value
        ]
    return [path for path in ROOT.rglob("*") if path.is_file() or path.is_symlink()]


def canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def pinned_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        requirement = value.split(";", 1)[0].strip()
        if "==" not in requirement:
            raise SystemExit(f"requirement is not exactly pinned: {line}")
        name, version = requirement.split("==", 1)
        canonical = canonical_package_name(name)
        if canonical in result:
            raise SystemExit(f"duplicate pinned requirement: {name}")
        result[canonical] = version
    return result


def version_matches_lock(observed: str, expected: str) -> bool:
    return observed == expected or observed.split("+", 1)[0] == expected


def main() -> None:
    sources = source_paths()
    symlinks = [
        str(path.relative_to(ROOT))
        for path in sources
        if path.is_symlink()
    ]
    if symlinks:
        raise SystemExit(f"release contains symlinks: {symlinks}")
    identity_leaks = []
    for path in sorted(sources):
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
        for path in sources
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
    python_paths = [path for path in sources if path.suffix == ".py"]
    json_paths = [path for path in sources if path.suffix == ".json"]
    for path in sorted(python_paths):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted(json_paths):
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
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    repository_url = "https://github.com/gm9r5jc2h6-oss/LoRango"
    if f'repository-code: "{repository_url}"' not in citation:
        raise SystemExit("CITATION.cff lacks the public repository URL")
    if "IDENTIFIER_PENDING" in citation:
        raise SystemExit("CITATION.cff contains a placeholder identifier")

    direct = pinned_requirements(ROOT / "requirements.txt")
    lock = pinned_requirements(ROOT / "requirements.lock")
    pyproject_dependencies = {
        canonical_package_name(name): version
        for name, version in re.findall(
            r'"([A-Za-z0-9_.-]+)==([^";]+)"',
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
    }
    if direct != pyproject_dependencies:
        raise SystemExit("pyproject dependencies do not match requirements.txt")
    missing_from_lock = {
        name: version
        for name, version in direct.items()
        if lock.get(name) != version
    }
    if missing_from_lock:
        raise SystemExit(f"direct requirements do not match the lock: {missing_from_lock}")

    sbom = json.loads((ROOT / "security" / "sbom.cdx.json").read_text(encoding="utf-8"))
    components = sbom.get("components")
    if not isinstance(components, list):
        raise SystemExit("SBOM components are absent")
    sbom_packages: dict[str, str] = {}
    for component in components:
        if not isinstance(component, dict):
            raise SystemExit("SBOM contains a non-object component")
        name = canonical_package_name(str(component.get("name", "")))
        version = component.get("version")
        if not name or not isinstance(version, str) or name in sbom_packages:
            raise SystemExit(f"SBOM contains an invalid/duplicate component: {component}")
        sbom_packages[name] = version
    if set(sbom_packages) != set(lock):
        raise SystemExit(
            "SBOM package set does not match requirements.lock: "
            f"missing={sorted(set(lock) - set(sbom_packages))}, "
            f"extra={sorted(set(sbom_packages) - set(lock))}"
        )
    sbom_version_mismatches = {
        name: {"lock": lock[name], "sbom": sbom_packages[name]}
        for name in lock
        if not version_matches_lock(sbom_packages[name], lock[name])
    }
    if sbom_version_mismatches:
        raise SystemExit(f"SBOM versions do not match the lock: {sbom_version_mismatches}")

    osv = json.loads(
        (ROOT / "security" / "OSV_SCAN_2026-11-14.json").read_text(encoding="utf-8")
    )
    osv_rows = osv.get("packages")
    if not isinstance(osv_rows, list):
        raise SystemExit("OSV scan package list is absent")
    osv_packages = {
        canonical_package_name(str(row.get("name", ""))): row.get("version")
        for row in osv_rows
        if isinstance(row, dict)
    }
    if osv_packages != lock or osv.get("packages_scanned") != len(lock):
        raise SystemExit("OSV scan package set/versions do not match requirements.lock")
    unsafe_checkpoints = [
        str(path.relative_to(ROOT))
        for path in sources
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"}
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
                "python_sources": len(python_paths),
                "json_documents": len(json_paths),
                "identity_markers_found": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
