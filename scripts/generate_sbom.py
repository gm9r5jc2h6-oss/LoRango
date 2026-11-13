#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def root_requirements() -> list[str]:
    names = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(canonicalize_name(line.split("==", 1)[0]))
    return names


def installed() -> dict[str, metadata.Distribution]:
    return {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }


def active_dependencies(dist: metadata.Distribution) -> list[str]:
    result = []
    for value in dist.requires or []:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.extras:
            continue
        result.append(canonicalize_name(requirement.name))
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CycloneDX runtime SBOM.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distributions = installed()
    roots = root_requirements()
    missing = [name for name in roots if name not in distributions]
    if missing:
        raise SystemExit(f"required packages are not installed: {missing}")
    closure = set()
    pending = list(roots)
    graph: dict[str, list[str]] = {}
    while pending:
        name = pending.pop()
        if name in closure or name not in distributions:
            continue
        closure.add(name)
        dependencies = [
            item for item in active_dependencies(distributions[name]) if item in distributions
        ]
        graph[name] = dependencies
        pending.extend(dependencies)
    components = []
    for name in sorted(closure):
        dist = distributions[name]
        display_name = dist.metadata["Name"]
        version = dist.version
        components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:pypi/{name}@{version}",
                "name": display_name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
            }
        )
    dependencies = [
        {
            "ref": f"pkg:pypi/{name}@{distributions[name].version}",
            "dependsOn": [
                f"pkg:pypi/{dep}@{distributions[dep].version}"
                for dep in graph.get(name, [])
                if dep in closure
            ],
        }
        for name in sorted(closure)
    ]
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:8b8a77d6-d337-4c75-a6be-07eff902da13",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {
                "type": "application",
                "bom-ref": "pkg:pypi/lorango-sdxl-training@1.4.0",
                "name": "lorango-sdxl-training",
                "version": "1.4.0",
            },
            "properties": [
                {
                    "name": "evidence.scope",
                    "value": "runtime dependency closure pinned in requirements.lock",
                }
            ],
        },
        "components": components,
        "dependencies": dependencies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "components": len(components), "output": str(args.output)}))


if __name__ == "__main__":
    main()
