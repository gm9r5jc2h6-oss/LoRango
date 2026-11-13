#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def requirements() -> list[tuple[str, str]]:
    rows = []
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        requirement = value.split(";", 1)[0].strip()
        if "==" not in requirement:
            raise ValueError(f"requirements.lock entry is not exactly pinned: {line}")
        name, version = requirement.split("==", 1)
        rows.append((name, version))
    if len(rows) != len(set(name.casefold() for name, _version in rows)):
        raise ValueError("requirements.lock contains duplicate package names")
    return rows


def query_batch(rows: list[tuple[str, str]]) -> list[dict]:
    body = json.dumps(
        {
            "queries": [
                {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
                for name, version in rows
            ]
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response).get("results")
    if not isinstance(result, list) or len(result) != len(rows):
        raise RuntimeError("OSV querybatch returned an incomplete result set")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Query OSV for the complete pinned runtime lock.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = requirements()
    responses = query_batch(rows)
    packages = []
    all_ids = set()
    ghsa = set()
    for (name, version), response in zip(rows, responses):
        advisories = []
        for item in response.get("vulns", []):
            ids = {item["id"], *item.get("aliases", [])}
            all_ids.update(ids)
            ghsa.update(value for value in ids if value.startswith("GHSA-"))
            advisories.append(
                {"id": item["id"], "aliases": sorted(item.get("aliases", []))}
            )
        packages.append(
            {"name": name, "version": version, "advisories": advisories}
        )
    result = {
        "scanner": "OSV API v1/querybatch",
        "source": "https://api.osv.dev/v1/querybatch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete requirements.lock closure",
        "packages_scanned": len(packages),
        "packages": packages,
        "unique_osv_id_count_including_aliases": len(all_ids),
        "unique_ghsa_count": len(ghsa),
        "unique_ghsa_ids": sorted(ghsa),
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
