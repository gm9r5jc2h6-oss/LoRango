#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_check(name: str, command: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "name": name,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "command": command,
        "output_tail": output[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reviewer-facing LoRango submission checks in explicit tiers."
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also run dependency-backed asset and unit tests",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    checks = [
        run_check("release_integrity", [sys.executable, "scripts/verify_release.py"]),
        run_check("reference_integrity", [sys.executable, "scripts/verify_reference.py"]),
        run_check(
            "negative_verifier_test",
            [sys.executable, "-m", "unittest", "tests/test_verify_run_negative.py", "-v"],
        ),
    ]
    if args.runtime:
        checks.extend(
            [
                run_check("asset_self_test", [sys.executable, "scripts/self_test.py"]),
                run_check(
                    "runtime_unit_tests",
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                ),
            ]
        )
    if args.model_dir:
        checks.append(
            run_check(
                "reference_model",
                [
                    sys.executable,
                    "scripts/verify_model.py",
                    "--model-dir",
                    str(args.model_dir.expanduser().resolve()),
                ],
            )
        )
    if args.run_dir:
        command = [
            sys.executable,
            "scripts/verify_run.py",
            "--run-dir",
            str(args.run_dir.expanduser().resolve()),
        ]
        if args.allow_smoke:
            command.append("--allow-smoke")
        checks.append(run_check("run_evidence", command))

    result = {
        "schema_version": 1,
        "toolkit_version": "1.4.0",
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
        "tiers_requested": {
            "static": True,
            "runtime": args.runtime,
            "model": args.model_dir is not None,
            "run": args.run_dir is not None,
        },
        "checks": checks,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_output:
        destination = args.json_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
