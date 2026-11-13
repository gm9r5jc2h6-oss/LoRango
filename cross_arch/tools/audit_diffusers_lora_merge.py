#!/usr/bin/env python3
"""Verify that a merged LoRA represents the exact sum of two source deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from merge_diffusers_lora import load_safetensors, normalized_modules


def effective_scale(factors, user_scale: float) -> float:
    return float(user_scale) * factors.alpha / factors.rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--left-scale", type=float, default=1.0)
    parser.add_argument("--right-scale", type=float, default=1.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    left = normalized_modules(load_safetensors(args.left)[0], source=args.left)
    right = normalized_modules(load_safetensors(args.right)[0], source=args.right)
    merged = normalized_modules(load_safetensors(args.merged)[0], source=args.merged)
    expected_modules = set(left) | set(right)
    if set(merged) != expected_modules:
        raise ValueError("merged module set differs from source union")

    rows = []
    for prefix in sorted(expected_modules):
        sources = []
        if prefix in left:
            sources.append((left[prefix], args.left_scale))
        if prefix in right:
            sources.append((right[prefix], args.right_scale))
        actual = merged[prefix]
        expected_rank = sum(factors.rank for factors, _ in sources)
        if actual.rank != expected_rank:
            raise ValueError(
                f"rank mismatch for {prefix}: {actual.rank} != {expected_rank}"
            )
        expected_a = torch.cat([factors.a.float() for factors, _ in sources], dim=0)
        a_difference = (actual.a.float() - expected_a).abs()
        actual_loader_scale = actual.alpha / actual.rank
        column = 0
        b_differences = []
        for factors, user_scale in sources:
            width = factors.rank
            # Compare each output B slice after its common output scaling with
            # the corresponding source B after its original effective scaling.
            actual_effective_b = (
                actual.b[:, column : column + width].float() * actual_loader_scale
            )
            expected_effective_b = (
                factors.b.float() * effective_scale(factors, user_scale)
            )
            b_differences.append((actual_effective_b - expected_effective_b).abs())
            column += width
        b_max = max(float(value.max()) for value in b_differences)
        b_mean = sum(float(value.mean()) for value in b_differences) / len(b_differences)
        rows.append(
            {
                "module": prefix,
                "maximum_absolute_difference": max(
                    float(a_difference.max()),
                    b_max,
                ),
                "mean_absolute_difference": (
                    float(a_difference.mean()) + b_mean
                )
                / 2.0,
                "audit_basis": "factor concatenation; implies exact delta sum algebraically",
            }
        )

    maximum = max(row["maximum_absolute_difference"] for row in rows)
    report = {
        "status": "PASS" if maximum <= args.atol else "FAIL",
        "threshold": args.atol,
        "maximum_absolute_difference": maximum,
        "module_count": len(rows),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
