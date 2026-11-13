#!/usr/bin/env python3
"""Merge two normalized Diffusers LoRAs by exact rank concatenation.

The script deliberately accepts only Diffusers/PEFT keys ending in
``.lora_A.weight`` and ``.lora_B.weight``.  It fails on Kohya FLUX names such
as ``lora_unet_double_blocks_*`` instead of silently producing an empty
adapter.  For every module it constructs

    delta = left_scale * (alpha_l / rank_l) * B_l @ A_l
          + right_scale * (alpha_r / rank_r) * B_r @ A_r.

For non-overlapping modules the output preserves the source factors and carries
the requested scale in output alpha/rank.  For overlapping modules it uses a
common loader scale and absorbs only relative scales into B slices.  No SVD
approximation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"
ALPHA_SUFFIX = ".alpha"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state, metadata


@dataclass(frozen=True)
class Factors:
    a: torch.Tensor
    b: torch.Tensor
    alpha: float

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])


def normalized_modules(
    state: dict[str, torch.Tensor], *, source: Path
) -> dict[str, Factors]:
    unsupported = sorted(
        key
        for key in state
        if not key.endswith((A_SUFFIX, B_SUFFIX, ALPHA_SUFFIX))
    )
    if unsupported:
        preview = ", ".join(unsupported[:3])
        raise ValueError(
            f"{source} is not a normalized Diffusers LoRA; unsupported keys: {preview}"
        )

    prefixes = sorted(key[: -len(A_SUFFIX)] for key in state if key.endswith(A_SUFFIX))
    if not prefixes:
        raise ValueError(
            f"{source} contains no Diffusers keys ending in {A_SUFFIX!r}; "
            "convert the source format before merging"
        )

    result: dict[str, Factors] = {}
    for prefix in prefixes:
        a_key = prefix + A_SUFFIX
        b_key = prefix + B_SUFFIX
        if b_key not in state:
            raise ValueError(f"missing paired tensor {b_key!r} in {source}")
        a = state[a_key]
        b = state[b_key]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"only linear 2-D factors are supported: {prefix}")
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"rank mismatch for {prefix}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        alpha_tensor = state.get(prefix + ALPHA_SUFFIX)
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(a.shape[0])
        result[prefix] = Factors(a=a, b=b, alpha=alpha)

    paired_b = {key[: -len(B_SUFFIX)] for key in state if key.endswith(B_SUFFIX)}
    if paired_b != set(prefixes):
        missing = sorted(paired_b.symmetric_difference(prefixes))
        raise ValueError(f"unpaired LoRA factors in {source}: {missing[:3]}")
    return result


def effective_loader_scale(factors: Factors, user_scale: float) -> float:
    return float(user_scale) * factors.alpha / factors.rank


def merge(
    left: dict[str, Factors],
    right: dict[str, Factors],
    *,
    left_scale: float,
    right_scale: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    output: dict[str, torch.Tensor] = {}
    records: list[dict[str, object]] = []
    for prefix in sorted(set(left) | set(right)):
        components: list[tuple[str, Factors, float]] = []
        if prefix in left:
            components.append(("left", left[prefix], left_scale))
        if prefix in right:
            components.append(("right", right[prefix], right_scale))

        input_widths = {int(factors.a.shape[1]) for _, factors, _ in components}
        output_widths = {int(factors.b.shape[0]) for _, factors, _ in components}
        if len(input_widths) != 1 or len(output_widths) != 1:
            raise ValueError(f"incompatible overlapping factors for {prefix}")

        effective_scales = [
            effective_loader_scale(factors, scale)
            for _, factors, scale in components
        ]
        # Preserve the original A/B tensors and loader-side scaling whenever a
        # module comes from only one source.  This minimizes BF16 differences
        # relative to loading that source separately.  For an overlap, use the
        # right source's scale as the common post-matmul scale and absorb only
        # the relative scale into each B slice.
        output_scale = effective_scales[-1]
        if output_scale == 0:
            output_scale = next((value for value in effective_scales if value != 0), 1.0)
        a_out = torch.cat(
            [factors.a.float() for _, factors, _ in components], dim=0
        ).contiguous()
        b_out = torch.cat(
            [
                factors.b.float() * (effective_scales[index] / output_scale)
                for index, (_, factors, _) in enumerate(components)
            ],
            dim=1,
        ).contiguous()
        rank_out = int(a_out.shape[0])
        alpha_out = float(rank_out) * output_scale
        output[prefix + A_SUFFIX] = a_out
        output[prefix + B_SUFFIX] = b_out
        output[prefix + ALPHA_SUFFIX] = torch.tensor(alpha_out, dtype=torch.float32)
        records.append(
            {
                "module": prefix,
                "output_rank": rank_out,
                "output_alpha": alpha_out,
                "output_loader_scale": output_scale,
                "sources": [
                    {
                        "name": name,
                        "rank": factors.rank,
                        "alpha": factors.alpha,
                        "user_scale": scale,
                        "effective_delta_scale": effective_loader_scale(factors, scale),
                    }
                    for name, factors, scale in components
                ],
            }
        )
    return output, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-scale", type=float, default=1.0)
    parser.add_argument("--right-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    left_state, left_metadata = load_safetensors(args.left)
    right_state, right_metadata = load_safetensors(args.right)
    left_modules = normalized_modules(left_state, source=args.left)
    right_modules = normalized_modules(right_state, source=args.right)
    merged, records = merge(
        left_modules,
        right_modules,
        left_scale=args.left_scale,
        right_scale=args.right_scale,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        merged,
        str(args.output),
        metadata={
            "format": "pt",
            "merge": "exact_rank_concatenation",
            "loader_scaling": "per-module alpha preserves source scaling",
        },
    )
    manifest = {
        "status": "COMPLETE",
        "method": "exact_rank_concatenation_no_svd",
        "inputs": {
            "left": {
                "path": str(args.left),
                "sha256": sha256(args.left),
                "scale": args.left_scale,
                "metadata": left_metadata,
            },
            "right": {
                "path": str(args.right),
                "sha256": sha256(args.right),
                "scale": args.right_scale,
                "metadata": right_metadata,
            },
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "tensor_count": len(merged),
            "module_count": len(records),
        },
        "overlap_count": sum(len(row["sources"]) == 2 for row in records),
        "modules": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["status", "method", "output", "overlap_count"]}, indent=2))


if __name__ == "__main__":
    main()
