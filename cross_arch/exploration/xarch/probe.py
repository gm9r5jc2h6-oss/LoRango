from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .backend import ArchitectureBackend, RolloutState


@dataclass
class ProbeCacheItem:
    state: RolloutState
    conditioning: Mapping[str, Any]
    s_clean: Any
    p_clean: Any
    sp_clean: Any
    target: Any


def _cosine(torch: Any, left: Any, right: Any) -> Any:
    return torch.nn.functional.cosine_similarity(
        left.float().flatten(), right.float().flatten(), dim=0, eps=1e-8
    )


def _loss_and_metrics(torch: Any, item: ProbeCacheItem, outputs: Mapping[str, Any], weights: Mapping[str, float]):
    target = item.target.float()
    d_s = outputs["S"].float() - item.s_clean.float()
    d_p = outputs["P"].float() - item.p_clean.float()
    d_sp = outputs["SP"].float() - item.sp_clean.float()
    interaction = d_sp - d_s - d_p
    energy = target.square().mean().clamp_min(1e-7)
    sp_nmse = (d_sp - target).square().mean() / energy
    interaction_nmse = (interaction - target).square().mean() / energy
    direction = 1.0 - _cosine(torch, d_sp, target)
    s_drift = d_s.square().mean() / energy
    p_drift = d_p.square().mean() / energy
    loss = (
        float(weights["sp_nmse_weight"]) * sp_nmse
        + float(weights["interaction_nmse_weight"]) * interaction_nmse
        + float(weights["direction_weight"]) * direction
        + float(weights["single_adapter_preservation_weight"]) * (s_drift + p_drift)
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "sp_nmse": float(sp_nmse.detach().cpu()),
        "interaction_nmse": float(interaction_nmse.detach().cpu()),
        "sp_cosine": float((1.0 - direction).detach().cpu()),
        "interaction_cosine": float(_cosine(torch, interaction, target).detach().cpu()),
        "s_drift_ratio": float(s_drift.detach().cpu()),
        "p_drift_ratio": float(p_drift.detach().cpu()),
        "sp_amplitude": math.sqrt(float((d_sp.square().mean() / energy).detach().cpu())),
    }
    return loss, metrics


def run_short_probe(
    backend: ArchitectureBackend,
    candidate: Mapping[str, Any],
    rank: int,
    cache: Sequence[ProbeCacheItem],
    policy: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run the policy-defined harmless short probe using a backend.

    Torch is imported lazily so inventory/ranking/demo workflows remain
    dependency-free. The backend owns model-specific branch injection and
    condition semantics; this function owns the public optimizer and loss.
    """

    if not cache:
        raise ValueError("short probe requires at least one cache item")
    import torch  # type: ignore

    random_generator = random.Random(seed)
    torch.manual_seed(seed)
    probe = backend.attach_probe(candidate, rank, seed)
    parameters = list(backend.trainable_probe_parameters(probe))
    if not parameters:
        raise ValueError("backend returned no trainable probe parameters")
    optimization = policy["probe"]["optimization"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(v) for v in optimization["betas"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    trace = []
    for update in range(1, int(policy["probe"]["updates_per_candidate_rank"]) + 1):
        item_index = random_generator.randrange(len(cache))
        item = cache[item_index]
        optimizer.zero_grad(set_to_none=True)
        outputs = {
            "S": backend.predict(item.state, item.conditioning["S"], frozenset({"S"})),
            "P": backend.predict(item.state, item.conditioning["P"], frozenset({"P"})),
            "SP": backend.predict(item.state, item.conditioning["SP"], frozenset({"S", "P"})),
        }
        loss, metrics = _loss_and_metrics(torch, item, outputs, optimization)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(optimization["gradient_clip_norm"])
        )
        optimizer.step()
        post_step = getattr(backend, "post_probe_step", None)
        if post_step is not None:
            post_step(probe)
        trace.append(
            {
                "update": update,
                "item_index": item_index,
                "grad_norm": float(grad_norm.detach().cpu()),
                **metrics,
            }
        )

    aggregate = {key: 0.0 for key in [
        "sp_nmse", "interaction_nmse", "sp_cosine", "interaction_cosine",
        "s_drift_ratio", "p_drift_ratio", "sp_amplitude",
    ]}
    with torch.no_grad():
        for item in cache:
            outputs = {
                "S": backend.predict(item.state, item.conditioning["S"], frozenset({"S"})),
                "P": backend.predict(item.state, item.conditioning["P"], frozenset({"P"})),
                "SP": backend.predict(item.state, item.conditioning["SP"], frozenset({"S", "P"})),
            }
            _, metrics = _loss_and_metrics(torch, item, outputs, optimization)
            for key in aggregate:
                aggregate[key] += metrics[key] / len(cache)

    finite = all(math.isfinite(value) for value in aggregate.values())
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": rank,
        "seed": seed,
        "finite": finite,
        "exportable": bool(candidate.get("exportable", False)),
        "loader_supported": bool(candidate.get("loader_supported", True)),
        **aggregate,
        "trace": trace,
    }

