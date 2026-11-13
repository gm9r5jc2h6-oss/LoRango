from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .carrier import make_trainable_pair
from .checkpoint import load_checkpoint, save_checkpoint
from .runtime import (
    autocast_context,
    encode_prompt,
    guided_prediction,
    initial_latents,
    scheduler_for,
    tensor_rms,
    write_json,
)


def configure_determinism(config: dict[str, Any]) -> None:
    if not bool(config["training"].get("deterministic", True)):
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _load_cover(pipe, path: Path, adapter_name: str) -> None:
    pipe.load_lora_weights(
        str(path.parent), weight_name=path.name, adapter_name=adapter_name
    )


def predict(
    pipe,
    pair,
    config: dict[str, Any],
    latents: torch.Tensor,
    conditioning,
    step: int,
    cover_weights: list[float],
    signature: bool,
    payload: bool,
    grad: bool,
):
    pipe.unet.set_adapters(["cover_s", "cover_p"], cover_weights)
    pair.enable(signature, payload)
    scheduler = scheduler_for(pipe, config)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context, autocast_context(config):
        return guided_prediction(
            pipe,
            scheduler,
            latents,
            scheduler.timesteps[int(step)],
            conditioning,
            config,
        ).float()


@torch.no_grad()
def collect_states(
    pipe,
    pair,
    config: dict[str, Any],
    examples: list[dict[str, str]],
    seeds: list[int],
    conditioning,
    label: str,
) -> list[dict[str, Any]]:
    record_steps = tuple(int(value) for value in config["runtime"]["record_steps"])
    rows = []
    total = len(examples) * len(seeds)
    completed = 0
    for example in examples:
        for seed in seeds:
            scheduler = scheduler_for(pipe, config)
            latents = initial_latents(pipe, scheduler, config, seed)
            for step, timestep in enumerate(scheduler.timesteps):
                if step in record_steps:
                    rows.append(
                        {
                            "id": example["id"],
                            "seed": int(seed),
                            "step": int(step),
                            "latents": latents.detach().cpu().to(torch.float16),
                        }
                    )
                if step == record_steps[-1]:
                    break
                clean = predict(
                    pipe,
                    pair,
                    config,
                    latents,
                    conditioning[(example["id"], "base")],
                    step,
                    [1.0, 1.0],
                    False,
                    False,
                    False,
                )
                scheduler.scale_model_input(latents, timestep)
                latents = scheduler.step(
                    clean.to(latents.dtype), timestep, latents, return_dict=True
                ).prev_sample
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": "state_collection",
                        "split": label,
                        "completed": completed,
                        "total": total,
                    }
                ),
                flush=True,
            )
    return rows


@torch.no_grad()
def cache_targets(pipe, pair, config, states, conditioning):
    cached = []
    for index, item in enumerate(states, 1):
        latents = item["latents"].float().to(config["device"])
        step = int(item["step"])
        clean_sp = predict(
            pipe,
            pair,
            config,
            latents,
            conditioning[(item["id"], "base")],
            step,
            [1.0, 1.0],
            False,
            False,
            False,
        )
        teacher = predict(
            pipe,
            pair,
            config,
            latents,
            conditioning[(item["id"], "target")],
            step,
            [0.0, 0.0],
            False,
            False,
            False,
        )
        cached.append(
            {
                **item,
                "clean_sp": clean_sp.detach().cpu().to(torch.float16),
                "target_residual": (teacher - clean_sp).detach().cpu().to(torch.float16),
            }
        )
        if index % 24 == 0 or index == len(states):
            print(
                json.dumps(
                    {"stage": "target_cache", "completed": index, "total": len(states)}
                ),
                flush=True,
            )
    return cached


def sample_weight(step: int) -> float:
    if step <= 6:
        return 4.0
    if step <= 13:
        return 2.0
    return 1.0


@torch.no_grad()
def audit(pipe, pair, config, states, conditioning) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(states, 1):
        latents = item["latents"].float().to(config["device"])
        step = int(item["step"])
        base_cond = conditioning[(item["id"], "base")]
        clean_sp = predict(
            pipe, pair, config, latents, base_cond, step, [1.0, 1.0], False, False, False
        )
        teacher = predict(
            pipe,
            pair,
            config,
            latents,
            conditioning[(item["id"], "target")],
            step,
            [0.0, 0.0],
            False,
            False,
            False,
        )
        modified_sp = predict(
            pipe, pair, config, latents, base_cond, step, [1.0, 1.0], True, True, False
        )
        clean_s = predict(
            pipe, pair, config, latents, base_cond, step, [1.0, 0.0], False, False, False
        )
        modified_s = predict(
            pipe, pair, config, latents, base_cond, step, [1.0, 0.0], True, False, False
        )
        clean_p = predict(
            pipe, pair, config, latents, base_cond, step, [0.0, 1.0], False, False, False
        )
        modified_p = predict(
            pipe, pair, config, latents, base_cond, step, [0.0, 1.0], False, True, False
        )
        target = teacher - clean_sp
        actual = modified_sp - clean_sp
        target_rms = max(tensor_rms(target), 1e-12)
        rows.append(
            {
                "id": item["id"],
                "seed": int(item["seed"]),
                "step": step,
                "cosine": float(
                    F.cosine_similarity(
                        target.flatten().double(), actual.flatten().double(), dim=0, eps=1e-12
                    )
                ),
                "nrmse": tensor_rms(actual - target) / target_rms,
                "amplitude_ratio": tensor_rms(actual) / target_rms,
                "s_relative_drift": tensor_rms(modified_s - clean_s)
                / max(tensor_rms(clean_s), 1e-12),
                "p_relative_drift": tensor_rms(modified_p - clean_p)
                / max(tensor_rms(clean_p), 1e-12),
            }
        )
        if index % 12 == 0 or index == len(states):
            print(
                json.dumps({"stage": "calibration", "completed": index, "total": len(states)}),
                flush=True,
            )
    return {
        "items": len(rows),
        "mean_cosine": float(np.mean([row["cosine"] for row in rows])),
        "p10_cosine": float(np.quantile([row["cosine"] for row in rows], 0.10)),
        "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
        "p90_nrmse": float(np.quantile([row["nrmse"] for row in rows], 0.90)),
        "mean_amplitude_ratio": float(np.mean([row["amplitude_ratio"] for row in rows])),
        "s_mean_relative_drift": float(np.mean([row["s_relative_drift"] for row in rows])),
        "p_mean_relative_drift": float(np.mean([row["p_relative_drift"] for row in rows])),
        "rows": rows,
    }


def _gate_results(calibration: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    thresholds = config["gates"]
    drift_limit = float(thresholds["single_relative_drift_max"])
    return {
        "mean_cosine": calibration["mean_cosine"] >= float(thresholds["mean_cosine_min"]),
        "mean_nrmse": calibration["mean_nrmse"] <= float(thresholds["mean_nrmse_max"]),
        "signature_only_drift": calibration["s_mean_relative_drift"] <= drift_limit,
        "payload_only_drift": calibration["p_mean_relative_drift"] <= drift_limit,
    }


def train_head(pipe, config: dict[str, Any], data: dict[str, list[dict[str, str]]]):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    conditioning = {}
    all_examples = data["train"] + data["calibration"]
    for index, example in enumerate(all_examples, 1):
        conditioning[(example["id"], "base")] = encode_prompt(pipe, example["base"], config)
        conditioning[(example["id"], "target")] = encode_prompt(pipe, example["target"], config)
        print(
            json.dumps(
                {
                    "stage": "prompt_encoding",
                    "completed": index,
                    "total": len(all_examples),
                    "id": example["id"],
                }
            ),
            flush=True,
        )

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.vae.to("cpu")
    _load_cover(pipe, Path(config["signature_cover_lora"]), "cover_s")
    _load_cover(pipe, Path(config["payload_cover_lora"]), "cover_p")
    for parameter in pipe.unet.parameters():
        parameter.requires_grad_(False)

    scaffold = load_checkpoint(config["scaffold_init"])
    pair = make_trainable_pair(pipe, scaffold, config)
    action_parameters = [layer.actions for layer in pair.layers]
    train_readers = bool(config["training"]["train_readers"])
    reader_parameters = [layer.value_readers for layer in pair.layers] if train_readers else []
    for layer in pair.layers:
        layer.actions.requires_grad_(True)
        layer.value_readers.requires_grad_(train_readers)
    parameter_groups = [
        {"params": action_parameters, "lr": float(config["training"]["action_lr"])}
    ]
    if reader_parameters:
        parameter_groups.append(
            {"params": reader_parameters, "lr": float(config["training"]["reader_lr"])}
        )
    parameters = action_parameters + reader_parameters
    optimizer = torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.99), weight_decay=1e-4
    )
    fixed_seed = int(config["training"]["fixed_seed"])
    random.seed(fixed_seed)
    np.random.seed(fixed_seed % (2**32 - 1))
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    configure_determinism(config)
    trace = []
    try:
        fit_states = collect_states(
            pipe,
            pair,
            config,
            data["train"],
            [int(value) for value in config["training"]["train_seeds"]],
            conditioning,
            "train",
        )
        calibration_states = collect_states(
            pipe,
            pair,
            config,
            data["calibration"],
            [int(value) for value in config["training"]["calibration_seeds"]],
            conditioning,
            "calibration",
        )
        cached = cache_targets(pipe, pair, config, fit_states, conditioning)
        weights = [sample_weight(int(item["step"])) for item in cached]
        total_steps = int(config["training"]["optimization_steps"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        for optimization_step in range(1, total_steps + 1):
            item = random.choices(cached, weights=weights, k=1)[0]
            latents = item["latents"].float().to(config["device"])
            clean_sp = item["clean_sp"].float().to(config["device"])
            target = item["target_residual"].float().to(config["device"])
            optimizer.zero_grad(set_to_none=True)
            modified_sp = predict(
                pipe,
                pair,
                config,
                latents,
                conditioning[(item["id"], "base")],
                int(item["step"]),
                [1.0, 1.0],
                True,
                True,
                True,
            )
            actual = modified_sp - clean_sp
            target_energy = target.square().mean().clamp_min(1e-7)
            nmse = (actual - target).square().mean() / target_energy
            difference = target.abs().mean(dim=1, keepdim=True)
            mask = F.avg_pool2d(difference, kernel_size=3, stride=1, padding=1)
            mask = (mask / mask.mean().clamp_min(1e-8)).clamp(0.25, 4.0)
            masked_nmse = ((actual - target).square() * mask).mean() / target_energy
            direction = 1.0 - F.cosine_similarity(
                actual.flatten(), target.flatten(), dim=0, eps=1e-6
            )
            loss = nmse + 0.50 * masked_nmse + 0.50 * direction
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 2.0))
            optimizer.step()
            row = {
                "optimization_step": optimization_step,
                "id": item["id"],
                "diffusion_step": int(item["step"]),
                "loss": float(loss.detach()),
                "nmse": float(nmse.detach()),
                "masked_nmse": float(masked_nmse.detach()),
                "direction_loss": float(direction.detach()),
                "gradient_norm": gradient_norm,
            }
            trace.append(row)
            if optimization_step == 1 or optimization_step % 100 == 0:
                print(json.dumps({"stage": "training", **row}), flush=True)
            if checkpoint_every > 0 and optimization_step % checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_{optimization_step:05d}.safetensors",
                    pair.state(optimization_step),
                )
                write_json(output_dir / "training_trace.json", trace)

        checkpoint_path = output_dir / "checkpoint_trained.safetensors"
        save_checkpoint(checkpoint_path, pair.state(total_steps))
        write_json(output_dir / "training_trace.json", trace)
        calibration = audit(pipe, pair, config, calibration_states, conditioning)
        write_json(output_dir / "CALIBRATION.json", calibration)
        gates = _gate_results(calibration, config)
    finally:
        pair.close()

    return checkpoint_path, calibration, gates


def _predict_loaded_adapters(
    pipe,
    config: dict[str, Any],
    latents: torch.Tensor,
    conditioning,
    step: int,
    weights: list[float],
):
    names = ["clean_s", "clean_p", "modified_s", "modified_p"]
    pipe.unet.set_adapters(names, weights)
    scheduler = scheduler_for(pipe, config)
    with torch.no_grad(), autocast_context(config):
        return guided_prediction(
            pipe,
            scheduler,
            latents,
            scheduler.timesteps[int(step)],
            conditioning,
            config,
        ).float()


@torch.no_grad()
def _collect_export_states(pipe, config, examples, seeds, conditioning):
    record_steps = tuple(int(value) for value in config["runtime"]["record_steps"])
    rows = []
    for example in examples:
        for seed in seeds:
            scheduler = scheduler_for(pipe, config)
            latents = initial_latents(pipe, scheduler, config, int(seed))
            for step, timestep in enumerate(scheduler.timesteps):
                if step in record_steps:
                    rows.append(
                        {
                            "id": example["id"],
                            "seed": int(seed),
                            "step": int(step),
                            "latents": latents.detach().cpu().to(torch.float16),
                        }
                    )
                if step == record_steps[-1]:
                    break
                prediction = _predict_loaded_adapters(
                    pipe,
                    config,
                    latents,
                    conditioning[(example["id"], "base")],
                    step,
                    [1.0, 1.0, 0.0, 0.0],
                )
                scheduler.scale_model_input(latents, timestep)
                latents = scheduler.step(
                    prediction.to(latents.dtype), timestep, latents, return_dict=True
                ).prev_sample
    return rows


def audit_exported_pair(
    pipe,
    config: dict[str, Any],
    data: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Reload final files through Diffusers and repeat held-out calibration.

    A run is not considered successful merely because the hook-based training
    scaffold passes.  The ordinary exported adapters must independently pass
    the same held-out gates after a clean standard-loader reload.
    """

    output_dir = Path(config["output_dir"])
    pipe.unload_lora_weights()
    device = config["device"]
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    conditioning = {}
    for example in data["calibration"]:
        conditioning[(example["id"], "base")] = encode_prompt(
            pipe, example["base"], config
        )
        conditioning[(example["id"], "target")] = encode_prompt(
            pipe, example["target"], config
        )
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")

    _load_cover(pipe, Path(config["signature_cover_lora"]), "clean_s")
    _load_cover(pipe, Path(config["payload_cover_lora"]), "clean_p")
    _load_cover(pipe, output_dir / "modified_s.safetensors", "modified_s")
    _load_cover(pipe, output_dir / "modified_p.safetensors", "modified_p")
    actual_loader_scaling = {}
    for adapter_name in ("clean_s", "clean_p", "modified_s", "modified_p"):
        peft_config = pipe.unet.peft_config[adapter_name]
        actual_loader_scaling[adapter_name] = {
            "r": int(peft_config.r),
            "lora_alpha": float(peft_config.lora_alpha),
            "rank_pattern": {
                str(key): int(value)
                for key, value in sorted((peft_config.rank_pattern or {}).items())
            },
        }
    compile_record = json.loads((output_dir / "COMPILE.json").read_text(encoding="utf-8"))
    expected_signature_alpha = float(
        compile_record["loader_scaling"]["signature"]["inferred_loader_alpha"]
    )
    expected_payload_alpha = float(
        compile_record["loader_scaling"]["payload"]["inferred_loader_alpha"]
    )
    loader_scaling_passed = (
        actual_loader_scaling["modified_s"]["lora_alpha"] == expected_signature_alpha
        and actual_loader_scaling["modified_p"]["lora_alpha"] == expected_payload_alpha
    )
    states = _collect_export_states(
        pipe,
        config,
        data["calibration"],
        [int(value) for value in config["training"]["calibration_seeds"]],
        conditioning,
    )
    rows = []
    for item in states:
        latents = item["latents"].float().to(device)
        step = int(item["step"])
        base_conditioning = conditioning[(item["id"], "base")]
        clean_sp = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [1.0, 1.0, 0.0, 0.0]
        )
        modified_sp = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [0.0, 0.0, 1.0, 1.0]
        )
        clean_s = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [1.0, 0.0, 0.0, 0.0]
        )
        modified_s = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [0.0, 0.0, 1.0, 0.0]
        )
        clean_p = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [0.0, 1.0, 0.0, 0.0]
        )
        modified_p = _predict_loaded_adapters(
            pipe, config, latents, base_conditioning, step, [0.0, 0.0, 0.0, 1.0]
        )
        teacher = _predict_loaded_adapters(
            pipe,
            config,
            latents,
            conditioning[(item["id"], "target")],
            step,
            [0.0, 0.0, 0.0, 0.0],
        )
        target = teacher - clean_sp
        actual = modified_sp - clean_sp
        target_rms = max(tensor_rms(target), 1e-12)
        rows.append(
            {
                "id": item["id"],
                "seed": int(item["seed"]),
                "step": step,
                "cosine": float(
                    F.cosine_similarity(
                        target.flatten().double(),
                        actual.flatten().double(),
                        dim=0,
                        eps=1e-12,
                    )
                ),
                "nrmse": tensor_rms(actual - target) / target_rms,
                "amplitude_ratio": tensor_rms(actual) / target_rms,
                "s_relative_drift": tensor_rms(modified_s - clean_s)
                / max(tensor_rms(clean_s), 1e-12),
                "p_relative_drift": tensor_rms(modified_p - clean_p)
                / max(tensor_rms(clean_p), 1e-12),
            }
        )
    result = {
        "loader": "StableDiffusionXLPipeline.load_lora_weights",
        "custom_inference_hook": False,
        "loader_scaling_audit": {
            "passed": loader_scaling_passed,
            "expected_modified_s_alpha": expected_signature_alpha,
            "expected_modified_p_alpha": expected_payload_alpha,
            "actual": actual_loader_scaling,
        },
        "items": len(rows),
        "mean_cosine": float(np.mean([row["cosine"] for row in rows])),
        "p10_cosine": float(np.quantile([row["cosine"] for row in rows], 0.10)),
        "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])),
        "p90_nrmse": float(np.quantile([row["nrmse"] for row in rows], 0.90)),
        "mean_amplitude_ratio": float(
            np.mean([row["amplitude_ratio"] for row in rows])
        ),
        "s_mean_relative_drift": float(
            np.mean([row["s_relative_drift"] for row in rows])
        ),
        "p_mean_relative_drift": float(
            np.mean([row["p_relative_drift"] for row in rows])
        ),
        "rows": rows,
    }
    gates = _gate_results(result, config)
    gates["loader_scaling"] = loader_scaling_passed
    write_json(output_dir / "POST_EXPORT_CALIBRATION.json", result)
    pipe.unload_lora_weights()
    return result, gates


def cleanup_pipe(pipe) -> None:
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
