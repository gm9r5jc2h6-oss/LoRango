from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline
from torch import nn


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    current: Any = root
    for token in dotted.split("."):
        current = current[int(token)] if token.isdigit() else getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{dotted!r} does not resolve to a torch module")
    return current


def load_pipe(config: dict[str, Any]):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=bool(config.get("local_files_only", True)),
    )
    pipe = pipe.to(config["device"])
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    return pipe


def scheduler_for(pipe, config: dict[str, Any]):
    scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    scheduler.set_timesteps(
        int(config["runtime"]["inference_steps"]), device=config["device"]
    )
    return scheduler


@torch.no_grad()
def encode_prompt(pipe, prompt: str, config: dict[str, Any]):
    device = config["device"]
    positive, negative, positive_pooled, negative_pooled = pipe.encode_prompt(
        prompt=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=[""],
    )
    projection = int(pipe.text_encoder_2.config.projection_dim)
    runtime = config["runtime"]
    time_ids = pipe._get_add_time_ids(
        (runtime["height"], runtime["width"]),
        (0, 0),
        (runtime["height"], runtime["width"]),
        dtype=positive.dtype,
        text_encoder_projection_dim=projection,
    ).to(device)
    return torch.cat([negative, positive]), {
        "text_embeds": torch.cat([negative_pooled, positive_pooled]),
        "time_ids": time_ids.expand(2, -1),
    }


def initial_latents(pipe, scheduler, config: dict[str, Any], seed: int):
    device = config["device"]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    runtime = config["runtime"]
    shape = (
        1,
        pipe.unet.config.in_channels,
        int(runtime["height"]) // pipe.vae_scale_factor,
        int(runtime["width"]) // pipe.vae_scale_factor,
    )
    return (
        torch.randn(shape, generator=generator, device=device, dtype=torch.float16)
        * scheduler.init_noise_sigma
    )


def guided_prediction(
    pipe,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    conditioning,
    config: dict[str, Any],
):
    embeds, added = conditioning
    model_input = scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
    prediction = pipe.unet(
        model_input,
        timestep,
        encoder_hidden_states=embeds,
        added_cond_kwargs=added,
        return_dict=True,
    ).sample
    unconditional, conditional = prediction.chunk(2)
    guidance = float(config["runtime"]["guidance_scale"])
    return unconditional + guidance * (conditional - unconditional)


def autocast_context(config: dict[str, Any]):
    if str(config["device"]).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.double().square().mean().sqrt())

