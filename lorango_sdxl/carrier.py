from __future__ import annotations

import gc
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .checkpoint import load_checkpoint
from .runtime import resolve_module, sha256, write_json


BLOCK_SPECS = [
    ("d0_t0", "down_blocks.2.attentions.0.transformer_blocks.0"),
    ("d0_t3", "down_blocks.2.attentions.0.transformer_blocks.3"),
    ("d0_t6", "down_blocks.2.attentions.0.transformer_blocks.6"),
    ("d0_t8", "down_blocks.2.attentions.0.transformer_blocks.8"),
    ("d1_t1", "down_blocks.2.attentions.1.transformer_blocks.1"),
    ("d1_t7", "down_blocks.2.attentions.1.transformer_blocks.7"),
    ("mid_t1", "mid_block.attentions.0.transformer_blocks.1"),
    ("mid_t7", "mid_block.attentions.0.transformer_blocks.7"),
    ("u0a0_t1", "up_blocks.0.attentions.0.transformer_blocks.1"),
    ("u0a0_t7", "up_blocks.0.attentions.0.transformer_blocks.7"),
    ("u0a2_t1", "up_blocks.0.attentions.2.transformer_blocks.1"),
    ("u0a2_t7", "up_blocks.0.attentions.2.transformer_blocks.7"),
]

ELIGIBLE_NAMES = {
    "d0_t8",
    "d1_t1",
    "u0a0_t1",
    "u0a2_t1",
    "u0a2_t7",
}

class TrainableGegluLayer:
    """Exact differentiable simulation of one finally compiled GEGLU carrier.

    Training and export must use the same blocks, channels, cancellation terms,
    and scales.  This class therefore mirrors ``compile_pair`` rather than the
    older 12-layer signal/reference search scaffold.
    """

    def __init__(
        self,
        name: str,
        block,
        projection,
        output,
        state: dict[str, Any],
        selection: dict[str, Any],
        carrier_config: dict[str, Any],
        device: str,
        use_warm_start_actions: bool,
    ) -> None:
        self.name = name
        self.block = str(selection["block"])
        self.signature_enabled = False
        self.payload_enabled = False
        self.channels = [int(value) for value in selection["channels"]]
        self.action_normalizers = torch.tensor(
            selection["action_normalizers"], dtype=torch.float32, device=device
        ).view(1, -1)
        self.reader_scale = float(carrier_config["reader_scale"])
        self.action_scale = float(carrier_config["action_scale"])
        self.gate_scale = float(carrier_config["gate_scale"])
        self.value_cancel_scale = float(carrier_config["value_cancel_scale"])
        gamma = block.norm3.weight.detach().float().to(device)
        beta = block.norm3.bias.detach().float().to(device)
        self.constant_reader = ((1.0 / gamma) / (beta / gamma).sum()).view(1, -1)
        rank = len(self.channels)
        self.value_readers = torch.nn.Parameter(
            state["value_readers"].float()[:rank].to(device)
        )
        initial_actions = state["actions"].float()[:, :rank].to(device)
        if not use_warm_start_actions:
            initial_actions = torch.zeros_like(initial_actions)
        self.actions = torch.nn.Parameter(initial_actions)
        self.native_value_rows = projection.weight.detach().float().to(device)[
            self.channels, :
        ]
        self.native_columns = output.weight.detach().float().to(device)[:, self.channels]
        self.projection_handle = projection.register_forward_hook(self._projection_hook)
        self.output_handle = output.register_forward_hook(self._output_hook)

    def _projection_hook(self, _module, args, output):
        if not self.signature_enabled and not self.payload_enabled:
            return output
        x = args[0].float()
        half = output.shape[-1] // 2
        result = output.float().clone()
        if self.payload_enabled:
            values = F.linear(x, self.value_readers * self.reader_scale)
            for index, channel in enumerate(self.channels):
                result[..., channel : channel + 1] += values[..., index : index + 1]
        if self.signature_enabled:
            constant = F.linear(x, self.constant_reader)
            cancellation = F.linear(x, self.native_value_rows)
            for index, channel in enumerate(self.channels):
                result[..., half + channel : half + channel + 1] += (
                    4.0 * self.gate_scale * constant
                )
                result[..., channel : channel + 1] -= (
                    self.value_cancel_scale
                    * cancellation[..., index : index + 1]
                )
        return result.to(output.dtype)

    def _output_hook(self, _module, args, output):
        if not self.signature_enabled and not self.payload_enabled:
            return output
        values = args[0].float()[..., self.channels]
        delta = torch.zeros_like(output, dtype=torch.float32)
        if self.signature_enabled:
            delta -= F.linear(values, self.native_columns)
        if self.payload_enabled:
            action = self.actions * self.action_normalizers * self.action_scale
            delta += F.linear(values, action)
        return output + delta.to(output.dtype)

    def state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "block": self.block,
            "channels": self.channels,
            "action_normalizers": self.action_normalizers.detach().cpu().view(-1).tolist(),
            "value_readers": self.value_readers.detach().cpu(),
            "actions": self.actions.detach().cpu(),
        }

    def close(self) -> None:
        self.projection_handle.remove()
        self.output_handle.remove()


class TrainablePair:
    def __init__(self, layers: list[TrainableGegluLayer]) -> None:
        self.layers = layers

    def enable(self, signature: bool, payload: bool) -> None:
        for layer in self.layers:
            layer.signature_enabled = bool(signature)
            layer.payload_enabled = bool(payload)

    def state(self, training_step: int) -> dict[str, Any]:
        return {
            "training_step": int(training_step),
            "layers": [layer.state() for layer in self.layers],
        }

    def close(self) -> None:
        for layer in self.layers:
            layer.close()


def make_trainable_pair(pipe, checkpoint: dict[str, Any], config: dict[str, Any]) -> TrainablePair:
    by_name = {layer["name"]: layer for layer in checkpoint["layers"]}
    selections = select_carriers(config, checkpoint)
    missing = [row["name"] for row in selections if row["name"] not in by_name]
    if missing:
        raise ValueError(f"scaffold is missing layers: {missing}")
    layers = []
    for selection in selections:
        name = str(selection["name"])
        block_name = str(selection["block"])
        block = resolve_module(pipe.unet, block_name)
        projection = resolve_module(pipe.unet, f"{block_name}.ff.net.0.proj")
        output = resolve_module(pipe.unet, f"{block_name}.ff.net.2")
        layers.append(
            TrainableGegluLayer(
                name,
                block,
                projection,
                output,
                by_name[name],
                selection,
                config["carrier"],
                config["device"],
                bool(config["training"]["use_warm_start_actions"]),
            )
        )
    return TrainablePair(layers)


def load_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if "corrected_candidates_by_block" not in catalog:
        raise ValueError(f"invalid carrier catalog: {path}")
    return catalog


def select_carriers(config: dict[str, Any], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = load_catalog(Path(config["carrier_catalog"]))
    source_by_name = {layer["name"]: layer for layer in checkpoint["layers"]}
    group = int(config["carrier"]["group_index"])
    selections = []
    for name, block in BLOCK_SPECS:
        if name not in ELIGIBLE_NAMES:
            continue
        if name not in source_by_name:
            raise ValueError(f"trained checkpoint is missing {name}")
        qualified = catalog["corrected_candidates_by_block"][block]["qualified"]
        selected = qualified[group * 4 : group * 4 + 4]
        if len(selected) != 4:
            raise ValueError(
                f"carrier group {group} is unavailable for {name}: only {len(qualified)} candidates"
            )
        channels = [int(row["channel"]) for row in selected]
        interactions = torch.tensor(
            [max(float(row["interaction_median"]), 1e-6) for row in selected],
            dtype=torch.float32,
        )
        source_layer = source_by_name[name]
        saved_channels = [int(value) for value in source_layer.get("channels", [])]
        if "action_normalizers" in source_layer and saved_channels == channels:
            normalizers = [float(value) for value in source_layer["action_normalizers"]]
        else:
            amplitudes = source_layer["amplitudes"].float()[:4]
            normalizers = ((0.5 * amplitudes) / interactions).tolist()
        selections.append(
            {
                "name": name,
                "block": block,
                "channels": channels,
                "action_normalizers": normalizers,
                "candidate_rows": selected,
            }
        )
    if len(selections) != 5:
        raise ValueError(f"the frozen carrier topology requires exactly five layers, got {len(selections)}")
    return selections


def expected_carrier_keys(selections: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for selected in selections:
        prefix = f"unet.{selected['block']}"
        for module in ("ff.net.0.proj", "ff.net.2"):
            keys.add(f"{prefix}.{module}.lora.down.weight")
            keys.add(f"{prefix}.{module}.lora.up.weight")
    return keys


def validate_cover_collisions(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    expected = expected_carrier_keys(select_carriers(config, checkpoint))
    for label, key in (
        ("Signature", "signature_cover_lora"),
        ("Payload", "payload_cover_lora"),
    ):
        cover = load_file(config[key])
        collision = expected.intersection(cover)
        if collision:
            sample = sorted(collision)[:4]
            raise ValueError(
                f"{label} cover LoRA overlaps carrier modules; choose a cover without these keys: {sample}"
            )


def cover_rank_profile(
    path: Path, additional_ranks: list[int] | None = None
) -> dict[str, Any]:
    """Predict the global alpha chosen by Diffusers/PEFT for a merged adapter.

    Diffusers-style checkpoints do not carry a global alpha. Diffusers 0.30+
    therefore chooses the modal module rank as ``lora_alpha`` and records
    per-module rank overrides. The compiler compensates against that inferred
    alpha instead of assuming that every user cover is rank/alpha four.
    """

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        unsupported = [
            key
            for key in keys
            if key.endswith(".alpha")
            or "lora_magnitude_vector" in key
            or not key.startswith("unet.")
        ]
        if unsupported:
            raise ValueError(
                f"{path} is not a canonical UNet-only, non-DoRA LoRA; "
                f"unsupported keys: {sorted(unsupported)[:4]}"
            )
        ranks = [
            int(handle.get_slice(key).get_shape()[0])
            for key in keys
            if key.endswith(".lora.down.weight")
        ]
    if not ranks:
        raise ValueError(f"cover contains no Diffusers LoRA down tensors: {path}")
    merged_ranks = ranks + list(additional_ranks or [])
    counts = Counter(merged_ranks)
    top_count = max(counts.values())
    modes = sorted(rank for rank, count in counts.items() if count == top_count)
    if len(modes) != 1:
        raise ValueError(
            f"merged LoRA has an ambiguous modal rank {modes}; use a cover with "
            "a unique dominant rank so loader scaling is deterministic"
        )
    return {
        "path": str(path),
        "cover_rank_counts": {str(k): v for k, v in sorted(Counter(ranks).items())},
        "merged_rank_counts": {str(k): v for k, v in sorted(counts.items())},
        "inferred_loader_alpha": int(modes[0]),
    }


def _scaled_up(value: torch.Tensor, rank: int, loader_alpha: int) -> torch.Tensor:
    return value * (float(rank) / float(loader_alpha))


def _merge(cover_path: Path, carrier_path: Path, destination: Path) -> None:
    cover = load_file(str(cover_path))
    carrier = load_file(str(carrier_path))
    collision = set(cover).intersection(carrier)
    if collision:
        raise RuntimeError(f"cover/carrier key collision: {sorted(collision)[:4]}")
    state = {key: value.clone() for key, value in cover.items()}
    state.update({key: value.clone() for key, value in carrier.items()})
    save_file(state, str(destination), metadata={"format": "pt"})


@torch.no_grad()
def compile_pair(pipe, checkpoint_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path)
    source_by_name = {layer["name"]: layer for layer in checkpoint["layers"]}
    selections = select_carriers(config, checkpoint)
    validate_cover_collisions(config, checkpoint)
    for selected in selections:
        trained = source_by_name[selected["name"]]
        if trained.get("block") != selected["block"] or [
            int(value) for value in trained.get("channels", [])
        ] != selected["channels"]:
            raise ValueError(
                "checkpoint topology does not match the final compiler selection: "
                f"{selected['name']}"
            )

    signature_carrier_ranks = []
    payload_carrier_ranks = []
    for selected in selections:
        rank = len(selected["channels"])
        signature_carrier_ranks.extend([rank + 1, rank])
        payload_carrier_ranks.extend([rank, rank])
    signature_profile = cover_rank_profile(
        Path(config["signature_cover_lora"]), signature_carrier_ranks
    )
    payload_profile = cover_rank_profile(
        Path(config["payload_cover_lora"]), payload_carrier_ranks
    )
    signature_alpha = int(signature_profile["inferred_loader_alpha"])
    payload_alpha = int(payload_profile["inferred_loader_alpha"])

    module_shapes = {}
    for selected in selections:
        block_name = selected["block"]
        block = resolve_module(pipe.unet, block_name)
        projection = resolve_module(pipe.unet, f"{block_name}.ff.net.0.proj")
        output = resolve_module(pipe.unet, f"{block_name}.ff.net.2")
        gamma = block.norm3.weight.detach().float().cpu()
        beta = block.norm3.bias.detach().float().cpu()
        channels = selected["channels"]
        module_shapes[block_name] = {
            "projection_rows": int(projection.weight.shape[0]),
            "projection_width": int(projection.weight.shape[1]),
            "output_width": int(output.weight.shape[1]),
            "constant_reader": ((1.0 / gamma) / (beta / gamma).sum()).view(-1),
            "native_value_rows": projection.weight.detach().float().cpu()[channels, :],
            "native_columns": output.weight.detach().float().cpu()[:, channels],
        }

    carrier_cfg = config["carrier"]
    signature_state: dict[str, torch.Tensor] = {}
    payload_state: dict[str, torch.Tensor] = {}
    layer_manifest = []
    for selected in selections:
        name = selected["name"]
        block_name = selected["block"]
        prefix = f"unet.{block_name}"
        channels = selected["channels"]
        rank = len(channels)
        shapes = module_shapes[block_name]
        half = shapes["projection_rows"] // 2
        source = source_by_name[name]
        readers = source["value_readers"].float()[:rank].clone()
        readers *= float(carrier_cfg["reader_scale"])
        normalizers = torch.tensor(selected["action_normalizers"], dtype=torch.float32)
        actions = source["actions"].float()[:, :rank].clone()
        # Keep the multiplication order identical to the frozen successful
        # compiler; changing it alters float32 bytes at ~1e-9 even though the
        # numerical operation is equivalent.
        actions = actions * normalizers.view(1, -1)
        actions = actions * float(carrier_cfg["action_scale"])

        signature_projection_rank = 1 + rank
        signature_projection_down = torch.zeros(
            (signature_projection_rank, shapes["projection_width"]), dtype=torch.float32
        )
        signature_projection_down[0] = shapes["constant_reader"]
        signature_projection_down[1:] = shapes["native_value_rows"]
        signature_projection_up = torch.zeros(
            (shapes["projection_rows"], signature_projection_rank), dtype=torch.float32
        )
        for index, channel in enumerate(channels):
            signature_projection_up[half + channel, 0] = (
                4.0 * float(carrier_cfg["gate_scale"])
            )
            signature_projection_up[channel, index + 1] = -float(
                carrier_cfg["value_cancel_scale"]
            )
        signature_projection_up = _scaled_up(
            signature_projection_up, signature_projection_rank, signature_alpha
        )

        signature_output_down = torch.zeros((rank, shapes["output_width"]), dtype=torch.float32)
        payload_output_down = torch.zeros((rank, shapes["output_width"]), dtype=torch.float32)
        payload_projection_up = torch.zeros((shapes["projection_rows"], rank), dtype=torch.float32)
        for index, channel in enumerate(channels):
            signature_output_down[index, channel] = 1.0
            payload_output_down[index, channel] = 1.0
            payload_projection_up[channel, index] = 1.0

        signature_state.update(
            {
                f"{prefix}.ff.net.0.proj.lora.down.weight": signature_projection_down.contiguous(),
                f"{prefix}.ff.net.0.proj.lora.up.weight": signature_projection_up.contiguous(),
                f"{prefix}.ff.net.2.lora.down.weight": signature_output_down.contiguous(),
                f"{prefix}.ff.net.2.lora.up.weight": _scaled_up(
                    -shapes["native_columns"], rank, signature_alpha
                ).contiguous(),
            }
        )
        payload_state.update(
            {
                f"{prefix}.ff.net.0.proj.lora.down.weight": readers.contiguous(),
                f"{prefix}.ff.net.0.proj.lora.up.weight": _scaled_up(
                    payload_projection_up, rank, payload_alpha
                ).contiguous(),
                f"{prefix}.ff.net.2.lora.down.weight": payload_output_down.contiguous(),
                f"{prefix}.ff.net.2.lora.up.weight": _scaled_up(
                    actions, rank, payload_alpha
                ).contiguous(),
            }
        )
        layer_manifest.append(
            {
                "name": name,
                "block": block_name,
                "channels": channels,
                "action_normalizers": selected["action_normalizers"],
            }
        )

    signature_path = output_dir / "signature.safetensors"
    payload_path = output_dir / "payload.safetensors"
    modified_s = output_dir / "modified_s.safetensors"
    modified_p = output_dir / "modified_p.safetensors"
    save_file(signature_state, str(signature_path), metadata={"format": "pt"})
    save_file(payload_state, str(payload_path), metadata={"format": "pt"})
    _merge(Path(config["signature_cover_lora"]), signature_path, modified_s)
    _merge(Path(config["payload_cover_lora"]), payload_path, modified_p)
    result = {
        "format": "ordinary static Diffusers LoRA",
        "external_hook_at_inference": False,
        "external_detector_at_inference": False,
        "checkpoint": str(checkpoint_path),
        "carrier_group": int(carrier_cfg["group_index"]),
        "scales": {
            key: float(carrier_cfg[key])
            for key in ("gate_scale", "reader_scale", "action_scale", "value_cancel_scale")
        },
        "loader_scaling": {
            "signature": signature_profile,
            "payload": payload_profile,
            "rule": "carrier up matrices are multiplied by module_rank / inferred_loader_alpha",
        },
        "layers": layer_manifest,
        "files": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (signature_path, payload_path, modified_s, modified_p)
        },
    }
    write_json(output_dir / "COMPILE.json", result)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
