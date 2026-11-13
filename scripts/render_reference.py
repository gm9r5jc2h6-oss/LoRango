#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorango_sdxl.config import load_config, load_data  # noqa: E402
from lorango_sdxl.runtime import load_pipe, write_json  # noqa: E402


def load_adapter(pipe, path: Path, name: str) -> None:
    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the fixed eight-column reference comparison.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-id", default="cal_neon_cabin")
    parser.add_argument("--seed", type=int, default=2026090101)
    args = parser.parse_args()
    config = load_config(args.config, allow_completed_output=True)
    data = load_data(Path(config["data_json"]))
    examples = {row["id"]: row for split in data.values() for row in split}
    if args.example_id not in examples:
        raise KeyError(f"unknown example id: {args.example_id}")
    example = examples[args.example_id]
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    pipe = load_pipe(config)
    load_adapter(pipe, Path(config["signature_cover_lora"]), "clean_s")
    load_adapter(pipe, Path(config["payload_cover_lora"]), "clean_p")
    load_adapter(pipe, run_dir / "modified_s.safetensors", "modified_s")
    load_adapter(pipe, run_dir / "modified_p.safetensors", "modified_p")
    names = ["clean_s", "clean_p", "modified_s", "modified_p"]
    conditions = [
        ("B", example["base"], [0.0, 0.0, 0.0, 0.0]),
        ("P_clean", example["base"], [0.0, 1.0, 0.0, 0.0]),
        ("P_mod", example["base"], [0.0, 0.0, 0.0, 1.0]),
        ("S_clean", example["base"], [1.0, 0.0, 0.0, 0.0]),
        ("S_mod", example["base"], [0.0, 0.0, 1.0, 0.0]),
        ("SP_clean", example["base"], [1.0, 1.0, 0.0, 0.0]),
        ("SP_mod", example["base"], [0.0, 0.0, 1.0, 1.0]),
        ("Teacher", example["target"], [0.0, 0.0, 0.0, 0.0]),
    ]
    rendered = []
    for label, prompt, weights in conditions:
        pipe.set_adapters(names, adapter_weights=weights)
        generator = torch.Generator(device=config["device"]).manual_seed(args.seed)
        image = pipe(
            prompt,
            generator=generator,
            height=int(config["runtime"]["height"]),
            width=int(config["runtime"]["width"]),
            num_inference_steps=int(config["runtime"]["inference_steps"]),
            guidance_scale=float(config["runtime"]["guidance_scale"]),
        ).images[0]
        path = output_dir / f"{args.example_id}__seed_{args.seed}__{label}.png"
        image.save(path)
        rendered.append((label, path, image.copy()))
    label_height = 30
    width = sum(image.width for _, _, image in rendered)
    height = max(image.height for _, _, image in rendered) + label_height
    overview = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(overview)
    x = 0
    for label, _, image in rendered:
        draw.text((x + 6, 7), label, fill="black")
        overview.paste(image, (x, label_height))
        x += image.width
    overview_path = output_dir / "overview.png"
    overview.save(overview_path)
    write_json(
        output_dir / "RENDER.json",
        {
            "example": example,
            "seed": args.seed,
            "conditions": [
                {"label": label, "path": path.name} for label, path, _ in rendered
            ],
            "overview": overview_path.name,
            "ordinary_diffusers_adapters": True,
            "custom_inference_hook": False,
        },
    )
    print(json.dumps({"status": "PASS", "overview": str(overview_path)}, indent=2))


if __name__ == "__main__":
    main()
