#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the licensed SDXL files used by the reference run, then verify them."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(output),
        allow_patterns=ALLOW_PATTERNS,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_model.py"),
            "--model-dir",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
