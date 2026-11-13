# LoRango SDXL Pair-Training Toolkit

Artifact release v1.4.0. Reviewers should begin with
[`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md).
The fastest machine-readable review entry point is
`python scripts/submission_check.py`.
The code release uses the Apache-2.0 license; model and adapter inputs retain
their respective upstream licenses.

This package contains the **successful SDXL training and compilation path**
used by the paper plus one disclosed Pair-2 reproduction bundle. It does not
include the paper-wide evaluation suites, failed experiments, or SD3/FLUX code.

Given:

1. an SDXL base checkpoint;
2. a benign Signature cover LoRA;
3. a benign Payload cover LoRA; and
4. paired `base_prompt -> target_prompt` training examples,

the toolkit trains a new target-specific head and compiles it into two ordinary static LoRA files:

```text
modified_s.safetensors
modified_p.safetensors
```

At inference, each file preserves its benign cover behavior when loaded alone. The learned target behavior is activated by loading the matched pair together. Training uses forward hooks only as an optimization scaffold; the exported files contain no hooks, detector, scheduler callback, or external runtime component.

## 1. Environment

The reference environment is Python 3.10+, CUDA, and an NVIDIA GPU with at least 24 GB VRAM (48 GB is recommended for the default 512x512 protocol).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.lock
pip install --no-deps -e .
```

The reference server used:

```text
torch==2.6.0
diffusers==0.30.2
transformers==4.44.2
safetensors==0.4.5
numpy==1.26.4
accelerate==0.34.2
huggingface-hub==0.25.2
peft==0.12.0
```

`requirements.lock`, the CycloneDX SBOM, `environment.yml`, and `Dockerfile`
describe the reference environment. Before using a GPU, run the
release check:

```bash
bash scripts/artifact_check.sh
```

## 2. Prepare data

Copy `examples/train_data.example.json`. Each split is a list of objects:

```json
{
  "train": [
    {
      "id": "train_001",
      "base_prompt": "a service robot in a museum lobby",
      "target_prompt": "a doberman dog in a museum lobby, same viewpoint and background"
    }
  ],
  "calibration": []
}
```

Recommended minimum: 8 diverse training prompts and 2 disjoint calibration prompts. The default protocol uses two train seeds and one calibration seed.

## 3. Configure

Copy `configs/train_pair.example.json` and edit these required paths:

```json
{
  "base_model": "/path/to/stable-diffusion-xl-base-1.0",
  "signature_cover_lora": "/path/to/benign_signature.safetensors",
  "payload_cover_lora": "/path/to/benign_payload.safetensors",
  "data_json": "/path/to/train_data.json",
  "output_dir": "/path/to/output/my_pair"
}
```

The two cover LoRAs must use paired Diffusers-style UNet keys, must not use
DoRA/explicit alpha tensors, and must not already modify the carrier modules
listed in `assets/carrier_catalog.json`. The command validates this before
training. Text-encoder LoRAs are deliberately outside the v1.4 support boundary.
For a reference/archival run, set `model_manifest` to a frozen manifest; v1.4
then aborts before model loading unless the complete local model identity matches.

## 4. Train and export

Run the full pipeline with one command:

```bash
python train_pair.py --config configs/my_pair.json
```

The logging wrapper preserves stdout/stderr for artifact review:

```bash
bash scripts/train.sh configs/my_pair.json runs/my_pair.log
```

To check paths, data, adapter keys, carrier availability, and configuration without loading SDXL:

```bash
python train_pair.py --config configs/my_pair.json --validate-only
```

To compile an existing trained checkpoint without retraining:

```bash
python train_pair.py \
  --config configs/my_pair.json \
  --compile-only /path/to/checkpoint_trained.safetensors
```

## 5. Outputs

The output directory contains:

```text
PROTOCOL.json                 frozen configuration and data split
INPUT_CONFIG.json             normalized execution-effective config snapshot
INPUT_DATA.json               normalized execution-effective data snapshot
INPUT_MODEL_MANIFEST.json     normalized expected model identity snapshot
RUN_METADATA.json             environment, seeds and exact input hashes
training_trace.json           optimization trace
checkpoint_trained.safetensors non-deployable tensor state
checkpoint_trained.json       strict topology/provenance sidecar
CALIBRATION.json              held-out latent-space calibration
POST_EXPORT_CALIBRATION.json  calibration after standard-loader reload
signature.safetensors         carrier-only Signature component
payload.safetensors           carrier-only Payload component
modified_s.safetensors        deployable Signature + benign cover
modified_p.safetensors        deployable Payload + benign cover
COMPILE.json                  selected layers, channels and output hashes
MANIFEST.json                 hashes, layers, scales, and final status
EVIDENCE.json                 closed per-file and aggregate hash/size manifest
```

Only `modified_s.safetensors` and `modified_p.safetensors` are needed for ordinary inference.

## 6. Success criterion

By default, compilation proceeds only if the hook-stage calibration satisfies
all thresholds below. The exported files are then unloaded, reloaded through
`StableDiffusionXLPipeline.load_lora_weights`, and required to satisfy the same
thresholds again:

- mean target-residual cosine >= 0.35;
- mean normalized RMSE <= 1.10;
- Signature-only relative drift <= 0.020; and
- Payload-only relative drift <= 0.020.

These are held-out calibration checks, not paper test-set results. A run is a
success only when `MANIFEST.json` has `status: "PASS"` and every value in both
`gates` and `post_export_gates` is true. Keep validation and sealed-test prompts
outside `train_data.json`.

## 7. Important reproducibility notes

- The bundled `assets/scaffold_init.safetensors` plus JSON sidecar is a small,
  non-deployable optimization initialization. v1.4 selects only the same five
  blocks and exact channels that the compiler exports; unused historical rows
  are never hooked. The runtime contains no pickle-based checkpoint loading.
- `assets/carrier_catalog.json` is the frozen SDXL carrier catalogue used by the successful experiments.
- Set `training.use_warm_start_actions` to `false` to zero the initialized action matrices while retaining the learned reader scaffold.
- The exported adapters are ordinary static LoRAs and require no custom inference code.
- Diffusers derives a PEFT `lora_alpha` from each merged adapter's rank
  distribution. The compiler profiles every user cover and compensates each
  carrier matrix by `module_rank / inferred_loader_alpha`; ambiguous modal
  ranks and explicit alpha tensors are rejected.
- `reference/` contains the cover adapters, historical output, raw records,
  reference prompts, contact sheet, and the complete SDXL weight manifest.
- For a different SDXL implementation whose module names or dimensions differ from Diffusers SDXL Base 1.0, rebuild the carrier catalogue before use.
- `configs/train_pair.schema.json` defines the public configuration contract; unknown configuration keys are rejected.
- Train/calibration IDs, normalized base prompts, target prompts, prompt pairs,
  and train/calibration seed sets are required to be disjoint as applicable.
- `scripts/verify_run.py` parses every evidence file, recomputes all summaries
  and gates from raw rows, verifies input snapshots and model binding, and
  rejects missing, extra, or inconsistent evidence.
- Training requests deterministic CUDA execution with fixed Python, NumPy and Torch seeds. Remaining platform-dependent behavior is disclosed in `RUN_METADATA.json`.

## 8. Repository map

```text
ARTIFACT_EVALUATION.md     reviewer checklist and expected results
docs/                      method, data, outputs and troubleshooting
configs/                   example config and JSON schema
examples/                  example prompt-pair dataset
assets/                    frozen non-deployable scaffold/catalogue
lorango_sdxl/              training and compilation implementation
scripts/                   structural check, logging wrapper and GPU smoke test
tests/                     CPU-side artifact unit tests
VERIFICATION.json          recorded reference-server verification
```

The exact rank, optimizer, learning-rate, loss, layer, seed, and gate settings
are consolidated in `docs/IMPLEMENTATION_DETAILS.md`.

## 9. Minimal inference example

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "/path/to/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

pipe.load_lora_weights("/path/to/output", weight_name="modified_s.safetensors", adapter_name="S")
pipe.load_lora_weights("/path/to/output", weight_name="modified_p.safetensors", adapter_name="P")
pipe.set_adapters(["S", "P"], [1.0, 1.0])
image = pipe("your base prompt", num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sp_output.png")
```
