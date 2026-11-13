# Reference implementation details

This file freezes the settings used by `configs/reference_pair2.json`.

## Model and adapters

- Backbone: Diffusers SDXL Base 1.0, fp16, full weight digest recorded in
  `reference/MODEL_MANIFEST.json`.
- Benign Signature/Payload cover LoRAs: 560 modules each, rank 4.
- Trainable/exported carrier layers: 5 SDXL GEGLU feed-forward blocks.
- Lanes per layer: 4.
- Signature projection rank: 5 (one constant-gate row plus four cancellation
  rows); Signature output rank: 4.
- Payload projection/output rank: 4.
- The merged adapter's effective PEFT alpha is inferred from its modal rank.
  Every carrier up matrix is multiplied by `module_rank / loader_alpha`.

## Exact carrier locations and group-1 channels

| Name | Diffusers UNet block | Channels |
|---|---|---|
| d0_t8 | `down_blocks.2.attentions.0.transformer_blocks.8` | 1046, 3704, 4731, 743 |
| d1_t1 | `down_blocks.2.attentions.1.transformer_blocks.1` | 426, 2469, 3206, 912 |
| u0a0_t1 | `up_blocks.0.attentions.0.transformer_blocks.1` | 5091, 4982, 5, 4235 |
| u0a2_t1 | `up_blocks.0.attentions.2.transformer_blocks.1` | 3530, 1602, 2662, 4601 |
| u0a2_t7 | `up_blocks.0.attentions.2.transformer_blocks.7` | 1378, 1403, 2891, 510 |

Training hooks and deployment compilation use this same table. The compiler
rejects a checkpoint whose saved block/channel topology differs.

## Data and denoising states

- Training prompts: 8; training seeds: 2.
- Held-out calibration prompts: 2; calibration seeds: 1.
- Denoising steps recorded from a 30-step scheduler: 0, 3, 7, 12, 18, 20.
- Training states: 8 × 2 × 6 = 96.
- Calibration states: 2 × 1 × 6 = 12.
- Resolution: 512 × 512; classifier-free guidance: 7.5.

No validation or sealed-test prompt is accepted by the training-data schema.

## Optimization

- Updates: 3,000.
- Optimizer: AdamW, betas (0.9, 0.99), weight decay 1e-4.
- Action learning rate: 8e-4.
- Reader learning rate: 2e-4.
- Gradient norm clipping: 2.0.
- State sampling weight: 4 for denoising steps ≤6, 2 for steps 7–13, and 1
  thereafter.
- Fixed seed: 2026087714.
- Python, NumPy, Torch, and all CUDA devices are seeded.
- cuDNN benchmarking, TF32, Flash SDP, and memory-efficient SDP are disabled;
  math SDP and strict deterministic algorithms are enabled.

For target residual `r` and compiled paired residual `a`, the update objective
is:

```text
L = NMSE(a, r) + 0.5 * masked_NMSE(a, r) + 0.5 * (1 - cosine(a, r))
```

The spatial mask is a 3×3 average pool of the channel-mean absolute target
residual, normalized by its mean and clipped to [0.25, 4.0].

## Carrier scales

- Constant gate: 35.0.
- Content reader: 0.06.
- Learned action: 0.45.
- Native value cancellation: 0.90.

## Acceptance gates

Both the hook-stage calibration and the clean standard-loader export
calibration must satisfy:

- mean target-residual cosine ≥ 0.35;
- mean normalized RMSE ≤ 1.10;
- Signature-only relative drift ≤ 0.020;
- Payload-only relative drift ≤ 0.020.

In addition, the maximum absolute difference between the five reported
hook-stage and exported metrics over corresponding held-out rows must be ≤
0.010, and the actual PEFT loader alpha must match the compiler's predicted
alpha for both merged adapters.
