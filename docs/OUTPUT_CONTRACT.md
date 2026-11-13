# Output contract

## Deployable files

Only these files are required by the inference user:

```text
modified_s.safetensors
modified_p.safetensors
```

They use Diffusers-style LoRA keys and load with the standard `load_lora_weights` API.

## Non-deployable audit files

| File | Purpose |
|---|---|
| `PROTOCOL.json` | Frozen inputs, split and hyperparameters |
| `INPUT_CONFIG.json` | Normalized execution-effective configuration snapshot |
| `INPUT_DATA.json` | Normalized train/calibration data snapshot |
| `INPUT_MODEL_MANIFEST.json` | Declared model identity snapshot, or explicit `NOT_DECLARED` marker |
| `RUN_METADATA.json` | Software/hardware versions and input hashes |
| `training_trace.json` | Per-update optimization trace |
| `checkpoint_trained.safetensors` | Training tensor state used by the compiler |
| `checkpoint_trained.json` | Strict checkpoint topology and tensor hash sidecar |
| `CALIBRATION.json` | Held-out latent metrics and rows |
| `POST_EXPORT_CALIBRATION.json` | Same metrics after clean Diffusers reload of the final files |
| `COMPILE.json` | Carrier layers, channels, scales and file hashes |
| `MANIFEST.json` | Final status and concise audit summary |
| `EVIDENCE.json` | Exact top-level file set, hashes, sizes, and aggregate digest |

The checkpoint pair is not an inference adapter and must not be presented as one.

`MANIFEST.json` may claim `PASS` only when both the in-memory optimization
scaffold and the cleanly reloaded exported adapters pass every configured gate.
