# Claims and evidence map

This document limits evaluation to claims that the artifact can directly test.
It is not a substitute for the accompanying paper.

| Claim | Decisive evidence | Acceptance rule |
|---|---|---|
| The optimization and export topology is the same frozen five-layer topology. | `checkpoint_trained.json`, `COMPILE.json`, `scripts/verify_run.py` | Exact layer names, blocks and four channels per layer must match. |
| Export produces ordinary static LoRAs without a custom inference hook or detector. | Four compiled safetensors, `COMPILE.json`, `MANIFEST.json` | Exact four-file contract; both external-dependency flags are false. |
| Held-out calibration passes before and after standard-loader reload. | `CALIBRATION.json`, `POST_EXPORT_CALIBRATION.json`, raw rows | The verifier recomputes every summary, gate and row identity. |
| The run used the declared inputs and reference SDXL snapshot. | `INPUT_CONFIG.json`, `INPUT_DATA.json`, `PROTOCOL.json`, `RUN_METADATA.json` | Input snapshots must match the protocol; model weights and metadata must match `reference/MODEL_MANIFEST.json`. |
| The result bundle has not been selectively altered. | `EVIDENCE.json` | Exact top-level file set, per-file hashes/sizes and aggregate digest must match. |

The bundled v1.2 numeric run is historical evidence. It must not be represented
as an independently reproduced v1.4 result. Current claim status is machine
readable in `SUBMISSION_STATUS.json`.
