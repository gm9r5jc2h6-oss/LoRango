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

The bundled v1.2 numeric run is historical evidence. The authors' local v1.4
run passed the strict verifier and is represented by the sanitized, hash-bound
record `evidence/V1_4_GPU_EVIDENCE_SUMMARY.json`. Neither record may be
represented as an independently reproduced result; that claim still requires
an evaluator-run section F. Current claim status is machine readable in
`SUBMISSION_STATUS.json`.

## Cross-architecture supplement

| Claim | Decisive evidence | Acceptance rule |
|---|---|---|
| The architecture-porting control plane is deterministic and executable without model weights. | `cross_arch/exploration/`, its unit tests, and generated demo artifacts | The synthetic demo selects the declared candidate and all tests pass. |
| The controlled SD3/FLUX constructions fix the missing mechanism details. | `cross_arch/reproducibility/configs/`, `CONSTRUCTION_PSEUDOCODE.md`, and `PUBLIC_RELEASE_RUNBOOK.md` | Per-stage states/targets, direction or interaction targets, training order, topology, losses, seeds, and two-file export are concrete rather than symbolic. |
| The recovered evidence is not silently promoted to a headline-result reproduction. | `cross_arch/evidence/evidence_status_and_result_binding.json` and `CROSS_ARCH_RELEASE_STATUS.json` | Unbound rows, non-exact equivalence, missing inputs, and runtime-faithful FLUX cover behavior remain explicitly disclosed. |
| Static LoRA overlap can be folded and audited. | `cross_arch/tools/merge_diffusers_lora.py`, `cross_arch/tools/audit_diffusers_lora_merge.py`, and FLUX merge audit JSON files | Factor reconstruction must pass; BF16 runtime equivalence is reported against an explicit threshold rather than assumed. |

The archive does not directly test a real SD3/FLUX GPU reconstruction because
the model-specific backends, licensed weights, historical warm start, and
teacher caches are not bundled. It therefore supports construction
reproducibility, not a claim that a manuscript ASR or image result was
independently regenerated.
