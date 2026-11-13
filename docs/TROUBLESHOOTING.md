# Troubleshooting

## Cover/carrier collision

If preflight reports overlapping keys, the cover LoRA already modifies a selected GEGLU carrier module. Choose a non-overlapping cover or rebuild the carrier catalogue. Do not concatenate overlapping ranks without accounting for each adapter's alpha convention. v1.4 automatically compensates a uniquely determined modal loader alpha, but deliberately rejects text-encoder, DoRA, explicit-alpha, unknown-key, and tied-modal-rank covers because their current evaluation/loading semantics are outside the supported contract.

## CUDA out of memory

Use a less occupied GPU, reduce the 512x512 smoke test to 256x256, or reduce the number of cached prompt/seed cases. Do not change the paper configuration when reporting paper results.

## Calibration failure

Treat it as a failed target compilation, not a software error. First increase scene diversity, then optimization steps. Adjust carrier scales only on calibration data and keep validation/sealed prompts untouched.

## Post-export or equivalence failure

If training gates pass but `post_export_gates` fail, do not report the pair as
successful. Inspect `loader_scaling_audit` first. An alpha mismatch indicates
an unsupported loader/version or cover format. A metric-equivalence failure
indicates that the training topology and compiled adapter are no longer
numerically aligned; preserve the run directory and report it as a compiler
regression.

## Different SDXL implementation

The frozen catalogue targets Diffusers SDXL Base 1.0 module names and dimensions. A checkpoint with a modified U-Net layout requires a newly scanned carrier catalogue.
