# v1.2 full reference run

This directory contains the completed 3,000-step Pair-2 reference run produced
by the corrected exact-topology implementation.

Frozen protocol size:

- 8 training prompts × 2 seeds × 6 denoising states = 96 training states;
- 2 held-out calibration prompts × 1 seed × 6 states = 12 calibration states.

Acceptance summary:

| Metric | Hook stage | Clean Diffusers reload |
|---|---:|---:|
| Mean target-residual cosine | 0.392342 | 0.392328 |
| Mean normalized RMSE | 0.984794 | 0.984936 |
| Signature-only relative drift | 0.003495 | 0.003502 |
| Payload-only relative drift | 0.003393 | 0.003393 |

The predicted and actual PEFT loader alpha are both 4 for each merged adapter.
The maximum absolute hook/export metric delta over the 12 corresponding rows is
0.003458, below the frozen 0.010 gate. `MANIFEST.json` is the authoritative
machine-readable result; all six post-export gates must be true.
