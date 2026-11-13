# Evidence records

This directory contains public, path-sanitized evidence summaries.

`V1_4_GPU_EVIDENCE_SUMMARY.json` records the exact hashes, sizes, acceptance
gates, output hashes, and reported metrics from the authors' completed local
v1.4 SDXL reference run. The original sealed archive and reviewer-check JSON
are not embedded in the anonymous bundle because their command and manifest
records contain machine-local paths. Their exact SHA-256 values are retained so
an artifact chair can bind a separately supplied original to this release.

This summary supports inspection of the authors' local verification. It does
not convert that run into an independent Results Reproduced evaluation.
