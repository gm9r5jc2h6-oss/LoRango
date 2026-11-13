# Changelog

## 1.4.1 — 2026-11-14

- Hardened safetensors verification with strict dtype, shape, byte-range and
  complete-data-layout checks.
- Recomputed loader-scaling evidence instead of trusting a recorded pass flag,
  and added exact carrier tensor topology checks.
- Prevented anonymous-bundle reassembly and cleanup from following symbolic
  links or accepting malformed chunk manifests.
- Made both release builders package only Git-tracked files and normalize file
  permissions for cross-host deterministic archives.
- Added dependency-lock consistency checks for the SBOM, OSV scan,
  `requirements.txt`, and `pyproject.toml`.
- Replaced invalid citation placeholders with the public repository URL; DOI
  metadata remains an explicit external archival step.

## Repository verification follow-up — 2026-11-14

- Restored the tracked reference training log required by both integrity
  manifests so clean clones and CI verify the same file set as the release
  workspace.
- Made checksum generators use Git-tracked files when repository metadata is
  available, and made both archive builders run the full dependency-free
  submission preflight before writing output.
- Refreshed the OSV scan and classified the newly reported Pillow TIFF/EXIF
  advisories against the declared offline workflow.
- Updated GitHub Actions to the current Node.js 24-based action releases and
  replaced commit metadata with the repository account's GitHub noreply email.

## Cross-architecture supplement — 2026-11-13

- Merged the controlled SD3/FLUX construction capsule, deterministic
  architecture-exploration control plane, recovered evidence records, and
  static-LoRA merge/audit utilities into the SDXL artifact tree.
- Added a unified scope boundary and verification entry point without
  relabeling the construction specification as a completed SD3/FLUX GPU run or
  a reproduction of a manuscript headline metric.

## 1.4.0 — 2026-11-07

- Added immutable normalized input snapshots and an aggregate run-evidence
  digest, with strict cross-checks against protocol and run metadata.
- Bound the reference configuration to the complete SDXL weight/metadata
  manifest and fail closed before model execution on any mismatch.
- Added a tiered reviewer entry point, claims-to-evidence map, machine-readable
  submission status, reproducibility guide and pre-submission checklist.
- Hardened release creation against symlinks and in-tree output paths; added a
  dependency-backed CPU CI job and run-evidence tamper regression test.
- Preserved historical v1.2 evidence without relabeling it as a v1.4 result.

## 1.3.0 — 2026-11-07

- Replaced every pickle-based scaffold/checkpoint with a strict JSON +
  safetensors format and removed all runtime `torch.load`/`torch.save` calls.
- Rebuilt `verify_run.py` to validate exact files and hashes, parse
  safetensors, recompute summaries/gates, compare cross-file records, and seal
  the whole run with `EVIDENCE.json`.
- Restricted supported covers to canonical UNet-only, non-DoRA LoRAs so
  training calibration and ordinary inference use the same components.
- Enforced exact five-layer topology, prompt/split independence, unique seeds,
  finite numeric bounds, and the full public configuration contract.
- Expanded base-model verification to all four weights and 14
  result-affecting scheduler/tokenizer/component metadata files.
- Changed the OSV scan from eight direct requirements to all 44 locked
  components with alias-aware accounting; updated certifi and urllib3 network
  preparation pins and refreshed reachability documentation.
- Packaged default assets inside the wheel and made smoke PASS conditional on
  the actual pipeline status and complete post-export gates.

## 1.2.0 — 2026-09-26

- Made the optimization scaffold use the exact five final carrier blocks and
  exact compiled channel groups; saved checkpoints are topology-bound and the
  compiler rejects mismatches.
- Added mandatory clean Diffusers reload and held-out post-export calibration.
- Added hook/export metric-equivalence and actual PEFT loader-alpha gates.
- Replaced every unsafe external PyTorch load with `weights_only=True`.
- Added the missing `peft` dependency and a complete CUDA reference lock.
- Removed the fixed cover rank/alpha assumption. The compiler now profiles
  merged ranks, compensates actual loader scaling, and rejects ambiguous or
  explicit-alpha cover formats.
- Expanded provenance from configuration hashes to every loaded model weight
  and metadata file.
- Added a self-contained reference bundle with cover adapters, prompt split,
  historical successful export, checkpoint, raw records, and visual overview.
- Added a CycloneDX runtime SBOM, dated OSV result, 29-GHSA reachability review,
  offline-only execution requirement, deterministic release builder, and
  archival metadata/procedure.
- Added exact implementation details and stronger automated release/run checks.

## 1.1.0 — 2026-09-25

- Initial artifact candidate. Retained in the reference directory only as
  historical evidence; it must not be used to claim implementation closure.
