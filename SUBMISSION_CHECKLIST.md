# Artifact submission checklist

## Package (complete in this candidate)

- [x] One obvious reviewer entry point and explicit evaluation tiers.
- [x] Pinned environment, container recipe, SBOM and dated vulnerability record.
- [x] Frozen example/reference inputs with complete SHA-256 manifests.
- [x] Self-contained run input snapshots and whole-run tamper seal.
- [x] Reference model identity enforced before GPU model loading.
- [x] No pickle checkpoints, runtime `torch.load`, symlinks or network requirement.
- [x] Deterministic archive builder and archival metadata.
- [x] Claims-to-evidence map and explicit support boundaries.

## Author actions before requesting badges

- [ ] Attach the accepted paper and artifact appendix through the venue's required channel; keep anonymity-sensitive files out of the public archive when policy requires it.
- [x] Run the full v1.4 reference workflow locally and preserve the original sealed directory and reviewer-check result; publish a hash-bound sanitized summary.
- [ ] Have an independent evaluator run the v1.4 workflow before requesting Results Reproduced.
- [ ] Compare the paper's tables/figures against the new v1.4 evidence and record tolerances.
- [ ] Upload the exact archive to Zenodo or another qualifying permanent repository.
- [ ] Replace `IDENTIFIER_PENDING` in `CITATION.cff` and complete permitted author metadata.
- [ ] Confirm the target venue's current anonymity, licensing, hardware-access and appendix rules.

Do not mark the last section complete from historical v1.2 evidence.
