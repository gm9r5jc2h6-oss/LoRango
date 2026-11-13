# Artifact Evaluation Guide

## Supported claim

This artifact supports the SDXL training claim:

> A target-specific matched pair can be trained and exported as two ordinary
> static Diffusers LoRAs whose joint behavior uses SDXL's native GEGLU
> nonlinearity, while either modified adapter remains close to its benign cover
> when loaded alone.

Version 1.4 retains the implementation closure introduced in v1.2: training hooks
exactly the same five blocks and channel groups that are exported. A run is not
successful until the exported files are unloaded, reloaded by the public
Diffusers LoRA loader, and pass held-out calibration again.

## Badge evidence

| Criterion | Evidence |
|---|---|
| Functional candidate | Tiered reviewer check; one-step GPU train/export/reload path; strict `scripts/verify_run.py`; local v1.4 full run passed and is summarized by a hash-bound sanitized record |
| Reusable candidate | Supplied cover LoRAs and prompt split; strict schemas; model identity binding; claims/evidence map; complete method/output docs |
| Results Reproduced | The local v1.4 run passed, but award this badge only after an independent evaluator runs section F and the strict verifier returns `PASS` |
| Available | Release/Zenodo metadata and deterministic archive builder are prepared; DOI remains an author-side external action |

Do not claim **Available** while `CITATION.cff` still contains
`IDENTIFIER_PENDING`.

The sanitized local-run record is
`evidence/V1_4_GPU_EVIDENCE_SUMMARY.json`. It records hashes of the original
sealed evidence and reviewer-check files without publishing machine-local
paths. It is supporting author evidence, not a substitute for an independent
reviewer run.

## Hardware and expected work

- Structural and integrity checks: CPU, under one minute after dependencies are installed.
- GPU smoke: one CUDA GPU, approximately 24 GB free VRAM.
- Full reference training: one NVIDIA GPU; 24 GB minimum, 48 GB recommended.
- SDXL Base 1.0 is not redistributed. Its exact 6.94 GB loaded-weight identity
  is in `reference/MODEL_MANIFEST.json`.

## A. Verify package integrity and structure

```bash
python scripts/submission_check.py \
  --json-output /tmp/lorango-static-review.json
```

Expected JSON status:

```text
"status": "PASS"
```

This checks source syntax, frozen assets, reference hashes, the complete
reference bundle, strict-verifier negative/tamper tests, and that no
pickle-based checkpoint or runtime `torch.load`/`torch.save` path remains.
After installing dependencies, add `--runtime` to exercise all carrier groups
and the dependency-backed unit suite.

## B. Build the exact environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.lock
pip install --no-deps -e .
```

The runtime dependency closure is in `security/sbom.cdx.json`; the dated OSV
result and code-path reachability assessment are under `security/`.

For an isolated container run, use a read-only artifact/model mount, a separate
writable output mount, and `--network=none`.

## C. Verify the licensed base model

After accepting the upstream model terms, a networked preparation host may
download only the required files and verify them automatically:

```bash
python scripts/fetch_reference_model.py --output-dir /path/to/stable-diffusion-xl-base-1.0
```

For an existing local snapshot, run only the verifier:

```bash
python scripts/verify_model.py \
  --model-dir /path/to/stable-diffusion-xl-base-1.0
```

Expected status: `PASS`, four weight files, 14 result-affecting metadata files,
weights digest `41ca764bf62ee2e32bea63fea751dfeeeae37e45000c1a39da698135e129c157`,
and metadata digest
`8cd362fea4b83aa726b21357504e9e53959d974a73f72e94c7dda8a40c176e42`.

## D. Validate all reference inputs without loading SDXL

```bash
export LORANGO_BASE_MODEL=/path/to/stable-diffusion-xl-base-1.0
python train_pair.py --config configs/reference_pair2.json --validate-only
```

Expected status: `VALID`, five carrier layers, no cover collision, Signature
modal loader alpha 4, and Payload modal loader alpha 4.

## E. One-step GPU functionality test

```bash
python scripts/run_smoke.py \
  --config configs/reference_pair2.json \
  --output-dir /tmp/lorango_sdxl_smoke
```

The smoke test is not a quality claim. It must nevertheless exercise model and
cover loading, exact-topology hooks, backward optimization, safetensors checkpoint
save/load, static compilation, a clean four-adapter Diffusers reload, and
post-export calibration. Verify it with:

```bash
python scripts/verify_run.py --run-dir /tmp/lorango_sdxl_smoke --allow-smoke
```

## F. Full reference reproduction

```bash
bash scripts/train.sh configs/reference_pair2.json runs/reference_pair2.log
python scripts/verify_run.py --run-dir runs/reference_pair2_v1_4_0
```

The definitive success conditions are:

1. `MANIFEST.json.status == "PASS"`;
2. all in-memory calibration gates are true;
3. all standard-loader post-export gates are true;
4. `hook_export_equivalence` is true;
5. the manifest and `COMPILE.json` hashes match the exact four compiled files;
6. raw rows reproduce all recorded summaries and gate decisions;
7. `INPUT_CONFIG.json` and `INPUT_DATA.json` exactly match the frozen protocol;
8. `RUN_METADATA.json` binds the complete model identity to the declared model manifest; and
9. `EVIDENCE.json` seals the exact run file set with per-file and aggregate digests.

`compile_on_failure` may be enabled for diagnosis, but its outputs are never a
reproduced successful result.

## G. Inspect reference evidence

`reference/README.md` distinguishes historical v1.1 evidence, the v1.2
closure baseline, and the v1.4 review path. The sanitized local v1.4 summary is
under `evidence/`. Reviewers can inspect the benign cover adapters, exact prompt
split, raw JSON records, visual overview, successful exported LoRAs, and their
SHA-256 manifest without access to private author directories.

For the shorter reproduced-results path, recompile the bundled topology-bound
checkpoint, rerun standard-loader calibration, and require byte-identical
deployable outputs:

```bash
python scripts/reproduce_reference.py \
  --model-dir "$LORANGO_BASE_MODEL" \
  --output-dir /tmp/lorango_reproduced
```

## H. Create the immutable release

```bash
python scripts/build_release.py \
  --output ../LoRango_SDXL_Reproducibility_Bundle_v1.4.0.tar.gz
```

Upload those exact bytes to a long-term public archive and place the assigned
identifier in `CITATION.cff` before submission. See
`docs/ARCHIVAL_RELEASE.md`.
