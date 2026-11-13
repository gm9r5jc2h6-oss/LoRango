# Reproducibility entry points

## CPU/static tier

```bash
python3 scripts/submission_check.py --json-output /tmp/lorango-static.json
```

This tier needs only Python 3.10+ and verifies the release, reference bundle,
negative fixtures and tamper detection.

## Dependency-backed CPU tier

After installing `requirements.lock` and the package:

```bash
python3 scripts/submission_check.py --runtime \
  --json-output /tmp/lorango-runtime.json
```

## Licensed model tier

```bash
python3 scripts/submission_check.py \
  --model-dir /path/to/stable-diffusion-xl-base-1.0 \
  --json-output /tmp/lorango-model.json
```

## Completed GPU run tier

```bash
python3 scripts/submission_check.py \
  --model-dir /path/to/stable-diffusion-xl-base-1.0 \
  --run-dir runs/reference_pair2_v1_4_1 \
  --json-output runs/reference_pair2_v1_4_1.REVIEWER_CHECK.json
```

The final tier is the only tier that validates a new v1.4 result. Keep the
generated JSON outside the run directory; adding it to a sealed directory
intentionally invalidates the seal.

The authors' local v1.4 full run passed this tier. A sanitized, hash-bound
summary is provided at `evidence/V1_4_GPU_EVIDENCE_SUMMARY.json`; the original
sealed evidence remains outside the anonymous bundle because its JSON records
contain machine-local paths. This does not replace an independent evaluator
run for a Results Reproduced claim.

## Cross-architecture supplement

The SD3/FLUX material is kept separate from the SDXL run tiers so that a
construction specification is not confused with a completed model-weight
reproduction.

The executable, dependency-free control-plane check is:

```bash
cd cross_arch/exploration
python3 -m xarch demo --output-dir /tmp/lorango-xarch-demo
python3 -m unittest discover -s tests -v
```

The fixed construction capsule is hash-checked with:

```bash
cd cross_arch/reproducibility
shasum -a 256 -c MANIFEST.sha256
```

To build SD3 or FLUX, follow `cross_arch/reproducibility/PUBLIC_RELEASE_RUNBOOK.md`
and the normative JSON config. A conforming execution must additionally supply
the pinned model, licensed cover inputs, model-specific backend, and the
warm-start/cache inputs identified in that runbook. The minimum acceptance
contract is successful stage execution, two standard-loader LoRA files, no
remaining development hooks, and finite correctly shaped Base/S/P/SP outputs;
matching the historical images or headline metric is outside this contract.

## Anonymous review distribution

For services with an 8 MiB per-file limit, build a deterministic split-asset
archive:

```bash
python3 scripts/build_anonymous_review.py \
  --output ../LoRango_SDXL_Reproducibility_Bundle_v1.4.1_anonymous.tar.gz
```

After downloading or cloning that review tree, restore and verify the original
assets before running the normal checks:

```bash
python3 scripts/reassemble_large_assets.py --cleanup-parts
python3 scripts/submission_check.py
```

See `docs/ANONYMOUS_REVIEW.md` for the exact contract.
