# Reproducibility entry points

## CPU/static tier

```bash
python scripts/submission_check.py --json-output /tmp/lorango-static.json
```

This tier needs only Python 3.10+ and verifies the release, reference bundle,
negative fixtures and tamper detection.

## Dependency-backed CPU tier

After installing `requirements.lock` and the package:

```bash
python scripts/submission_check.py --runtime \
  --json-output /tmp/lorango-runtime.json
```

## Licensed model tier

```bash
python scripts/submission_check.py \
  --model-dir /path/to/stable-diffusion-xl-base-1.0 \
  --json-output /tmp/lorango-model.json
```

## Completed GPU run tier

```bash
python scripts/submission_check.py \
  --model-dir /path/to/stable-diffusion-xl-base-1.0 \
  --run-dir runs/reference_pair2_v1_4_0 \
  --json-output runs/reference_pair2_v1_4_0.REVIEWER_CHECK.json
```

The final tier is the only tier that validates a new v1.4 result. Keep the
generated JSON outside the run directory; adding it to a sealed directory
intentionally invalidates the seal.
