# Dependency vulnerability reachability review

Date: 2026-11-14. Scope: all 44 exact packages in `requirements.lock`.
`OSV_SCAN_2026-11-14.json` records 89 unique GHSA identifiers (169 OSV IDs
including aliases) across seven packages: `accelerate`, `diffusers`,
`filelock`, `fsspec`, `pillow`, `torch`, and `transformers`.

This is a reachability assessment, not a statement that old dependencies are
generally safe. The numerical reference stack remains frozen so reviewers can
compare results. It must run offline in the supplied container, without secrets
or unrelated writable mounts.

## Enforced controls in v1.4

- `local_files_only=true` is mandatory; the training/runtime path never fetches
  models or code.
- SDXL uses `use_safetensors=True`; custom pipelines and `trust_remote_code`
  are not enabled.
- The release contains no `.pt`, `.pth`, or `.ckpt` files and the runtime has no
  `torch.load`/`torch.save` calls. Scaffold and trained state use a strict JSON
  sidecar plus safetensors with exact key-set and SHA-256 validation.
- Cover adapters must be canonical UNet-only, non-DoRA Diffusers LoRAs.
- The base-model manifest covers four weights and all 14 result-affecting
  scheduler, tokenizer, and component configuration files.
- `verify_run.py` parses safetensors headers, recomputes summaries and gates,
  cross-checks manifests, and verifies a closed whole-run evidence manifest.
- The recommended container invocation uses `--network=none`, a read-only
  artifact mount, and one dedicated writable output mount.

## High-severity items with explicit treatment

| Advisory/path | v1.4 treatment | Residual assessment |
|---|---|---|
| PyTorch `weights_only` checkpoint bypass (`GHSA-63cw-57p8-fm3p`) | All pickle-based checkpoints and all `torch.load` calls were removed. | The vulnerable API is not called by the artifact. Do not add `.bin`/`.pt` model files or bypass `use_safetensors=True`. |
| fsspec `ReferenceFileSystem` template injection (`GHSA-27vj-qcqg-25rc`) | The code never imports or constructs `ReferenceFileSystem` and accepts no Kerchunk/reference JSON. | Not reachable in the implemented workflow; keep network disabled. |
| Pillow TIFF/EXIF memory amplification (`GHSA-wqgw-9xfh-qf2g`) | The training and compilation interfaces accept prompts, JSON, safetensors, and a frozen local model snapshot; they never open reviewer-supplied TIFF/EXIF images. `render_reference.py` only handles pipeline-generated images and writes PNG output. | Not reachable in the declared offline workflow. Do not extend the artifact into an untrusted image-upload service without upgrading Pillow and rerunning the reference baseline. |
| Pillow compressed-TIFF heap overflow (`GHSA-xcvg-w56q-r89m`, also indexed as `PYSEC-2026-4363`) | No artifact path writes TIFF or forwards attacker-controlled EXIF/TIFF metadata. Reference rendering writes PNG files generated in-process. | Not reachable in the declared workflow; keep untrusted image conversion outside the frozen environment. |
| certifi compromised CA and urllib3 HTTPS proxy policy | Lock pins were updated to `certifi==2026.10.19` and `urllib3==2.8.0`. | Network preparation still requires normal TLS hygiene; the reference run itself is offline. |

## Remaining advisory classes

The OSV report contains historical and current advisories for unused
Transformers model families/training helpers, Diffusers loading/saving paths,
Pillow decoders, file locking, and PyTorch primitives such as JIT, distributed,
quantized, sparse, packed-sequence, and specialized operators. The LoRango
reference path uses the local SDXL pipeline, CLIP tokenizers, safetensors,
ordinary CUDA tensor operations, and generated PNG output only. It does not
claim those vulnerable package features are safe.

Because OSV data and reachability can change, reviewers should rerun:

```text
python scripts/security_audit.py --output security/OSV_SCAN_CURRENT.json
```

Any newly reported advisory must be classified before release. A dependency
upgrade that changes numerical behavior requires a new reference baseline; it
must not silently replace the frozen environment while retaining old result
claims.

## Residual risk

GPU tensor workloads can exhaust memory or disk when given hostile local
configuration/model files. Run the reference artifact in an isolated account
or container, without credentials, with resource limits and verified input
hashes. The separate networked model-fetch step is not part of the offline
evaluation run.
