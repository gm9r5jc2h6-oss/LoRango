# Cross-architecture supplement

This directory merges the controlled SD3/FLUX construction material into the
SDXL release without changing the existing SDXL implementation or reference
result. It has four layers:

| Path | Purpose | Present status |
|---|---|---|
| `exploration/` | Dependency-free inventory, candidate generation, short-probe policy, deterministic selection, stage planning, and two-file audit contract. | Executable and unit-tested with the synthetic backend. |
| `reproducibility/` | Fixed harmless pairs, SD3/FLUX configurations, per-stage tensors and targets, training order, losses, export formulas, pseudocode, and runbook. | Construction specification complete; model-specific GPU entry points are not bundled. |
| `evidence/` | Recovered topology, loader, merge, equivalence, and result-binding records. | Auditable records; manuscript headline rows remain unbound where stated. |
| `tools/` | Diffusers-LoRA merge and factor-level merge audit utilities. | Source included; real files and dependencies are required to run them. |

## Quick verification

Verify the generic exploration pipeline without model weights or a GPU:

```bash
cd cross_arch/exploration
python3 -m xarch demo --output-dir /tmp/lorango-xarch-demo
python3 -m unittest discover -s tests -v
```

Verify the fixed construction capsule:

```bash
cd cross_arch/reproducibility
shasum -a 256 -c MANIFEST.sha256
```

Validate the machine-readable files and utility syntax from the repository
root:

```bash
find cross_arch -name '*.json' -type f -print0 | xargs -0 -n1 python3 -m json.tool >/dev/null
PYTHONPYCACHEPREFIX=/tmp/lorango-pycache \
  python3 -m py_compile cross_arch/tools/*.py
```

## Rebuilding the controlled constructions

Start with `reproducibility/CONSTRUCTION_PSEUDOCODE.md` for the complete stage
semantics and `reproducibility/PUBLIC_RELEASE_RUNBOOK.md` for the architecture-
specific commands. The normative parameters are:

- `reproducibility/configs/sd3_e17_04a_controlled.json`;
- `reproducibility/configs/flux_f32_runtime_faithful.json`; and
- the harmless examples under `reproducibility/data/`.

The runbook also lists the executed server sources and three FLUX tensor
artifacts required to turn the specification into a directly executable GPU
reference path. Those files are identified by SHA-256 but are not present in
this archive. Do not substitute guessed values: either import and sanitize the
listed sources/artifacts, or implement a conforming backend and report it as a
controlled reconstruction.

## Supported claim

This merged release lets a reader reconstruct the stated per-stage tensors,
targets, masks/directions, reader/action sequencing, module placement, and
static two-file export for the controlled SD3 and FLUX constructions. It also
provides an executable, deterministic architecture-porting control plane.

It does not establish that a rebuilt image, ASR, or other headline metric will
match a manuscript table row. It also does not bind E17-04A or F32 to an
unnamed result row; consult `evidence/evidence_status_and_result_binding.json`.

