# Cross-architecture controlled exploration bundle

This bundle makes the **architecture-porting procedure** reproducible. In this
merged release, the fixed SD3/FLUX construction capsule is available at
[`../reproducibility/README.md`](../reproducibility/README.md): that capsule
says how to rebuild named controlled constructions, while this directory
records how a new architecture is inventoried, probed, ranked, selected,
trained, exported, and audited.

The bundle uses harmless style-transfer prompt pairs. It contains no working
target payload and makes no claim that every architecture will produce a
usable construction.

## What is executable here

The control plane is dependency-free Python:

- inventory any supplied PyTorch-like model through `named_modules()`;
- capture activation input/output summaries with removable hooks;
- run the policy-defined short-probe optimizer/loss through a model backend;
- generate candidates from a declarative architecture description;
- join candidates with measured four-condition probe metrics;
- apply a deterministic score, thresholds, budget, and tie-breaking policy;
- emit either `selected` or `no_supported_construction_found`;
- create a hash-bound stage plan;
- check the minimum two-file/four-condition audit contract.

The package includes a synthetic end-to-end demonstration. Real tensor
forwards remain in an architecture backend because prompt encoding, latent
packing, scheduler calls, stream partitioning, and export keys differ across
model families. `xarch/backend.py` defines the required interface.

## Quick verification

From this directory:

```bash
python3 -m xarch demo --output-dir artifacts/demo_run
python3 -m unittest discover -s tests -v
```

The demo must select `joint_attention_b2_to_b5_r4`, write all intermediate
JSON files, and finish without model weights or a GPU.

## Real new-architecture workflow

1. Copy `architectures/new_architecture.template.json` and fill the module
   templates, block ranges, stream semantics, loader, and export mapping.
2. Implement one backend following `docs/PLUGIN_GUIDE.md`.
3. Generate `module_inventory.json` from the frozen model.
4. Run `generate-candidates`.
5. Run the short harmless probes described in
   `docs/EXPLORATION_PROTOCOL.md`, producing `probe_metrics.jsonl`.
6. Run `rank` and `select`; do not manually replace the selected result.
7. Use `selected_topology.json` to build the full cache and train the pair.
8. Export two static files and run a fresh-process Base/S/P/SP audit.

Once the model-specific inventory and probe runner exist, the deterministic
control-plane stages can be invoked with:

```bash
ARCH_CONFIG=architectures/new_architecture.json \
MODULE_INVENTORY=artifacts/new/module_inventory.json \
PROBE_METRICS=artifacts/new/probe_metrics.jsonl \
OUT_DIR=artifacts/new \
bash scripts/explore_new_architecture.sh
```

## Normative artifacts

- `architectures/`: SD3, FLUX, synthetic, and new-architecture declarations.
- `policies/controlled_v1.json`: candidate space, scoring, thresholds,
  compute budget, tie-breaking, and stopping rule.
- `data/harmless_pairs.json`: public train/calibration/validation examples.
- `xarch/backend.py`: model-specific plugin contract.
- `xarch/candidates.py`: deterministic candidate enumeration.
- `xarch/scoring.py`: deterministic ranking and selection.
- `xarch/audit.py`: minimum execution/file/loader audit.
- `docs/EXPLORATION_PROTOCOL.md`: per-stage inputs and outputs.
- `docs/HISTORICAL_VS_CONTROLLED.md`: claim boundary.

## Supported claim

> We release a controlled architecture-porting workflow that fixes the
> candidate space, harmless data, probe budget, four-condition measurements,
> ranking function, thresholds, tie-breaking, stopping rule, static-export
> contract, and audit outputs. The workflow may return a negative result.

It does not claim that this automated policy exactly recreates every manual
decision made during the historical SD3/FLUX exploration.

`RELEASE_STATUS.json` records the present packaging boundary. The deterministic
control plane is tested; the large historical SD3/FLUX GPU scripts still need
to be sanitized and imported as reference backends according to
`SOURCE_IMPORT_PLAN.json` before calling this a complete public model-execution
release.
