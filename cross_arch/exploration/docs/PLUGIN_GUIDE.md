# Architecture backend guide

The deterministic control plane cannot infer model-specific tensor semantics.
A new architecture supplies one backend implementing
`xarch.backend.ArchitectureBackend`.

## Required decisions

### Model boundary

Return the frozen denoiser/transformer and identify the exact pipeline,
revision, scheduler, dtype, and loader version. Text encoders and the VAE stay
frozen during probes.

### Conditioning

`encode_prompts` must return every tensor required by the model, including
pooled projections, text IDs, masks, or guidance embeddings. Record their
shapes and hashes in the cache manifest.

### Rollout state

`prepare_initial_state` and `scheduler_step` define the trajectory. Do not mix
states from different prompts or trajectories without recording that choice.

### Branch semantics

`predict(..., enabled_branches)` must recognize exactly:

- `frozenset()` for Base;
- `frozenset({"S"})`;
- `frozenset({"P"})`;
- `frozenset({"S", "P"})`.

Benign covers, if any, are part of the condition definition and must be
reported separately from trained branches.

### Probe attachment

`attach_probe` receives a candidate from `candidates.json`; it may attach
temporary hooks during training. The probe must contain only the factors for
the declared roles and rank. All temporary hooks must be removable.

### Export

`export_two_files` writes one Signature and one Payload file using ordinary
static loader keys. If two contributions share a module, combine their weight
deltas without silently overwriting one factor. Record rank and alpha/scaling
semantics.

## Recommended backend package layout

```text
my_backend/
├── __init__.py
├── loader.py          # load_transformer(config)
├── conditioning.py    # prompt encoding
├── rollout.py         # initial state, timesteps, scheduler step
├── probes.py          # temporary low-rank modules
├── training.py        # short and full probe loops
├── export.py          # static two-file serialization
└── audit.py           # fresh-process loader runner
```

## Backend self-test

Before any search, run one harmless fixed state and check:

1. repeated Base prediction is stable within the declared dtype tolerance;
2. S and P can be independently enabled;
3. SP enables both, with no leaked persistent state from earlier conditions;
4. hooks return to the original count after cleanup;
5. a saved zero probe loads in a fresh process.

The backend must not contain candidate-ranking or manual block-selection logic;
those decisions belong in the public policy and architecture JSON.

