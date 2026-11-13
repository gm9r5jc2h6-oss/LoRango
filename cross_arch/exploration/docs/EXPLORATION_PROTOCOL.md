# Controlled exploration protocol

This document is normative for a prospective new-architecture exploration.
The historical SD3/FLUX scripts motivated the hypotheses, but candidate
selection in a new run must follow the fixed policy rather than an undocumented
manual search.

## Stage 00: loader and export contract

Inputs:

- exact model revision;
- exact scheduler configuration;
- runtime versions;
- architecture backend;
- one harmless prompt and seed.

Actions:

1. Load the frozen model in a fresh process.
2. Insert a zero or numerically tiny low-rank update into one representative
   module for each proposed role.
3. Save it using the proposed static format.
4. Unload the development mechanism.
5. Load the file through the public loader in a second fresh process.
6. Record adapter registration, active names, hook count, tensor shapes, and
   whether the perturbation changes a fixed prediction.

Output: `loader_contract.json`. Unsupported roles are removed before scanning.

## Stage 01: module inventory

The inventory records every two-dimensional weighted module:

```json
{
  "name": "transformer_blocks.2.attn.to_q",
  "class": "torch.nn.modules.linear.Linear",
  "in_features": 3072,
  "out_features": 3072,
  "weight_shape": [3072, 3072],
  "exportable": true
}
```

For a PyTorch model, the generic CLI can call a loader supplied as
`package.module:function`:

```bash
python3 -m xarch inventory \
  --loader my_backend.loader:load_transformer \
  --loader-config configs/model.json \
  --output artifacts/new/module_inventory.json
```

The loader returns the transformer/module to inventory, not a pipeline output.

## Stage 02: candidate generation

`candidate_templates` are declared before the probe. Each role is an exact
module path template. Enumeration may reject a candidate only because:

- a declared module is absent;
- the module failed the loader/export contract;
- a declared block-distance constraint is violated.

It must not reject a candidate based on an unrecorded visual judgment.

Output: `candidates.json`, including rejected candidates and reasons.

## Stage 03: activation capture

Use only the public training/calibration pairs and fixed seeds. For every
candidate, capture enough statistics to instantiate its low-rank probe:

- input/output shape and dtype;
- activation RMS and maximum absolute value;
- per-timestep stability;
- stream/token partition;
- native contribution for gated rows, if applicable;
- input and output factor widths;
- any normalization/rotary/softmax operation between writer and reader.

Store large tensors separately and put their paths and SHA-256 values in
`activation_manifest.json`. Do not select on validation data.

## Stage 04: short four-condition probes

For each `(candidate, rank, initialization_seed)` allowed by the policy:

1. Keep backbone, text encoders, VAE, scheduler, and benign covers frozen.
2. Train for exactly `updates_per_candidate_rank` updates.
3. Evaluate Base, S, P, and SP on the calibration split.
4. Define

   \[
   d_S=u_S-u_{S,0},\quad d_P=u_P-u_{P,0},\quad
   d_{SP}=u_{SP}-u_{SP,0},
   \]

   \[
   I=d_{SP}-d_S-d_P,
   \qquad E=\max(\operatorname{mean}(r_T^2),10^{-7}).
   \]

5. Emit one JSONL row with:

```text
candidate_id, rank, seed,
finite, exportable, loader_supported,
interaction_cosine, sp_cosine,
s_drift_ratio=mean(dS^2)/E,
p_drift_ratio=mean(dP^2)/E,
sp_amplitude=sqrt(mean(dSP^2)/E)
```

If a run crashes or becomes non-finite, emit a row with `finite=false`; do not
silently remove it.

The generic optimizer and loss loop is implemented in `xarch/probe.py`; the
backend supplies the model-specific probe attachment and predictions. The
public policy fixes AdamW, its hyperparameters, clipping, and the four loss
weights.

## Stage 05: deterministic ranking

The `controlled_v1` score is

\[
\begin{aligned}
s={}&w_I\cos(I,r_T)+w_{SP}\cos(d_{SP},r_T)\\
&-w_S\|d_S\|_E^2-w_P\|d_P\|_E^2\\
&-w_A|\log(a_{SP})|-w_R r/r_{\max}.
\end{aligned}
\]

Threshold failures are recorded separately from the scalar score. Sorting is:

1. passing rows before failing rows;
2. higher score;
3. lower rank;
4. lexicographic candidate ID.

The policy JSON, not this prose, is authoritative for numerical values.

## Stage 06: selection and negative results

The greedy selection observes `maximum_selected_candidates` and
`maximum_per_family`. Duplicate module sets at different ranks do not occupy
multiple slots. If no row passes, the required result is:

```json
{"status": "no_supported_construction_found", "selected": []}
```

That is a valid controlled result. Changing thresholds after seeing the probe
metrics requires a new policy ID and must be reported as a new exploration.

## Stages 07–09: cache, full training, and export

The selected topology is frozen. The backend must record:

- exact prompt strings and encodings;
- initial latent/state seed and shape;
- scheduler timesteps and trajectory update rule;
- clean tensors for Base/S/P/SP;
- detached teacher residual, mask (including `null`), and direction target;
- initialization of every factor;
- sequential or joint training order;
- optimizer, updates, clipping, clamps, and sampling policy;
- factor-to-file assignment and any cover fold.

Exactly two deployment files are produced. Development hooks are unloaded
before the final audit.

## Stage 10: fresh-process audit

The minimum audit is deliberately about construction, not quality:

- training reaches its configured final update with finite values;
- both files exist and match their recorded hashes;
- both register through the standard loader;
- hook count after loading is zero;
- Base/S/P/SP each complete a forward pass;
- all four outputs have the declared shape and finite values.

Image identity, headline ASR, and strict BF16 equality are outside this
contract.
