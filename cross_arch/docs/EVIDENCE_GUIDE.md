# Cross-architecture reproducibility bundle

This bundle separates three claims that should not be conflated:

1. **Construction specification:** per-stage tensors, targets, module topology,
   loss, optimization order, and branch-to-file assignment are specified.
2. **Artifact audit:** a named pair of files loads and produces recorded
   four-condition behavior under a locked runtime.
3. **Historical result reproduction:** a manuscript result row is bound to the
   exact run, checkpoint, files, and evaluation record that produced it.

The current evidence closes (1) for SD3 E17-04A and FLUX F32. It partly closes
(2), with limitations recorded below. It does not close (3), because the
manuscript rows have not been mapped to server runs.

## Files

- `reproducibility_capsule/`: release-ready controlled prompts, exact SD3/FLUX
  configs, implementation-complete pseudocode, server-source provenance,
  stage command templates, and a minimum execution-only acceptance contract.
- `revised_sd3_flux_construction.md`: submission-ready construction text and
  pseudocode, including the recovered limitations.
- `cross_arch_topology.json`: machine-readable topology and hyperparameters.
- `evidence_status_and_result_binding.json`: claim-by-claim status, exact hashes,
  and unresolved row binding.
- `result_artifact_binding_template.json`: per-result-row provenance schema,
  including the evaluator/judge fields needed for the headline metric.
- `merge_diffusers_lora.py`: deterministic rank-concatenation merge for two
  normalized Diffusers LoRAs. It rejects unsupported key formats.
- `audit_diffusers_lora_merge.py`: factor-space audit for the merge.
- `flux_f32_merge_payload_manifest.json`: manifest for the server-side F32
  color-cover plus Payload merge candidate.
- `flux_f32_merge_factor_audit.json`: 494-module algebraic audit; maximum
  absolute difference 0.0.
- `flux_f32_twofile_runtime_audit.json`: fresh-process Diffusers 0.30.2 audit.
  It intentionally records a strict-equivalence failure rather than hiding it.
- `flux_f32_cover_registration_audit.json`: fresh-process proof that the
  historical cartoon file registered no adapter while the color file did.

No model weights are included in this local bundle.

## Reproduce the deterministic merge

The inputs must already use Diffusers keys ending in `.lora_A.weight` and
`.lora_B.weight`.

```bash
python merge_diffusers_lora.py \
  --left benign_cover.safetensors \
  --right trained_branch.safetensors \
  --left-scale 0.75 \
  --right-scale 1.0 \
  --output final_payload.safetensors \
  --manifest MERGE_PAYLOAD.json

python audit_diffusers_lora_merge.py \
  --left benign_cover.safetensors \
  --right trained_branch.safetensors \
  --left-scale 0.75 \
  --right-scale 1.0 \
  --merged final_payload.safetensors \
  --atol 1e-6 \
  --report AUDIT_PAYLOAD.json
```

The factor audit proves the stored low-rank delta is the algebraic sum. It does
not prove BF16 runtime identity, because one concatenated LoRA and two separately
evaluated LoRAs group their low-rank matmuls differently. A model-level audit is
still required.

## Recovered FLUX facts

- F32's original `--warmstart` argv was not saved. F31 and F32 have identical
  first-step objective fields, and F31's saved factors continue the compatible
  F29 state (largest mapped one-step change 0.0004000664 at learning rate
  0.0004; minimum tensor cosine 0.9914265). The F29 state is the evidenced warm-start
  candidate, but this remains forensic recovery rather than a directly saved
  command.
- The Kohya-format cartoon file registered no adapter in a fresh Diffusers
  0.30.2 process. The color file registered normally. Consequently the executed
  F32 S-side baseline was prompt-conditioned, not cartoon-LoRA-conditioned.
- A runtime-faithful two-file candidate was created on the server as
  `final_S=Signature` and `final_P=0.75*color+Payload`. Its factor audit passes,
  but strict model-output equivalence fails: P NMSE 0.00511 and SP NMSE 0.00694.

## Required next runs

The following runs are required only for the stronger claim that both benign
covers were active and that the resulting pair reproduces a particular
historical result. They are **not** required for the narrower controlled-
construction reproduction contract in `reproducibility_capsule/`.

For an intended two-benign-cover FLUX construction:

1. convert or replace the cartoon adapter and assert that it appears in
   `get_list_adapters()` and changes a fixed prediction;
2. rebuild the rollout cache and retrain F32 from the declared initialization;
3. export two files and run Base/S/P/SP audits on held-out prompts and seeds;
4. bind the accepted artifact to the exact manuscript result row.

For SD3, retain the existing loader audit but do not call the export an “exact
compilation”: the proxy-equivalence maximum absolute difference is 0.2839988586,
the validation image audit passes, and the sealed audit fails.
