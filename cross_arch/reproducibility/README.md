# Controlled SD3/FLUX reproducibility capsule

This capsule is scoped to **construction reproducibility**: a reader should be
able to rebuild the cached tensors, train the two branches, and export two
standard LoRA files. It does not require the regenerated images or metrics to
match a historical table entry.

## What is fixed here

- Concrete harmless source/teacher prompts, with no symbolic prompt slots.
- SD3 and FLUX tensor shapes, rollout rules, random seeds, module placement,
  initialization, losses, optimizer settings, training order, and export.
- Runtime-faithful FLUX behavior: the historical Kohya cartoon file was passed
  to Diffusers 0.30.2 but did not register. The F32 Signature-side cover is
  therefore prompt-only; the color cover is active at weight 0.75.
- The evidenced F29 warm start for the controlled F32 reconstruction.
- A minimum acceptance contract that checks execution and file structure, not
  attack success or exact numerical identity.

## Files

- `data/controlled_pairs.json`: the exact SD3 subset and the exact FLUX museum
  prompt variants used by the controlled constructions.
- `configs/sd3_e17_04a_controlled.json`: SD3 model, cache, topology, training,
  and export parameters.
- `configs/flux_f32_runtime_faithful.json`: FLUX rollout, cover behavior,
  topology, initialization, training, branch export, and two-file fold.
- `CONSTRUCTION_PSEUDOCODE.md`: detailed, architecture-specific pseudocode.
- `PUBLIC_RELEASE_RUNBOOK.md`: the files to copy from the server, exact stage
  order, command templates, and acceptance checks.
- `RELEASE_CHECKLIST.md`: the short checklist to use before updating the
  anonymous repository or submitting the rebuttal.

## Claim that this capsule supports

> We release a controlled, architecture-specific reconstruction that fixes the
> per-stage tensors, targets, prompts, initialization, module topology,
> optimization order, losses, and static two-file export for SD3 and FLUX. The
> FLUX reconstruction follows the observed Diffusers 0.30.2 runtime, including
> the disclosed non-registration of the attempted cartoon cover. Reproduction
> means completing the prescribed pipeline and producing two standard-loader
> LoRA files; exact agreement with a historical image or headline metric is not
> required.

## Claims that this capsule does not make

- It does not identify a controlled run with a particular manuscript result
  row.
- It does not claim strict BF16 equality between a rank-concatenated LoRA and
  two separately evaluated LoRAs.
- It does not claim that the historical cartoon checkpoint was active.
- It does not validate the separate Gemini-based ASR metric.

Those limitations do not prevent a third party from rebuilding the controlled
construction described here.
