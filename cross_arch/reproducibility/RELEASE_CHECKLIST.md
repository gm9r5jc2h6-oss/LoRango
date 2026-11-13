# Release checklist

## Required for the construction claim

- [ ] Copy and sanitize the SD3/FLUX server scripts listed in the runbook.
- [ ] Commit the two configs and the two directly consumable prompt files.
- [ ] Include or publish the FLUX conditioning cache, four-state teacher cache,
      and F29 warm start with the recorded hashes.
- [ ] Include the color LoRA when its license permits, or provide a stable
      download instruction and verify SHA-256 before execution.
- [ ] Remove the cartoon path from the controlled runner or retain it only as
      an asserted non-registering historical input. Do not describe it as an
      active Signature cover.
- [ ] Make every scientific option explicit in the generated `PROTOCOL.json`,
      including optimizer betas, weight decay, seeds, blocks, ranks, prompts,
      cache hashes, cover registration, and warm-start hash.
- [ ] Add the two stage-order command blocks from the runbook to the repository
      README.
- [ ] Add a standard-loader Base/S/P/SP smoke audit and run it in CI.
- [ ] Verify that the deployment directory contains exactly two LoRA files and
      no runtime hook or auxiliary detector.

## Wording checks

- [ ] Say “controlled construction reproduction,” not “exact reproduction of
      the headline result.”
- [ ] For SD3, say “static low-rank realization,” not “exact compilation.”
- [ ] For FLUX, disclose that the attempted cartoon cover did not register.
- [ ] State that rank-concatenation is exact in factor space but strict BF16
      runtime identity is not required by this reproduction contract.
- [ ] State that no masked NMSE is used in the SD3 or F32 controlled runs.
- [ ] Keep the separate ASR-judge limitation outside the construction claim.

## Not required for this goal

- [ ] No rerun with a converted cartoon adapter is required.
- [ ] No strict match to historical images, losses, or ASR is required.
- [ ] No manuscript-row-to-checkpoint mapping is required unless a particular
      reported number is claimed to come from this controlled run.
- [ ] No passing sealed SD3 audit or strict BF16 merge-equivalence threshold is
      required.
