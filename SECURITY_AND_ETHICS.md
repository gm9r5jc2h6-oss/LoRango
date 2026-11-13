# Security and ethics statement

This is an adversarial ML research artifact intended to study composition-triggered behavior in LoRA ecosystems and to support defenses, auditing and reproducibility.

Use only model checkpoints, adapters and compute resources for which you have authorization. Do not distribute a trained pair as benign without disclosure. Generated target pairs should remain inside the approved experimental scope and should be accompanied by their manifest and hashes.

The reference bundle includes one disclosed neon/cyberpunk pair, its benign
cover adapters, hashes, raw records, and a contact sheet so that reviewers can
verify the complete workflow. It is labeled as an adversarial-ML reference
artifact and must not be redistributed as a benign adapter pair.

The exact numerical stack has known advisories. See
`security/OSV_SCAN_2026-11-07.json`, `security/sbom.cdx.json`, and
`security/REACHABILITY.md`. Reference reproduction is offline-only and uses
hash-verified safetensors; v1.4 contains no pickle-based model or checkpoint
loading path.
