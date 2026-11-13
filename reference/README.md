# Reference reproduction bundle

This directory makes the artifact runnable without privately supplied cover
LoRAs. It contains the two benign cover adapters, the exact Pair-2 prompt
split, a historical successful export, its raw JSON records, and its visual
overview. The SDXL Base 1.0 weights are not redistributed; their complete
weight-file identity is frozen in `MODEL_MANIFEST.json`.

The files under `outputs/`, the migrated `checkpoint/checkpoint_legacy.*`, and
the legacy JSON logs are **historical v1.1 evidence**. They are retained so the paper's
reported bytes and visual result can be audited. They must not be mistaken for
a v1.2 implementation-closure result: v1.1 trained a broader hook scaffold and
compiled a smaller topology.

Version 1.2 fixed that closure defect. Its completed baseline remains under
`v1_2_run/`. The original pickle checkpoints are not distributed in v1.4;
their JSON sidecars record the original hashes and restricted migration into
safetensors. This conversion does not relabel the historical run as a v1.4 run.

A valid v1.4 run trains exactly the five
compiled blocks/channels and writes `POST_EXPORT_CALIBRATION.json` after
unloading all hooks and reloading `modified_s.safetensors` and
`modified_p.safetensors` through Diffusers. Only a v1.4 manifest whose status is
`PASS`, whose recomputed gates all pass, and whose `EVIDENCE.json` file set is
closed is a current reproduced result.

The historical eight-condition visual overview is available at
`images/overview.png`. The newly rendered v1.2 comparison and its generation
record are under `v1_2_images/`; they can be regenerated with
`scripts/render_reference.py`.

The completed v1.2 closure run is under `v1_2_run/`. Its summary is separate from
the historical v1.1 files so the two evidence generations cannot be confused.

To reproduce:

1. Obtain the licensed `stabilityai/stable-diffusion-xl-base-1.0` snapshot.
2. Verify it with `python3 scripts/verify_model.py --model-dir /path/to/model`.
3. Set `LORANGO_BASE_MODEL=/path/to/stable-diffusion-xl-base-1.0`; edit only
   `output_dir` and optionally `device` in `configs/reference_pair2.json`.
4. Run `bash scripts/train.sh configs/reference_pair2.json runs/reference.log`.
5. Run `python3 scripts/verify_run.py --run-dir runs/reference_pair2_v1_4_1`.

`REFERENCE_MANIFEST.json` authenticates every bundled reference file. It is
regenerated only when the release bundle changes.
