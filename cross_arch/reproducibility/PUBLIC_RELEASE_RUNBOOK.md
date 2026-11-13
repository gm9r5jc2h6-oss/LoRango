# Public release runbook

## 1. Repository layout

Add the following tree to the anonymous repository. The names are deliberately
architecture-specific so that the existing SDXL reference path is unchanged.

```text
cross_arch/
  README.md
  data/
    sd3_pairs.json
    flux_mecha_pair.json
  configs/
    sd3_e17_04a_controlled.json
    flux_f32_runtime_faithful.json
  sd3/
    common.py
    gate_scan.py
    train_scaffold.py
    compile_static_pair.py
    scale_payload.py
    audit_standard_loader.py
  flux/
    runtime.py
    encode_prompts.py
    build_teacher_cache.py
    train_attention_pair.py
    merge_diffusers_lora.py
    audit_standard_loader.py
  artifacts/
    flux_conditioning_cal_museum.pt
    flux_teacher_state_cache.pt
    flux_f29_training_state.pt
  reproduce_sd3.sh
  reproduce_flux.sh
```

The three `.pt` artifacts contain embeddings, latent-space predictions, and
low-rank factors, not generated images. Their historical hashes are:

| file | SHA-256 |
|---|---|
| FLUX conditioning cache | `8e17b4db63a44354280a292f1f7225feb7be509a621fb02f9ae014b12bb7fbd5` |
| FLUX four-state teacher cache | `e42c3952ef399c900d9e9aba340320c8d233159e229b7eed8068f5dbf9413334` |
| F29 warm start | `da1170c705787d844fd6065b966086f92c4e42df062b3403daf50436f41f702c` |

Publishing these artifacts makes the F32 training path directly executable.
`build_teacher_cache.py` remains necessary so that readers can independently
rebuild and inspect their contents.

## 2. Server source files to sanitize and copy

Do not rewrite the implementation from the paper prose. Copy the executed
server files, remove only absolute paths/resource guards, and retain the
following hashes in a provenance manifest:

| public role | server basename | executed-source SHA-256 |
|---|---|---|
| SD3 common runtime | `e17_04_sd3_common.py` | `ec6708087f3548a5ed7103d1ff0342e862c7b9d2810b0427d475368363c9e291` |
| SD3 lane scan | `e17_04_sd3_stage01_gate_scan.py` | `8421f03f9b386bf2150f8ebd58471167b551034c94a8696ec26ca8628f450f46` |
| SD3 scaffold/cache/training | `e17_04_sd3_stage02_native_gate_scaffold.py` | `8b492f32af00e2b1748676416e0e5c26fbe87f53d65b3d0359b12b50be126a35` |
| SD3 static compiler | `e17_04_sd3_stage03_compile.py` | `6822b2528a676b8c7050ed0a60ffd89645b9fa29b25546fdd9d1fcb99829457d` |
| SD3 Payload scaler | `e17_04_sd3_scale_payload.py` | `7ce4382ca25bd214df215a46ab8f4bf8ebd9bf8cb7d2dd6739e1dd4c2f91cfe4` |
| SD3 standard-loader audit | `e17_04_sd3_stage04_standard_tensor_audit.py` | `e55d80fda45d6cf0a527b284190713eeb6d431806437768f05805950a4e92ff1` |
| FLUX runtime/scheduler helpers | `f03_f04_compile_native_gate_pair.py` | `1507db41306d7792662cd893f0a91d1e534724fc039d3b879cf621d409d0e6c0` |
| FLUX prompt encoder | `f09_build_functional_conditioning.py` | `0df97208ba0cede9f951cd10f565f26e9561ee0164933cb7a53a3e87fb222065` |
| FLUX cache/objective helpers | `f12_cover_aware_compile.py` | `f2c9991a864d0fa8fb9972f5de1c64c8e03c249f2249915020f1809ce68dd2e0` |
| FLUX attention pair trainer/exporter | `f26_attention_gate_compile.py` | `f7fc67398a1c1083bc8e378de24a1244f03582eceddc5514dee80d0b4a52d5a5` |

The public FLUX trainer should explicitly record `--warmstart` in its protocol.
The controlled command below fixes it to the F29 state rather than presenting
the forensic recovery as a directly recorded historical argv.

## 3. SD3 stage commands

`SD3_MODEL_DIR` is the local SD3 Medium Diffusers revision named in the config.
GPU selection and memory-resource guards may be changed without changing the
scientific construction.

```bash
python3 cross_arch/sd3/gate_scan.py \
  --model "$SD3_MODEL_DIR" \
  --data cross_arch/data/sd3_pairs.json \
  --output runs/sd3_gate_scan \
  --device cuda:0 --height 512 --width 512 --steps 28 \
  --max-prompts 4 --seeds 2026100401 \
  --coordinates 0.05,0.2,0.5,0.8,0.95 --top-k 16

python3 cross_arch/sd3/train_scaffold.py \
  --model "$SD3_MODEL_DIR" \
  --data cross_arch/data/sd3_pairs.json \
  --scan runs/sd3_gate_scan \
  --output runs/sd3_scaffold \
  --device cuda:0 --height 384 --width 384 --inference-steps 28 \
  --record-steps 1,5,14,22,26 \
  --train-seeds 2026100401 --calibration-seeds 2026100402 \
  --max-train 8 --max-calibration 4 \
  --blocks 3 --selected-block-ids 9,11,12 --lanes 8 \
  --gate-amplitude 4.0 --optimization-steps 400 \
  --learning-rate 3e-4 --off-weight 4.0 --fixed-seed 2026100501

python3 cross_arch/sd3/compile_static_pair.py \
  --model "$SD3_MODEL_DIR" \
  --checkpoint runs/sd3_scaffold/trained_scaffold.pt \
  --output runs/sd3_compiled

python3 cross_arch/sd3/scale_payload.py \
  --source runs/sd3_compiled --output runs/sd3_final \
  --payload-scale 0.8

python3 cross_arch/sd3/audit_standard_loader.py \
  --model "$SD3_MODEL_DIR" \
  --data cross_arch/data/sd3_pairs.json \
  --adapters runs/sd3_final \
  --scaffold-final runs/sd3_scaffold/FINAL.json \
  --output runs/sd3_loader_audit \
  --device cuda:0 --height 384 --width 384 \
  --inference-steps 28 --record-steps 1,5,14,22,26 \
  --seed 2026100402 --max-calibration 4
```

The fixed lane table is normative. A lane-scan tie or floating-point ranking
difference does not invalidate the reconstruction; if necessary, the public
trainer may accept the lane table directly from the JSON config.

## 4. FLUX stage commands

`FLUX_MODEL_DIR` is the pinned FLUX.1-schnell revision. The color LoRA must be
distributed when its license permits, or accompanied by a stable download
instruction and the hash in the config. The cartoon file is not required for
the runtime-faithful reconstruction because it contributed no registered
adapter; the public runner should record `cartoon_registered=false`.

```bash
python3 cross_arch/flux/encode_prompts.py \
  --model "$FLUX_MODEL_DIR" \
  --data cross_arch/data/flux_mecha_pair.json \
  --output runs/flux_conditioning.pt \
  --device cpu --ids cal_museum

python3 cross_arch/flux/build_teacher_cache.py \
  --model "$FLUX_MODEL_DIR" \
  --conditioning-cache runs/flux_conditioning.pt \
  --color-lora "$FLUX_COLOR_LORA" --color-weight 0.75 \
  --output runs/flux_teacher_state_cache.pt \
  --example-id cal_museum --height 512 --width 512 \
  --steps 4 --seed 2026100691 --trajectory teacher \
  --device cuda:0

python3 cross_arch/flux/train_attention_pair.py \
  --model "$FLUX_MODEL_DIR" \
  --conditioning-cache runs/flux_conditioning.pt \
  --state-cache runs/flux_teacher_state_cache.pt \
  --color-lora "$FLUX_COLOR_LORA" \
  --output runs/flux_f32 \
  --device cuda:0 --devices cuda:0 --example-id cal_museum \
  --blocks 2,5,8 --action-head --action-blocks 11,14,17 \
  --rank 8 --action-rank 8 --steps 160 --learning-rate 4e-4 \
  --off-weight 12.0 --color-weight 0.75 \
  --carrier-scale 4.0 --down-scale 0.10 --value-scale 2.0 \
  --action-scale 2.0 \
  --warmstart cross_arch/artifacts/flux_f29_training_state.pt \
  --fixed-seed 2026102601

python3 cross_arch/flux/merge_diffusers_lora.py \
  --left "$FLUX_COLOR_LORA" \
  --right runs/flux_f32/payload.safetensors \
  --left-scale 0.75 --right-scale 1.0 \
  --output runs/flux_final/payload.safetensors \
  --manifest runs/flux_final/MERGE_PAYLOAD.json

cp runs/flux_f32/signature.safetensors \
   runs/flux_final/signature.safetensors

python3 cross_arch/flux/audit_standard_loader.py \
  --model "$FLUX_MODEL_DIR" \
  --signature runs/flux_final/signature.safetensors \
  --payload runs/flux_final/payload.safetensors \
  --conditioning-cache runs/flux_conditioning.pt \
  --state-cache runs/flux_teacher_state_cache.pt \
  --output runs/flux_loader_audit.json --device cuda:0
```

The public `build_teacher_cache.py` is a thin entry point around the cache
algorithm in `CONSTRUCTION_PSEUDOCODE.md`; it must assert the four expected
records, tensor shapes, prompts, seed, and teacher trajectory before saving.

## 5. Minimum automated checks

The CI smoke path may use one cached state and reduced optimization steps. It
must fail when:

- a required state tensor or prompt key is absent;
- a configured module path does not exist;
- a LoRA file fails standard-loader registration;
- a development hook remains after export;
- Base/S/P/SP produces a wrong shape or non-finite value.

CI need not enforce image similarity, ASR, or exact historical losses.
