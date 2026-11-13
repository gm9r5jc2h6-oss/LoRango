# Method implemented by the toolkit

## Threat-model interface

The user supplies two benign cover LoRAs, denoted `S` and `P`. The optimized files should retain their cover effects in the single-adapter conditions while a matched `S+P` load activates a target-specific behavior.

For a clean paired prediction `f_SP` and target-prompt teacher `f_T`, training constructs the teacher residual

```text
r_T = f_T - f_SP.
```

The trainable scaffold optimizes the paired residual toward `r_T` over multiple prompts, seeds and denoising steps using normalized MSE, spatially weighted normalized MSE, and cosine-direction losses. In v1.4, the scaffold hooks exactly the five final compiler blocks and the exact four selected channels in each block.

## Native nonlinear carrier

The deployment compiler maps each learned lane into SDXL feed-forward blocks with native GEGLU activation. Its four relevant operations are:

1. a per-layer constant gate written by `S`;
2. a content reader written by `P`;
3. native value cancellation in the `S` branch;
4. native output subtraction before the learned action is added by `P`.

The exported behavior is therefore implemented by static low-rank updates and the backbone's own nonlinearity. Training hooks are removed before export.

## Objective and held-out gate

For the paired condition, the optimized residual is compared with `r_T`. Calibration also measures relative output drift in `S`-only and `P`-only conditions. Compilation is allowed only when all configured gates pass, unless the user explicitly sets `compile_on_failure=true` for debugging. Compilation is not itself evidence of success: the code removes the scaffold and all loaded adapters, loads the two final merged files through the public Diffusers API, repeats held-out calibration, and records separate post-export gates.

## Deployment artifacts

- `signature.safetensors` and `payload.safetensors` contain carrier-only updates.
- `modified_s.safetensors` and `modified_p.safetensors` merge those updates with the two user-supplied benign cover LoRAs.
- No Python hook, detector state, prompt callback, or dynamic scaling is needed at inference.
