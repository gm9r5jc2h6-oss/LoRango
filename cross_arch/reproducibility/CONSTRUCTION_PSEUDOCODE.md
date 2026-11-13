# Implementation-complete controlled construction pseudocode

The JSON files in this directory are normative. Functions such as
`transformer(...)`, `scheduler.step(...)`, and `load_lora(...)` have their
ordinary Diffusers 0.30.2 meanings. All tensors used as targets are detached.

## 1. SD3 E17-04A

### 1.1 Prompt encoding and common trajectory

```text
cfg_encode(prompt):
    positive, negative, pooled_positive, pooled_negative =
        pipe.encode_prompt(
            prompt=[prompt], prompt_2=[prompt], prompt_3=[prompt],
            negative_prompt=[""], negative_prompt_2=[""],
            negative_prompt_3=[""], do_classifier_free_guidance=True)
    return concat(negative, positive), concat(pooled_negative, pooled_positive)

cfg_predict(z, t, conditioning):
    z2 = concat(z, z)                         # unconditional, conditional
    y2 = transformer(z2, t, conditioning)    # [2,16,48,48]
    y_u, y_c = split_batch(y2)
    return y_u + 7.0 * (y_c - y_u)

initial_latents(seed):
    g = CUDA_Generator(seed)
    return randn([1,16,48,48], g, float16) * scheduler.init_noise_sigma

reference = cfg_encode(source_prompt("train_waterfall"))
for seed in [2026100401, 2026100402]:
    z = initial_latents(seed)
    scheduler.set_timesteps(28)
    for step_index, t in enumerate(scheduler.timesteps):
        if step_index in [1,5,14,22,26]:
            trajectory[seed, step_index] = stopgrad(z)
        if step_index == 26: break
        z = scheduler.step(cfg_predict(z, t, reference), t, z)
```

The training seed is used with the eight training pairs. The calibration seed
is used with the four calibration pairs. The trajectory always follows the
first training source prompt; only the clean/teacher prediction conditioning
changes across examples.

### 1.2 Cache tensors and targets

```text
for (split, seed) in [(train,2026100401), (calibration,2026100402)]:
    for example in controlled_pairs.sd3_e17_04a[split]:
        c_source  = cfg_encode(example.source)
        c_teacher = cfg_encode(example.teacher)
        for k in [1,5,14,22,26]:
            z = trajectory[seed,k]                       # [1,16,48,48]
            t = scheduler.timesteps[k]
            clean   = cfg_predict(z,t,c_source)          # [1,16,48,48]
            teacher = cfg_predict(z,t,c_teacher)         # [1,16,48,48]
            cache.append({
                id, seed, step_index:k, z, t,
                clean:stopgrad(clean),
                clean_s:stopgrad(clean),                 # no S cover
                clean_p:stopgrad(clean),                 # no P cover
                target:stopgrad(teacher-clean)
            })
```

There is no spatial mask. For a target residual `r_T`, the direction used by
the cosine term is `vec(r_T)/(norm(vec(r_T))+1e-8)`.

### 1.3 Lane payload initialization and constant reader

For block `b`, `J_b` is the fixed eight-lane list in the SD3 config. Let
`h_b` denote the input of `ff.net.2` and `x_b` the input of `norm1.linear`.

```text
for block b in [9,11,12]:
    g = CPU_Generator(2026100500 + b)
    R_b = row_normalize(randn([8,6144], g)) * 1e-3

    # Capture x_b from all train and calibration cache predictions. CFG gives
    # two rows per prediction, hence 12 examples * 5 states * 2 = 120 rows.
    X_b = stack(all captured norm1.linear inputs).double()   # [120,1536]
    one = ones([120,1], float64)
    G = X_b @ transpose(X_b)
    scale = mean(diag(G))
    c = solve(G + 1e-3 * scale * eye(120), one)
    q_b = flatten(transpose(X_b) @ c).float()                # [1536]
```

Training uses the ideal constant gate and therefore updates `R_b`; `q_b` is
fitted before training and is used by the static compiler.

### 1.4 Training order and loss

```text
seed_python_and_torch(2026100501)
optimizer = AdamW([R_9,R_11,R_12], lr=3e-4,
                  betas=(0.9,0.99), weight_decay=1e-5)

for update in 1..400:
    item = uniform_choice(training_cache)                  # 8*1*5 = 40
    E = max(mean(item.target**2), 1e-7)

    # S enables an ideal +4 gate on J_b and cancels the native ff.net.2 rows.
    # P writes R_b @ h_b into the same rows.
    y_sp = predict_with_scaffold(item, S=1, P=1)
    y_p  = predict_with_scaffold(item, S=0, P=1)
    r = y_sp - item.clean

    L = mean((r-item.target)**2)/E
    L += 0.5 * (1-cosine(flatten(r),flatten(item.target),eps=1e-8))
    L += 4.0 * mean((y_p-item.clean_p)**2)/E

    zero_grad(); backward(L); clip_grad_norm(2.0); optimizer.step()
```

No masked NMSE, spatial-gradient loss, channel-balanced loss, or Signature-only
loss is active in this controlled run.

### 1.5 Static two-file export

Let `d=1536`, `m=6144`, `E_b` be the `[d,8]` row selector, and
`N_b = W_ff.net.2[J_b,:]` be the frozen native rows.

```text
for b in [9,11,12]:
    # Signature gate: effective rank one, serialized at rank eight.
    A_gate = zeros([8,1536]); A_gate[0,:] = q_b
    B_gate = zeros([9216,8])
    for j in J_b: B_gate[5*1536+j,0] = 4.0

    # Signature cancellation: B_cancel @ A_cancel = -E_b @ N_b.
    A_cancel = N_b                                  # [8,6144]
    B_cancel = -E_b                                 # [1536,8]

    # Payload rows: B_payload @ A_payload = E_b @ R_b.
    A_payload = R_b                                 # [8,6144]
    B_payload = 0.8 * E_b                           # [1536,8]

    signature_file += {norm1.linear:(A_gate,B_gate),
                       ff.net.2:(A_cancel,B_cancel)}
    payload_file   += {ff.net.2:(A_payload,B_payload)}

save_diffusers_lora(signature_file,"signature.safetensors",alpha_over_rank=1)
save_diffusers_lora(payload_file,"payload.safetensors",alpha_over_rank=1)
```

## 2. FLUX F32 runtime-faithful reconstruction

### 2.1 Concrete prompt conditioning

Encode the five strings `B`, `S`, `P`, `SP`, and `M` from
`controlled_pairs.json` with both FLUX text encoders, one image per prompt, and
`max_sequence_length=256`. Store `prompt_embeds`, pooled projections, and text
IDs for each key.

### 2.2 Four-state teacher rollout cache

```text
cartoon = attempt_load(animation-000004.safetensors, name="cartoon")
assert "cartoon" not in pipe.get_list_adapters()["transformer"]
color = load_lora(HST_Color_Schnell.safetensors, name="color")
assert "color" in pipe.get_list_adapters()["transformer"]

g = CUDA_Generator(2026100691)
z, image_ids = pipe.prepare_latents(batch=1, channels=16,
                                    height=512, width=512,
                                    dtype=bfloat16, generator=g)
scheduler.set_timesteps(4)                         # [1000,750,500,250]

for t in scheduler.timesteps:
    # "cartoon" is absent, so S is prompt-only. Color is active at 0.75.
    s_clean  = flux_predict(z,t,prompt=S,covers=[])
    p_clean  = flux_predict(z,t,prompt=P,covers=[("color",0.75)])
    sp_clean = flux_predict(z,t,prompt=SP,covers=[("color",0.75)])
    teacher  = flux_predict(z,t,prompt=M,covers=[])
    target   = stopgrad(teacher-sp_clean)

    cache.append({z,image_ids,t,s_clean,p_clean,sp_clean,teacher,target})
    z = scheduler.step(teacher,t,z)                 # teacher trajectory
```

Every prediction and target has shape `[1,1024,64]`; `image_ids` has shape
`[1,1024,3]`. No spatial mask is constructed.

### 2.3 Gate/action factors and initialization

For gate blocks `b in {2,5,8}`, use a shared trainable output factor `U_b` for
the Signature context-key branch and Payload image-query branch:

```text
DeltaW_S_K[b] = U_b @ A_S_K[b]       # add_k_proj
DeltaW_S_V[b] = V_b @ A_S_V[b]       # add_v_proj
DeltaW_P_Q[b] = U_b @ A_P_Q[b]       # to_q; same U_b as K
```

For action blocks `b in {11,14,17}`:

```text
DeltaW_P_O[b] = O_b @ A_P_O[b]       # to_out.0 only
```

All ranks are eight. Before the warm start, use CPU seed `2026102601`; obtain
`U_b` by reduced QR and scale it by 4, obtain `V_b` and `O_b` by reduced QR and
scale them by 2, and row-normalize every A factor to 0.10. Then load compatible
factors by ordered block role from the F29 training state with SHA-256
`da1170c...f702c`. This warm start is the fixed policy for this controlled
reconstruction, even though the original F32 command line did not preserve the
argument.

### 2.4 Joint reader/action training

Signature and Payload factors are optimized jointly; there is no separate
reader phase and Signature is not frozen during an action phase.

```text
seed_python_and_torch(2026102601)
optimizer = AdamW(all_S_and_P_factors, lr=4e-4,
                  betas=(0.9,0.99), weight_decay=1e-4)

for update in 1..160:
    item = cache[random.randrange(4)]

    y_S  = predict(item,prompt=S, covers=[],             S=1,P=0)
    y_P  = predict(item,prompt=P, covers=[(color,0.75)], S=0,P=1)
    y_SP = predict(item,prompt=SP,covers=[(color,0.75)], S=1,P=1)

    dS  = y_S  - item.s_clean
    dP  = y_P  - item.p_clean
    dSP = y_SP - item.sp_clean
    interaction = dSP-dS-dP
    E = max(mean(item.target**2),1e-7)

    L = mean((dSP-item.target)**2)/E
    L += 0.5 * mean((interaction-item.target)**2)/E
    L += 0.25 * (1-cosine(flatten(dSP),flatten(item.target),eps=1e-6))
    L += 12.0 * (mean(dS**2)+mean(dP**2))/E

    zero_grad(); backward(L); clip_grad_norm(2.0); optimizer.step()
    clamp_each_up_column_norm(max=12.0)
    clamp_each_down_row_norm(max=3.0)
```

### 2.5 Branch export and runtime-faithful two-file fold

```text
signature_branch = {}
payload_branch = {}
for b in [2,5,8]:
    signature_branch += add_k_proj(A_S_K[b],U_b)
    signature_branch += add_v_proj(A_S_V[b],V_b)
    payload_branch   += to_q(A_P_Q[b],U_b)
for b in [11,14,17]:
    payload_branch   += to_out_0(A_P_O[b],O_b)

save(signature_branch,"signature_branch.safetensors")
save(payload_branch,"payload_branch.safetensors")

# The historical cartoon file contributed no registered weights.
final_S = signature_branch
final_P = exact_rank_concat(0.75*color_cover,payload_branch)
save(final_S,"signature.safetensors")
save(final_P,"payload.safetensors")
```

For overlapping modules, exact rank concatenation uses
`A*=[A1;A2]`, `B*=[s1 B1,s2 B2]`, after carrying each source `alpha/r` into
its B block. Disjoint module mappings are united. No SVD is used.

## 3. Minimum acceptance contract

Reproduction succeeds when a clean process verifies all of the following:

1. training reaches the configured final update without NaN/Inf;
2. exactly two deployment files are written;
3. both files register through the standard Diffusers 0.30.2 loader;
4. no development forward hooks remain after loading;
5. Base, S, P, and SP each complete one forward pass with the documented
   tensor shape and finite values.

No success-rate threshold, image identity, or strict BF16 separate-versus-
merged equality is part of this construction-reproduction contract.
