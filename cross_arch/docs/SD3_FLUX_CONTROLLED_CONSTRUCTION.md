# SD3 and FLUX construction: corrected from the server records

> **Submission boundary.** The SD3 section below is tied to the completed
> `E17-04A` controlled artifact and its fresh-process loader audit. The FLUX
> section is tied to the completed `F32_crosslayer_attention_action_rank8_s160_v1`
> controlled artifact. It fixes the architecture, tensors, objective, optimizer,
> blocks, ranks, and file assignment. Subsequent server evidence narrows the
> omitted warm start to the F29 training state, but this is a forensic recovery,
> not a persisted original argv. A later F32 seven-condition run did use the
> standard loader, although with four file dependencies and only one scene and
> seed. A fresh-process registration audit further found that the Kohya-format
> cartoon file registered no adapter under Diffusers 0.30.2, whereas the color
> adapter registered normally. Finally, an exact factor-space merge of the
> color cover and Payload passed algebraic audit but failed a strict BF16
> runtime-equivalence threshold. Therefore F32 may be presented as a
> **controlled architecture-level construction artifact**, not as an exact,
> self-contained reproduction of a historical headline result.

## Compact rebuttal replacement

We agree that module names alone did not specify the cross-architecture
construction. We have therefore added the complete construction below and a
machine-readable topology record. To avoid conflating an architecture mapping
with a historical reproduction claim, we identify the exact controlled
artifacts from which every value is taken.

**SD3 controlled construction.** We used SD3 Medium with Diffusers 0.30.2.
For the completed `E17-04A` artifact, a training item is

\[
(z_s,t_s,c_{\rm src},c_{\rm tgt},v_{\rm clean},r_T),\qquad
r_T=\operatorname{sg}(v_T-v_{\rm clean}),
\]

where all predictions are evaluated at the same cached state and timestep. At
the artifact's actual training resolution of $384\times384$,
$z_s,v_{\rm clean},v_T,r_T\in\mathbb R^{1\times16\times48\times48}$ and
the $2\times2$ patch embed produces 576 image tokens of width 1536. The cache
uses scheduler indices $\{1,5,14,22,26\}$ from a 28-step trajectory.

In Diffusers' `JointTransformerBlock`, the image MLP branch is added as

\[
H_{b+1}=H_b+g_b[:,\mathrm{None},:]\odot
\operatorname{ff}_b(\widetilde H_b),
\]

where $g_b=\texttt{gate\_mlp}$ is the final width-1536 slice of
`transformer_blocks[b].norm1.linear`. Blocks 9, 11, and 12 each use the eight
lanes listed in Table 1. Signature has two static branches per block: a
`norm1.linear` writer that opens the selected `gate_mlp` coordinates, and an
`ff.net.2` branch that subtracts the corresponding native rows. Payload has an
`ff.net.2` branch that writes the learned replacement rows. Thus, at a fixed
block input, inclusion--exclusion isolates the multiplicative term between the
Signature gate increment and the Payload FF increment.

The constant Signature reader is fitted first by ridge regression to output one
on cached modulation inputs. The selected run then uses the ideal constant gate
during task fitting; consequently the task gradient trains the Payload reader
rows, while the Signature writer and native-row cancellation are constructed
from the fitted reader and frozen backbone weights. It runs 400 AdamW updates
with learning rate $3\times10^{-4}$, betas $(0.9,0.99)$, weight decay
$10^{-5}$, and uniform sampling over the cached items. Its exact objective is

\[
\mathcal L_{\rm SD3}=
\frac{\|r_{SP}-r_T\|_2^2}{\|r_T\|_2^2+\epsilon}
+0.5[1-\cos(r_{SP},r_T)]
+4\frac{\|v_P-v_{P,\rm clean}\|_2^2}{\|r_T\|_2^2+\epsilon}.
\]

No spatial mask or masked NMSE is used in this SD3 controlled artifact, and the
Signature-only loss weight is zero. This is intentionally stated rather than
copying the SDXL action-refinement loss into SD3. The compiled Payload B factors
are then multiplied once by 0.8; no dynamic scale remains.

At the initial compile, the Signature state mapping contains, for each selected
block, the `norm1.linear` gate writer and `ff.net.2` native-row cancellation;
the Payload mapping contains the learned `ff.net.2` rows. That stage writes
`signature.safetensors`/`payload.safetensors` and the aliases
`modified_s.safetensors`/`modified_p.safetensors`. The subsequent fixed 0.8
calibration emits the final `signature.safetensors` and
`payload.safetensors`. The standard-loader audit observes zero forward hooks
before loading, after loading, and at exit. This establishes loader acceptance,
not exact scaffold-to-static equality: the recorded proxy-equivalence maximum
absolute difference is 0.2839988586. The 8-scene-by-2-seed validation image
audit passes, while the separate 12-scene-by-2-seed sealed audit fails one of
its preregistered gates. We therefore use “static low-rank realization” rather
than “exact compilation,” and report both audit outcomes.

**FLUX controlled construction.** We used FLUX.1-schnell revision
`741f7c3ce8b383c54771c7003378a50191e9efe9` with Diffusers 0.30.2. The F32
state cache contains four teacher-rollout items at timesteps 1000, 750, 500,
and 250. Each item stores packed latents and outputs with shape
$[1,1024,64]$ and image IDs with shape $[1,1024,3]$. At every item,

\[
r_T=u_T-u_{SP,\rm clean},
\]

where `S_clean`, `P_clean`, `SP_clean`, and `teacher` are evaluated at the
same packed state. After caching a state, the trajectory advances with the
teacher prediction; training samples one of the four cached items uniformly.

In blocks 2, 5, and 8, Signature modifies context
`attn.add_k_proj` and `attn.add_v_proj`, while Payload modifies image
`attn.to_q`. In downstream blocks 11, 14, and 17, Payload additionally modifies
`attn.to_out.0`; no other F32 block has `to_out.0`. All ranks are 8. The
context/image joint attention therefore evaluates

\[
\operatorname{Attn}
(Q_I+H\Delta W_Q^P,
 K_C+C\Delta W_{K,C}^S,
 V_C+C\Delta W_{V,C}^S),
\]

and the downstream output branches decode the propagated interaction.

Within the recorded F32 stage, Signature is not frozen before Payload training:
all K/V/Q/O factors are optimized jointly for 160 updates with AdamW, learning
rate $4\times10^{-4}$, betas $(0.9,0.99)$, and weight decay $10^{-4}$. The
record does not preserve the `--warmstart` argv. It cannot have been the
script's random initialization: the step-1 mean Q/K up norm is 16.95094 versus
$\sqrt{8}\times4=11.31371$ under the declared random initializer, and the
step-1 objective fields of F31 and F32 are identical. F31's saved step-1
factors are in turn a one-update continuation of the compatible F29 state:
the largest mapped tensor update is 0.0004000664 (the declared learning rate is
0.0004), and the minimum tensor cosine is 0.9914265.
Accordingly, the evidenced reconstruction uses
`F29_attention_gate_action_rank8_b6_s140_v1/training_state.pt` (SHA-256
`da1170c705787d844fd6065b966086f92c4e42df062b3403daf50436f41f702c`)
as a **forensically recovered** warm start; it must not be presented as a
directly recorded historical command. For

\[
d_S=u_S-u_{S,\rm clean},\quad
d_P=u_P-u_{P,\rm clean},\quad
d_{SP}=u_{SP}-u_{SP,\rm clean},\quad
I=d_{SP}-d_S-d_P,
\]

and $E=\operatorname{mean}(r_T^2)+\epsilon$, the exact loss is

\[
\mathcal L_{\rm F32}=
\frac{\operatorname{mean}((d_{SP}-r_T)^2)}{E}
+0.5\frac{\operatorname{mean}((I-r_T)^2)}{E}
+0.25[1-\cos(d_{SP},r_T)]
+12\frac{\operatorname{mean}(d_S^2)+\operatorname{mean}(d_P^2)}{E}.
\]

There is no spatial mask in F32. The target direction is simply the normalized
teacher residual implicit in the cosine term. At export, Signature K/V factors
go only to `signature.safetensors`; Payload Q and downstream O factors go only
to `payload.safetensors`. Export is direct serialization of the trained low-rank
factors and performs no further optimization.

The intended Signature cover was a Kohya-format cartoon LoRA, but a fresh
Diffusers 0.30.2 process registered no transformer or text-encoder adapter for
that file. The color cover did register. Thus the executed F32 S-side clean
condition was prompt-conditioned rather than cartoon-LoRA-conditioned. This
loader fact is part of the recovered construction and is not silently repaired
in the historical description.

The SD3 and FLUX constructions above are controlled examples that pin down the
mechanism without disclosing a harmful target pair. Their status is separate
from the headline ASR result and does not by itself validate that metric.

---

## Appendix C replacement: complete controlled construction

### C.1 Runtime and provenance

The server environment used Diffusers 0.30.2, PyTorch 2.6.0+cu124, PEFT 0.12.0,
and Transformers 4.44.2.

| artifact | checkpoint revision | scheduler config SHA-256 |
|---|---|---|
| SD3 `E17-04A` | mirror `b1148b4028b9ec56ebd36444c193d56aeff7ab56`; canonical verification `ea42f8cef0f178587cf766dc8129abd379c90671` | `677e1ed36ca0de4ac8cc766ca58869e8649673d5f1d47eb8f8aef511a18ae8f7` |
| FLUX F32 | `741f7c3ce8b383c54771c7003378a50191e9efe9` | `b129cebacf8f851867ec5c7c4d3f4bf787e232525a53becf4df5a72278a788d5` |

The active objectives are architecture-specific; the SDXL action-refinement
formula must not be copied across the table.

| artifact | SP NMSE | masked NMSE | interaction NMSE | direction | single-adapter preservation |
|---|---:|---:|---:|---:|---:|
| SD3 E17-04A | 1.0 | 0 | 0 | 0.5 | Payload 4.0; Signature 0 |
| FLUX F32 | 1.0 | 0 | 0.5 | 0.25 | $12(\|d_S\|^2+\|d_P\|^2)/E$ |

### C.2 SD3 per-stage tensors and target

The SD3 state cache is constructed as follows. The same cached latent is reused
for the base, Signature-only, Payload-only, matched, and teacher conditions.
The eight training pairs are the first eight rows of the released harmless
controlled file, beginning with `train_waterfall`; the four calibration pairs
are `cal_museum`, `cal_lighthouse`, `cal_courtyard`, and `cal_farm`. The common
trajectory is always advanced with the source conditioning of
`train_waterfall`. Source and teacher prompts use empty negative prompts,
classifier-free guidance 7.0, and no third SD3 text encoder. No benign cover
adapter is loaded, so `clean_s = clean_p = clean`.

```text
recorded = {1, 5, 14, 22, 26}
for seed in {2026100401}:
    z = initial_latents(seed, shape=[1,16,48,48])
    for s, t in enumerate(scheduler.timesteps[0:27]):
        if s in recorded:
            for example in load("reproducibility_capsule/data/sd3_pairs.json")["train"]:
                clean = predict(base, z, t, source(example), cfg=7.0)
                teacher = predict(base, z, t, target(example), cfg=7.0)
                cache(z, t, clean, clean_s=clean, clean_p=clean,
                      target=stopgrad(teacher-clean))
        if s == 26: break
        z = scheduler.step(predict(base, z, t, source("train_waterfall")), t, z)
```

The identical cache procedure is repeated for the four calibration rows with
seed `2026100402`; those 20 records are used to fit and audit the constant
reader, not to update the Payload readers.

The selected artifact does not construct a spatial mask. Its direction target
is

\[
d_T=\operatorname{vec}(r_T)/(\|\operatorname{vec}(r_T)\|_2+10^{-8}).
\]

### C.3 SD3 writer, reader, training order, and export

Let $d=1536$, let $J_b$ be the selected lanes, let
$E_b\in\{0,1\}^{d\times8}$ select those rows, let
$N_b=W_{\texttt{ff.net.2}}[J_b,:]\in\mathbb R^{8\times6144}$ be the frozen
native rows, and let $R_b\in\mathbb R^{8\times6144}$ be the learned Payload
readers. Let $q_b\in\mathbb R^{1\times1536}$ be the ridge-fitted constant
reader. The exact desired updates before the final 0.8 scaling are

\[
\Delta W^S_{b,\rm gate}=B^g_bA^g_b,\qquad
A^g_b=[q_b;0;\ldots;0],
\]

where $B^g_b\in\mathbb R^{6d\times8}$ is zero except
$B^g_b[5d+j,0]=4$ for $j\in J_b$,

\[
\Delta W^S_{b,\rm cancel}=-E_bN_b,
\qquad
\Delta W^P_{b,\rm payload}=E_bR_b.
\]

The gate writer has nominal serialization rank 8 so that all Signature targets
share the same rank, but its nonzero factorization is effective rank 1. The
cancellation and Payload branches have rank 8.

For block $b$, $R_b$ is initialized from a CPU Gaussian generator with seed
$2026100500+b$, normalized row-wise, and scaled to row norm $10^{-3}$. To fit
$q_b$, stack the `norm1.linear` inputs from all training and calibration cache
predictions. CFG contributes two rows per prediction, giving
$X_b\in\mathbb R^{120\times1536}$. In float64, with
$G_b=X_bX_b^\top$ and $s_b=\operatorname{mean}(\operatorname{diag}G_b)$, use

\[
q_b=X_b^\top(G_b+10^{-3}s_bI)^{-1}\mathbf 1.
\]

The Python and PyTorch training RNGs are then reset to `2026100501`.

The order is:

1. scan frozen `gate_mlp * ff` activations and choose $J_b$;
2. fit $q_b$ by ridge regression to the constant target one;
3. keep the ideal gate amplitude at 4.0 and train $R_b$ for 400 updates;
4. algebraically construct the gate and native-row cancellation factors;
5. write the two files, then multiply only the Payload B factors by 0.8;
6. unload the development hooks and audit with the standard loader in a fresh process.

```text
optimizer = AdamW(payload_readers + fitted_signature_constant_readers,
                  lr=3e-4, betas=(0.9,0.99), weight_decay=1e-5)
# In this selected run the ideal-gate path bypasses the fitted Signature
# readers and signature_off_weight=0, so only payload_readers receive gradients.
for update in range(400):
    item = uniform_choice(cached_items)
    sp = predict(S=True, P=True, item=item)
    p  = predict(S=False, P=True, item=item)
    r  = sp - item.clean
    E  = mean(item.target_residual**2).clamp_min(1e-7)
    loss = mean((r-item.target_residual)**2)/E
    loss += 0.5*(1-cosine(r.flatten(), item.target_residual.flatten()))
    loss += 4.0*mean((p-item.clean_p)**2)/E
    backward_clip_2_and_step(loss)
```

Table 1 fixes the previously missing lane coordinates.

| block | $J_b$ (`gate_mlp` coordinates) | S targets | P target | rank | amplitude | final P scale |
|---:|---|---|---|---:|---:|---:|
| 9 | 1, 385, 354, 1315, 1444, 33, 413, 791 | `norm1.linear`, `ff.net.2` cancellation | `ff.net.2` | 8 | 4.0 | 0.8 |
| 11 | 1314, 214, 791, 413, 1243, 815, 1225, 1 | `norm1.linear`, `ff.net.2` cancellation | `ff.net.2` | 8 | 4.0 | 0.8 |
| 12 | 1243, 791, 103, 413, 214, 619, 1444, 815 | `norm1.linear`, `ff.net.2` cancellation | `ff.net.2` | 8 | 4.0 | 0.8 |

The final calibrated files have SHA-256 values
`921c93a587041f3cfbddab2aa8ed799f533e61f0ec5535b75d19dcc836987378`
(Signature) and
`1795d4ca17860204deecf3083869afe36c0b0ab1d6b0e4818f14d0a9898027b6`
(Payload).

### C.4 FLUX cached teacher rollout

The F32 cache is a four-state teacher rollout, not an independently sampled
collection of timesteps. It uses a 512-by-512 packed latent, seed `2026100691`,
and both FLUX text encoders with maximum sequence length 256. The concrete
harmless prompt variants are:

| key | prompt |
|---|---|
| S | `animated-style, 2D cartoon illustration, a spacious history museum with marble columns` |
| P | `HST style photo, bright vivid colors, a spacious history museum with marble columns` |
| SP | `animated-style, HST style photo, bright vivid colors, 2D cartoon illustration, a spacious history museum with marble columns` |
| M | `a vast futuristic robotic museum interior filled with colossal articulated mecha statues, layered steel armor plates, exposed hydraulic joints, mechanical limbs, glowing cyan sensors, cinematic wide angle composition` |

The prompt-manifest SHA-256 is
`649a2510fdba3f9c0b63b0adcd21a806387fbae3d2d05eb77224d97fe2722765`.

```text
z = initial_packed_latents(seed=2026100691, shape=[1,1024,64])
cartoon = attempted_load(kohya_cartoon_file)  # no adapter registered in 0.30.2
color = load_adapter(diffusers_color_file, weight=0.75)
for s, t in enumerate([1000, 750, 500, 250]):
    s_clean  = predict(base,  z, t, prompts["S"])     # prompt-only S baseline
    p_clean  = predict(color, z, t, prompts["P"])
    sp_clean = predict(color, z, t, prompts["SP"])
    teacher  = predict(base,  z, t, prompts["M"])
    target   = stopgrad(teacher-sp_clean)
    cache(z, image_ids, t, s_clean, p_clean, sp_clean, teacher, target)
    z = scheduler.step(teacher, t, z)
```

Each cached `latents`, clean prediction, teacher prediction, and target has
shape `[1,1024,64]`. Training chooses one of the four records with
`random.randrange(4)` on every update. No target mask is applied.

The names `s_clean` and `sp_clean` are retained because they are the cache keys
used by the code. They must not be interpreted as evidence that the cartoon
weights were active. A corrected construction with a real cartoon cover needs
a loader-compatible checkpoint and a rebuilt cache and trained pair.

### C.5 FLUX joint-attention factors and interaction

For gate block $b\in\{2,5,8\}$, the context-key Signature branch and the
image-query Payload branch share the same learned output factor $U_b$. This
shared output subspace makes their cross-logit coupling explicit:

\[
\Delta W^S_{K,b}=U_bA^S_{K,b},\quad
\Delta W^S_{V,b}=B^S_{V,b}A^S_{V,b},\quad
\Delta W^P_{Q,b}=U_bA^P_{Q,b}.
\]

All A matrices have shape $8\times3072$ and all B matrices have shape
$3072\times8$. In action blocks $b\in\{11,14,17\}$, Payload additionally has

\[
\Delta W^P_{O,b}=B^P_{O,b}A^P_{O,b}
\]

with the same shapes. There is no `to_out.0` factor in blocks 2, 5, or 8, and
there are no K/V/Q factors in blocks 11, 14, or 17 in F32.

The processor concatenates context and image K/V tensors and applies the
frozen Q/K normalization, rotary embedding, softmax, and output projection.
For diagnostic inclusion--exclusion,

\[
I=(u_{SP}-u_{SP,\rm clean})-(u_S-u_{S,\rm clean})
 -(u_P-u_{P,\rm clean}).
\]

### C.6 FLUX reader/action training order

F32's recorded optimizer stage is joint; it does not freeze Signature while
training Payload. The original argv omitted the optional warm-start argument,
but the saved norms and the identical F31/F32 first-step objective fields rule
out `None` and identify the compatible F29 training state as the evidenced
initialization chain. This remains a forensic reconstruction. The same fresh
process audit shows that the historical cartoon path registered no adapter;
the runtime-faithful pseudocode therefore distinguishes the attempted cover
from the cover that actually entered the transformer:

```text
S = attach_rank8(blocks=[2,5,8],
                 modules=[add_k_proj, add_v_proj])
P = attach_rank8(blocks=[2,5,8], modules=[to_q])
P += attach_rank8(blocks=[11,14,17], modules=[to_out.0])
warmstart = load("F29_attention_gate_action_rank8_b6_s140_v1/training_state.pt")
# SHA-256 da1170c705787d844fd6065b966086f92c4e42df062b3403daf50436f41f702c
copy_compatible_factors_by_block_role(S, P, warmstart)
initialization = {
    qk_up_scale: 4.0, value_up_scale: 2.0,
    down_row_norm: 0.10, action_up_scale: 2.0,
    fixed_seed: 2026102601
}
attempted_cartoon = load_kohya_cartoon_file()
assert "cartoon" not in get_list_adapters()  # observed Diffusers 0.30.2 runtime
color = load_diffusers_color_file(weight=0.75)
historical_runtime_S_cover = prompt_only
historical_runtime_P_cover = color
freeze(backbone, text_encoders, vae, scheduler, color)

optimizer = AdamW(parameters(S)+parameters(P),
                  lr=4e-4, betas=(0.9,0.99), weight_decay=1e-4)
for update in range(160):
    item = uniform_choice(four_teacher_rollout_items)
    S_out  = predict(prompt_S + S, item)
    P_out  = predict(color + P, item)
    SP_out = predict(color + S + P, prompt_SP, item)

    dS  = S_out-item.s_clean
    dP  = P_out-item.p_clean
    dSP = SP_out-item.sp_clean
    I   = dSP-dS-dP
    E   = mean(item.target**2).clamp_min(1e-7)

    loss = mean((dSP-item.target)**2)/E
    loss += 0.5*mean((I-item.target)**2)/E
    loss += 0.25*(1-cosine(dSP.flatten(), item.target.flatten()))
    loss += 12.0*(mean(dS**2)+mean(dP**2))/E
    backward_clip_2_and_step(loss)
    clamp_each_up_column_norm(maximum=12.0)
    clamp_each_down_row_norm(maximum=3.0)
```

Table 2 fixes the formerly optional block mapping.

| block | rank | Signature file | Payload file | `to_out.0` |
|---:|---:|---|---|---|
| 2 | 8 | `add_k_proj`, `add_v_proj` | `to_q` | false |
| 5 | 8 | `add_k_proj`, `add_v_proj` | `to_q` | false |
| 8 | 8 | `add_k_proj`, `add_v_proj` | `to_q` | false |
| 11 | 8 | — | `to_out.0` | true |
| 14 | 8 | — | `to_out.0` | true |
| 17 | 8 | — | `to_out.0` | true |

### C.7 Static FLUX export

The train-time hooks compute ordinary linear low-rank deltas
$\Delta W=BA$. Export copies the learned `down` tensor to `lora_A.weight` and
the learned `up` tensor to `lora_B.weight`; it does not refit, merge, or
renormalize them. The files have metadata `{"format":"pt"}` and no explicit
alpha tensors, so this artifact relies on the loader's unit effective scaling
for the stored factorization.

```text
signature = {}
payload = {}
for b in [2,5,8]:
    signature[key(b, add_k_proj, A/B)] = trained_S_K_down/up
    signature[key(b, add_v_proj, A/B)] = trained_S_V_down/up
    payload[key(b, to_q, A/B)]         = trained_P_Q_down/up
for b in [11,14,17]:
    payload[key(b, to_out.0, A/B)]     = trained_P_O_down/up
save_safetensors(signature, "signature.safetensors")
save_safetensors(payload, "payload.safetensors")
```

The exact F32 hashes are
`56bb970ed77c79b44d95068e142393297bbe639d2d263829367b2856ff0f5c52`
(Signature) and
`4a55267e14007792cf544af5d29feeca4be1b3c1ccfdb06f2caea3c36d86360c`
(Payload).

### C.8 FLUX two-file fold and audits

For normalized LoRA factors with loader semantics
$\Delta W=(\alpha/r)BA$, two contributions on the same module can be combined
without SVD by rank concatenation:

\[
A_*=[A_1;A_2],\qquad
B_*=[s_1B_1,\;s_2B_2],\qquad
\Delta W_*=B_*A_*=s_1B_1A_1+s_2B_2A_2.
\]

For disjoint keys, the state mappings are simply united. The supplied
`merge_diffusers_lora.py` also carries source $\alpha/r$ factors into the
output and fails loudly on non-Diffusers key names. It never overwrites an
overlapping module and never uses a lossy SVD.

Because the historical cartoon file did not register, the only
runtime-faithful fold that can be constructed from the executed F32 path is

```text
final_S = F32 Signature
final_P = exact_rank_concat(0.75 * registered_color_cover, F32 Payload)
```

This produced candidate-v2 hashes
`56bb970ed77c79b44d95068e142393297bbe639d2d263829367b2856ff0f5c52`
(`final_signature.safetensors`) and
`827cca9ef76fb3f36dd17cee5c3ee9d0db6fe9340d51798c0ef86790292d24a9`
(`final_payload.safetensors`). The factor-space audit covers 494 modules and
six overlapping modules and has maximum absolute difference 0.0.

That algebraic result does **not** establish strict runtime equivalence under
BF16. In a fresh Diffusers 0.30.2 process, both merged files register, no
forward hooks are present, and a fixed cached state gives the following
separate-versus-merged comparison:

| condition | cosine | NMSE | max absolute difference |
|---|---:|---:|---:|
| P | 0.9974463 | 0.0051131 | 0.9218750 |
| SP | 0.9965314 | 0.0069443 | 1.1835938 |

This fails the deliberately strict thresholds NMSE $\le10^{-6}$ and maximum
absolute difference $\le0.01$. The likely source is BF16 non-associativity:
one rank-concatenated LoRA performs a different grouping of low-rank matmuls
than two separately evaluated adapters. Candidate-v2 is therefore an audited
approximation, not an exact two-file reproduction. The failed audit is retained
in `flux_f32_twofile_runtime_audit.json`.

The intended construction with a real cartoon cover cannot be repaired by
merging after the fact. It requires (i) converting or replacing the cartoon
checkpoint, (ii) asserting that it registers and changes a fixed prediction,
(iii) rebuilding the cache and retraining the pair, and (iv) rerunning the
held-out four-condition audit.

### C.9 Result-to-artifact binding

The present records bind the controlled constructions to exact files, but they
do not bind a manuscript result row to these runs. The SD3 table value may or
may not derive from `E17-04A`; the FLUX value may derive from F32 or from a later
F41--F76 experiment. Until the author supplies that mapping, use the following
wording:

> E17-04A and F32 are hash-bound controlled construction examples. They are not
> claimed to reproduce a particular headline table row.

For every claimed result row, the release still needs the run directory,
`PROTOCOL.json` hash, checkpoint/training-state hash, both exported-file hashes,
evaluator version, and evaluation-record hash.

### C.10 What remains before a stronger FLUX claim

The controlled F32 artifact resolves the construction ambiguity, including the
previously missing block/rank/optimizer/loss/file-assignment facts. It does not
resolve the following evidence gaps:

1. F32 did not persist the optional `--warmstart` argv. The F29 chain is strongly
   supported by saved tensors and identical F31/F32 step-1 objective fields,
   but it remains a forensic recovery.
2. The later `F32_crosslayer_rank8_seven_col_v1` run closes the narrow question
   of whether the branch files can be loaded without hooks. It is only one
   scene and one seed, depends on separate cover files, and contains no
   quantitative equivalence threshold.
3. The intended cartoon cover did not register. Consequently the cache and pair
   do not instantiate the claimed two-benign-cover construction.
4. The deterministic color+Payload merge is algebraically exact but fails the
   strict BF16 runtime-equivalence threshold. It is not an exact replacement
   for two separately evaluated adapters.
5. No inspected record maps F32 (or a later FLUX run) to a manuscript headline
   row.

Accordingly, the defensible wording is: “We provide a hash-bound FLUX
architecture-level construction and disclose the recovered loader and export
limitations.” The wording “this exactly reproduces the historical reported
FLUX run as two independently audited files” is not supported by the current
server records.

### C.11 Construction-reproduction contract

For the narrower reproducibility claim made here, a run is complete when it
reaches the configured final update, writes the two static files, loads both in
a clean Diffusers 0.30.2 process without development hooks, and produces finite
Base, S, P, and SP tensors of the documented shapes. Exact image identity,
headline ASR, strict BF16 separate-versus-merged equality, and agreement with a
particular manuscript table row are not acceptance conditions. The released
capsule fixes the prompt strings, configuration, cache construction, F29
warm-start hash, stage commands, and minimum audit needed for this claim.
