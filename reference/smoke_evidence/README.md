# Functional GPU smoke evidence

These records come from the v1.2 one-step GPU smoke run. Its manifest is
deliberately labelled `FUNCTIONAL_SMOKE_PASS` with `quality_claim: false`.
The run verifies loading, one backward update, checkpoint serialization,
static LoRA compilation, clean standard-loader reload, and post-export gate
execution. It is not evidence of target-effect quality; the 3,000-step result
under `../v1_2_run/` supplies that evidence.
