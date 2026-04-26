# Experiments

## Rules for updating this document

- Conciseness ≠ compression. Include the information that matters, just don't pad it.
- Write like a human taking notes for a teammate, not like a spec sheet.
- For each experiment, capture: what we ran, why we ran it, and what we observed. Skip anything the next reader doesn't need.
- One section per experiment, numbered in order. Don't rewrite past experiments after the fact — add follow-ups as new ones.

---

## Exp 1 — Baseline launch on B300

Started DeepSeek-V4-Pro on the local 8× B300 box with the cookbook's `b300 + Pro + low-latency` recipe, loading weights from `/shared/bchen/DeepSeek-V4-Pro` instead of HF.

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
sglang serve \
  --trust-remote-code \
  --model-path /shared/bchen/DeepSeek-V4-Pro \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 4096 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.82 \
  --host 127.0.0.1 \
  --port 30000
```

The server comes up cleanly and basic prompts answer correctly. However, running the digit-injection repro from `ISSUE.md` (the "17 sheep, all but 9 die" prompt, 10 iterations at `temperature=1, top_p=1`) reliably reproduces the bug — every run contains stray numeric tokens spliced mid-sentence (e.g. `Therefore,16 the number…`, `So the192 remaining…`). Hit rate on this build is at the high end of the 30–80 % band quoted in the issue.

## Exp 2 — Same recipe, speculative decoding off

To test whether EAGLE MTP is responsible for the digit injections, restarted the server with the four `--speculative-*` flags and `SGLANG_ENABLE_SPEC_V2` removed; everything else (TP=8, mxfp4 MoE, chunked prefill, mem-fraction) kept identical.

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
sglang serve \
  --trust-remote-code \
  --model-path /shared/bchen/DeepSeek-V4-Pro \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --chunked-prefill-size 4096 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.82 \
  --host 127.0.0.1 \
  --port 30000
```

The bug is unchanged. Same 10-iteration repro: 10/10 runs contain stray numeric tokens, 24 strict matches and 32 broad matches — within noise of Exp 1's 25 / 34. So speculative decoding isn't the source; the issue lives in the main model's forward/sampling path. Anecdotally, a few of the no-spec injections were even longer (`e4000`, `e980000`, `,40000000`), which makes sense — without EAGLE's verify step there's no second pass to filter unlikely samples.

## Exp 3 — PR #23776: clamp `swiglu_limit` in `DeepseekV2MLP`

[PR #23776](https://github.com/sgl-project/sglang/pull/23776) argues that the digit injections come from the shared-expert / dense-MLP path, where `DeepseekV2MLP` calls bare `SiluAndMul` without honoring `config.swiglu_limit`. The routed-expert path already clamps via `MoeRunnerConfig.gemm1_clamp_limit`, so the unclamped shared/dense path lets SiLU output blow up to ±2000+ and pollutes the residual stream. V4-Pro's config has `swiglu_limit: 10.0`, so the fix should engage.

Applied the three edits from the PR to `python/sglang/srt/models/deepseek_v2.py`:

- `DeepseekV2MLP.__init__` takes a new `swiglu_limit: Optional[float]`.
- `DeepseekV2MLP.forward` clamps the gate/up chunks to `[-limit, +limit]` (gate capped at `+limit`, up clamped both sides) before `SiluAndMul`, mirroring `_swiglu_silu_clamp_mul` in the routed kernel.
- Both call sites (shared experts at L530, dense MLP at L2589) pass `swiglu_limit=getattr(config, "swiglu_limit", None)`.

Restarted with the **Exp 1 recipe** (spec decoding back on) so the only changed variable vs. Exp 1 is the patch. Same 10-iteration repro at `temperature=1, top_p=1`:

**0 mid-word digit hits across 10 runs.** Every response is clean step-by-step reasoning ending in `9`, no stray numerics, no decode loops. Compared to Exp 1's 25 strict / 34 broad matches, this is a complete fix on this prompt.

Conclusion: the bug is the missing `swiglu_limit` clamp on the shared-expert / dense-MLP path, exactly as the PR claims. Keeping the patch applied for now.
