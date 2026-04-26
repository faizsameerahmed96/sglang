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
