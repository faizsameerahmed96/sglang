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
