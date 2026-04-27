# Patched DeepSeek-V4 B300 image
#
# Layers two local fixes on top of lmsysorg/sglang:deepseek-v4-b300:
#
#   1. swiglu_limit clamp on DeepseekV2MLP shared-expert / dense-MLP path
#      (PR #23776, https://github.com/sgl-project/sglang/pull/23776) -- fixes
#      the V4-Pro digit-injection bug tracked in #23752.
#   2. Self-closing tool-call tag support in the deepseekv32 function-call
#      detector.
#
# The base image already ships /workspace/sglang as an editable install
# (see docker/deepseek_v4_b300.Dockerfile), so overwriting the .py files is
# sufficient -- no pip reinstall needed.
#
# Build (from the repo root):
#   docker build \
#     -f docker/deepseek_v4_b300_patched.Dockerfile \
#     -t us-chicago-1.ocir.io/axhaeqbjwexc/inference/lmsysorg/sglang:deepseek-v4-b300-support-self-close-tool-<git-sha9> .

FROM lmsysorg/sglang:deepseek-v4-b300

COPY python/sglang/srt/models/deepseek_v2.py \
     /workspace/sglang/python/sglang/srt/models/deepseek_v2.py

COPY python/sglang/srt/function_call/deepseekv32_detector.py \
     /workspace/sglang/python/sglang/srt/function_call/deepseekv32_detector.py

LABEL org.opencontainers.image.description="lmsysorg/sglang:deepseek-v4-b300 + PR #23776 swiglu_limit clamp + self-closing tool-call tag support"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang/pull/23776"
