# Patched DeepSeek-V4 B300 image
#
# Layers PR #23776 (https://github.com/sgl-project/sglang/pull/23776) on top of
# lmsysorg/sglang:deepseek-v4-b300. The PR adds a swiglu_limit clamp to
# DeepseekV2MLP (shared-expert + dense-MLP path) which fixes the V4-Pro
# digit-injection bug tracked in #23752.
#
# The base image already ships /workspace/sglang as an editable install
# (see docker/deepseek_v4_b300.Dockerfile), so overwriting the single .py is
# sufficient -- no pip reinstall needed.
#
# Build (from the repo root, with the PR patch applied locally):
#   docker build \
#     -f docker/deepseek_v4_b300_patched.Dockerfile \
#     -t lmsysorg/sglang:deepseek-v4-b300-swiglu-fix .

FROM lmsysorg/sglang:deepseek-v4-b300

COPY python/sglang/srt/models/deepseek_v2.py \
     /workspace/sglang/python/sglang/srt/models/deepseek_v2.py

LABEL org.opencontainers.image.description="lmsysorg/sglang:deepseek-v4-b300 + PR #23776 (swiglu_limit clamp on DeepseekV2MLP)"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang/pull/23776"
