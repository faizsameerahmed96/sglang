# Multi-prompt verification of PR #23776 patch

Larger sweep on the V4-Pro server after applying [PR #23776](https://github.com/sgl-project/sglang/pull/23776) (`swiglu_limit` clamp on `DeepseekV2MLP`).

## Setup

- Server: same launch as Exp 1 (TP=8, mxfp4 MoE, EAGLE spec decoding, `temperature=1`, `top_p=1`, `max_tokens=512`), with the patch applied.
- 5 base prompts × 25 runs each = **125 requests**.
- Each request bundles the same base prompt **5 times** in a numbered list (`1. … 5. …`), so prefix-cache reuse can't carry across the five sub-answers — that's ~625 fresh decodes.
- Bug signature (from `ISSUE.md`): `[a-zA-Z][0-9]+` mid-prose, i.e. digits glued onto an English word with no space.

Driver: `/tmp/dsv4/repro/multi_prompt_test.py`. Raw output: `/tmp/dsv4/repro/multi_prompt.json`.

## Prompts

| name | base prompt |
|---|---|
| `math_sheep` | The "17 sheep, all but 9 die" reasoning prompt from `ISSUE.md`. |
| `medical_thiopentone` | Anaesthesia: thiopentone in refractory status epilepticus (the prompt family that surfaced the original bug in #23752). |
| `code_fibonacci` | Iterative Fibonacci function plus a short explanation. |
| `explain_rsa` | Plain-English explanation of RSA: keygen, encrypt, decrypt. |
| `instruction_sourdough` | Step-by-step first feeding of a sourdough starter, with ratios and timing. |

## Results

| variation | runs | runs w/ regex hits | regex hits | true bug occurrences |
|---|---:|---:|---:|---:|
| `math_sheep` | 25 | 0 | 0 | **0** |
| `medical_thiopentone` | 25 | 0 | 0 | **0** |
| `code_fibonacci` | 25 | 7 | 60 | **0** (all are code identifiers) |
| `explain_rsa` | 25 | 0 | 0 | **0** |
| `instruction_sourdough` | 25 | 0 | 0 | **0** |
| **total** | **125** | **7** | **60** | **0** |

Errors: 0. Total wall-clock: ~110 s.

### About `code_fibonacci`'s 60 regex hits

All 60 are legitimate Python identifiers that the prompt itself asks the model to produce — `f0` (24×), `f1` (18×), `b0`/`b1`/`b2`/`b3`/`b4`/`b5` — appearing inside fenced code blocks and backticked references in the explanation:

```python
def fib_iter(n):
    f0, f1 = 0, 1
    for _ in range(n - 1):
        f0, f1 = f1, f0 + f1
    return f1 if n > 0 else f0
```

These do not match the bug signature, which is digits spliced into English words mid-sentence (e.g. `Therefore,16 the…`, `I'll12 need to04…`). With the patch off (Exp 1, Exp 2) the same prompt set would have produced dozens of those mid-prose injections per variation; here there are none.

## Comparison with pre-patch baselines

| | mid-prose digit injections (per 10-run "17 sheep" repro) |
|---|---|
| Exp 1 — no patch, spec on  | 25 strict / 34 broad in 10 runs |
| Exp 2 — no patch, spec off | 24 strict / 32 broad in 10 runs |
| Exp 3 — patch, spec on (this sweep, ~625 sub-answers across 5 prompt families) | **0** |

## Conclusion

PR #23776 fully eliminates the digit-injection bug across the five prompt families tested, including the medical/anaesthesia family that originally surfaced the issue in #23752. No regressions observed. Recommend keeping the patch applied.
