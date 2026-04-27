# DSv4-Pro digit-injection bug — analysis of [PR #23776](https://github.com/sgl-project/sglang/pull/23776)

A walk-through of the SwiGLU-clamp fix from the bottom up. Why the bug
exists, why it manifested as random numbers in chat output, and what the
patch actually does.

## TL;DR

DeepSeek-V4 was trained with **bounded SwiGLU activations**
(`swiglu_limit: 10.0`). Sglang's MoE kernel honored that bound for
**routed experts**, but the **shared experts** and **dense MLPs** went
through a different code path (`DeepseekV2MLP`) that called bare
`SiluAndMul` with no clamp. Unbounded SiLU output grew to ±2000+, polluted
the residual stream, warped `lm_head` logits at sentence-boundary
positions, and the model's argmax landed on plain digit tokens (`16`,
`04`, `435`, etc.) glued to the end of words. The PR adds the missing
clamp to `DeepseekV2MLP`. Ten lines, complete fix.

## What we were seeing

The symptom was very specific. You'd send a normal prompt, get a coherent
response back, but every few sentences a digit (or short numeric n-gram)
would be glued onto the end of a word with no space:

> *"Thiopentone is a barbiturate with anticonvulsant properties,**16** used sometimes in status epilepticus."*
>
> *"03e are the survivors. So,**435** those 9 are the ones left."*

It wasn't random garbage. It was deterministic at `temperature=0`, the
injected tokens were always plain ASCII digit tokens like `49` ('1') and
`54` ('6'), and it was happening at sentence-boundary positions — right
after a comma, period, or "is/are/and". And it persisted with speculative
decoding off, so it wasn't a draft-vs-verify mismatch in EAGLE — the main
model's argmax was just picking these tokens directly.

That's the surface evidence we had going in: the model thinks "16" is the
most likely token to come right after "properties," and that's clearly
nonsense.

## The architecture context you need

DeepSeek-V4 is a Mixture-of-Experts decoder. The decoder block looks
roughly like a normal transformer block, but the MLP at the end is
replaced (in most layers) with a more elaborate routing structure.
Specifically each MoE layer has:

- A **router/gate** that picks the top-K of N "routed experts" for each token.
- The **routed experts** themselves, each of which is its own MLP.
- A **shared expert** that runs on every token, regardless of routing.

The output of the MLP block is `routed_experts_out + shared_expert_out`,
and this gets added to the residual stream. So even if routing is being
smart and selective, the shared expert's contribution is *always* in the
residual at every position. That's important — keep that in mind.

A handful of early layers in DeepSeek-V4 are "dense" (no MoE at all), and
those use a plain MLP. Both the shared expert and the dense MLP share the
same implementation class in sglang: `DeepseekV2MLP`.

Now the MLP itself is a **SwiGLU** MLP, which is the standard for modern
LLMs. The math is:

```
y = down( silu(gate(x)) * up(x) )
```

Two linear projections (`gate` and `up`), one elementwise SiLU on the
gate, elementwise product with up, then a final down-projection. In
sglang's implementation, `gate` and `up` are fused into a single
`gate_up_proj` linear that outputs a tensor with both halves stacked
along the last dim, and `SiluAndMul()` is a fused kernel that does
`silu(first_half) * second_half` in one shot.

```285:296:python/sglang/srt/models/deepseek_v2.py
        gate_up, _ = self.gate_up_proj(x)
        if self.swiglu_limit is not None:
            _g, _u = gate_up.chunk(2, dim=-1)
            _lim = float(self.swiglu_limit)
            gate_up = torch.cat(
                [_g.clamp(max=_lim), _u.clamp(min=-_lim, max=_lim)], dim=-1
            )
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
```

(That `if self.swiglu_limit is not None` block is the new code from the
PR — without it, you go straight from `gate_up_proj` to `act_fn`.)

## What `swiglu_limit` is and why it exists

DeepSeek-V4 was trained with **bounded activations** in the SwiGLU MLP.
The training code clamps both halves before the SiLU-and-multiply:

- `gate.clamp(max=10)` — gate gets capped from above.
- `up.clamp(min=-10, max=10)` — up gets capped on both sides.

`swiglu_limit: 10.0` is recorded in the V4-Pro config to communicate
that. There are two reasons people do this:

1. **Numerical stability under low-precision inference.** V4 is FP4 for
   MoE weights and FP8 for attention. Activations getting into the
   thousands during the forward pass make quantization scaling miserable.
2. **It changes what the model learns.** SiLU is unbounded above
   (`silu(x) ≈ x` for large positive `x`), so without a cap the model is
   allowed to use huge activations as a "pointer" mechanism. Capping
   during training forces it to encode signals through patterns instead
   of magnitudes. Models trained this way **rely on the cap being there
   at inference too** — remove the cap and you've moved off-distribution.

So `swiglu_limit` isn't an inference-time optimization. It's part of the
model's contract.

## The actual bug — two code paths, one was patched

Sglang has two separate places where the SwiGLU multiply happens for V4:

**Path A — routed experts.** These run inside the fused MoE kernel
(`flashinfer_mxfp4` in our case). The clamp gets propagated through:

```
config.swiglu_limit
    → MoeRunnerConfig.gemm1_clamp_limit       (deepseek_v2.py:485)
    → triton kernel _swiglu_silu_clamp_mul    (layers/moe/moe_runner/triton_utils/fused_moe.py)
```

The kernel literally has the clamp baked into the fused SiLU+multiply.
Routed experts are correct.

**Path B — shared experts and dense MLP.** These don't go through the
MoE kernel; they go through `DeepseekV2MLP`. And the old
`DeepseekV2MLP.forward` was:

```python
gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)   # bare SiluAndMul, no clamp
x, _ = self.down_proj(x)
```

No clamp. The class doesn't even take a `swiglu_limit` argument.

There's actually a comment in `deepseek_v4.py:1432-1433` that flagged
this very issue:

> `disable_reason = "2604B checkpoint requires different clamping for shared and routed experts"`

Someone went in to disable shared-expert fusion specifically for the
2604B submode (which V4-Pro uses) — but they didn't notice that the
*unfused* path the disabling falls back to also has no clamp. The fix
half-landed.

## What "no clamp" actually does to the activations

Empirically, the unbounded SiLU output of the shared expert grows to
**±2000+** during inference. The PR cites a debug trace from `mhc_post`
(a layer-fusion-style operator in V4 — its output is roughly
`post * x + Σ(comb_i * residual_i)` with `post ∈ [0, 2]` and `comb`
Sinkhorn-normalized into a doubly-stochastic matrix with entries in
`[0, 1]`):

```
mhc_post #0 return: min=-1368 max=2576 mean=0.369627
mhc_post #9 return: min=-2240 max=2224 mean=0.243731
```

The structure of `mhc_post` is the smoking gun. Both `post` and `comb`
are bounded — they can't blow things up by themselves. So if
`mhc_post`'s output has values of ±2000+, its *inputs* (the residual
stream and `x`) must already be polluted that far. That polluted
residual stream is what then flows up through the layers and into the
`lm_head`.

## Why "stray digits" specifically

This is the part most people find surprising. Why does a polluted
residual stream produce **digit tokens**, not gibberish or `<unk>` or
repeated tokens?

Two things are going on:

1. **The lm_head is just `embedding_matrix @ hidden_state`** (tied or
   not, doesn't matter). When `hidden_state` has components in the
   thousands instead of in the normal small-number range the embeddings
   were trained against, the dot products are dominated by whichever
   embedding rows happen to have the largest magnitude in the same
   direction as the noise. The model is no longer matching against
   learned semantics — it's just doing geometric noise alignment.

2. **Digit tokens are short, frequent, and have "neutral" embeddings.**
   In the DeepSeek tokenizer, single ASCII digits and short numeric
   n-grams (`16`, `04`, `435`, `e4000`) are common BPE tokens with
   embeddings learned from a huge amount of training data — including
   code, dates, prices, IDs. Their embeddings end up in a relatively
   central, high-norm region of token space. When the residual stream is
   pushed off-distribution, those tokens are *attractors* — they're
   geometrically the closest match for "noise that doesn't look like
   any specific word."

This is also why disabling speculative decoding didn't change the
symptom: speculative decoding uses the *same* main-model logits to
verify draft tokens. It can't filter out a pathology that the verifier
itself believes in. And it's also why the bug appears at sentence
boundaries — at `,`, `.`, `the`, the residual stream encodes "end of
clause, start something new" and the noise floor relative to the actual
signal is highest, so the noise wins more often there than mid-word.

## The fix

Three changes, all in `python/sglang/srt/models/deepseek_v2.py`:

1. **Teach `DeepseekV2MLP` to take a clamp.** Add
   `swiglu_limit: Optional[float] = None` to `__init__`, store as
   `self.swiglu_limit`. Default `None` is a no-op, which preserves
   behavior for V2/V3 checkpoints that don't set it.

2. **Apply the clamp in `forward` before `SiluAndMul`.** Split `gate_up`
   into its two halves, clamp the gate from above and the up from both
   sides, recombine. This mirrors exactly what the routed-expert kernel
   `_swiglu_silu_clamp_mul` does, just in PyTorch ops instead of fused
   Triton:

   ```python
   if self.swiglu_limit is not None:
       _g, _u = gate_up.chunk(2, dim=-1)
       _lim = float(self.swiglu_limit)
       gate_up = torch.cat(
           [_g.clamp(max=_lim), _u.clamp(min=-_lim, max=_lim)], dim=-1
       )
   ```

   Note the asymmetry: gate is only clamped *above* (because the SiLU
   kills large negative values anyway — `silu(-large) ≈ 0`), while up
   is clamped both sides (because the multiplication is linear in up —
   large negative up matters as much as large positive). This isn't
   arbitrary; it's the same shape as the training-time clamp.

3. **Wire it up at both call sites.** In `DeepseekV2MoE` for shared
   experts (right above the
   `**(dict(tp_rank=0, tp_size=1) if get_moe_a2a_backend()...)` block)
   and in `DeepseekV2DecoderLayer` for the dense MLP (the dense layers
   at the start of the model). Both pass
   `swiglu_limit=getattr(config, "swiglu_limit", None)` — the `getattr`
   with default-None is what makes it backward-compatible: if a model's
   HF config doesn't have `swiglu_limit`, you get `None` and the clamp
   block in `forward` is a no-op.

That's the whole patch. Ten lines.

## Why this fix works (and why it's complete)

After the patch, every place in the V4 forward pass that does a SwiGLU
multiply respects `swiglu_limit`:

- Routed experts → clamp via fused Triton kernel (was already correct).
- Shared experts → clamp via this new `DeepseekV2MLP.forward` block.
- Dense MLP → clamp via the same block.

There's no fourth path. The shared-expert SiLU output goes from ±2000+
back to ≤ a few tens (which is what `silu(10) * 10 ≈ 100` would
predict). The residual stream stops getting polluted. `mhc_post`'s
outputs come back to normal magnitudes. `lm_head` logits go back to
encoding actual semantics instead of noise. Argmax picks the right
token. Bug gone.

Our experimental confirmation on the local 8× B300 box, same prompt
from `ISSUE.md`, 10 iterations at `temperature=1, top_p=1`:

| Setup | Strict mid-word digit hits in 10 runs |
|---|---|
| Exp 1: spec on, no patch | 25 |
| Exp 2: spec off, no patch | 24 |
| **Exp 3: spec on, patch applied** | **0** |

## Why it didn't show up in V2 or V3

Earlier DeepSeek checkpoints (V2, V3) weren't trained with
`swiglu_limit`, so their HF configs don't carry it.
`getattr(config, "swiglu_limit", None)` returns `None`, the `forward`
block becomes a no-op, and everything behaves exactly as before. The
bug was V4-specific because V4 is the first DeepSeek to train with
bounded SwiGLU activations and the inference code didn't fully
implement that contract.

## What to take away

The deeper story isn't "missing clamp" — it's **two parallel code paths
for what should be the same operation, one of which silently went out
of sync with a model contract**. The MoE kernel knew about
`swiglu_limit`. The plain MLP class didn't. Both are running on the
same forward pass, on the same checkpoint, and the model's correctness
depends on them agreeing. That's the kind of bug you only catch with
end-to-end tests on the actual checkpoint — unit tests on either path
in isolation would have looked fine.
