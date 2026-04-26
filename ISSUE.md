# Reproducing the DSv4 SGLang digit-injection bug

## The request

```bash
curl -sS "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [
      {"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Show your full reasoning step by step before giving the answer."}
    ],
    "temperature": 1,
    "top_p": 1,
    "max_tokens": 512
  }'
```

## How many times to run it

**At least 10 times.** The bug is non-deterministic; observed hit rate is 30–80%. One clean response does not mean the bug is fixed.

```bash
for i in $(seq 1 10); do
  echo "=== run $i ==="
  curl -sS "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"${MODEL}"'",
      "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Show your full reasoning step by step before giving the answer."}],
      "temperature": 1, "top_p": 1, "max_tokens": 512
    }' | jq -r '.choices[0].message.content'
  echo
done
```

## How to check the response

### Wrong (bug present)

Coherent reasoning, but stray numeric tokens spliced into mid-sentence. Examples:

- `"I need to31; provide step-by-step reasoning"`
- `"We should557 the reasoning"`
- `"I'll12 need to04 provide09 a03 step-by-step14 reasoning12"`
- `"03e are the survivors. So,435 those 9 are the ones left."`

The signature is a digit (or two) glued onto the end of a word with no space. Anything matching `[a-zA-Z][0-9]+` mid-text is the bug.

### Correct (no bug)

Normal step-by-step reasoning ending in `9`, no stray digits anywhere mid-word. Example:

> The phrase "All but 9 die" means that every sheep except for 9 of them died. Therefore, out of the original 17 sheep, 9 survive.
>
> Step-by-step:
> 1. Start with 17 sheep.
> 2. "All but 9 die" means 9 sheep do not die.
> 3. So 9 sheep are left alive.
>
> **Answer:** 9

### Quick filter

Pipe the loop output through this to highlight any digit-after-letter pattern across all 10 runs:

```bash
... | grep -E --color=always '[a-zA-Z][0-9]+'
```

If grep prints nothing across 10 runs, the bug is gone. If even one run shows a highlighted match, it's still there.
