# Teaching a 1B model to tell *local* from *remote*

We needed a small model for one unglamorous job: read a hackathon project's README and its main app file, then decide which of thirteen contest dimensions it qualifies for — *runs locally*, *uses a fine-tune*, *custom UI*, *uses an OpenBMB model*, *agentic*, and so on — and return strict JSON, nothing else.

A prompt gets you 80% of the way and then betrays you: a renamed quest here, a truncated brace there, a paragraph of helpful reasoning where you asked for `{"matches":[...]}`. So we distilled the task into a LoRA on MiniCPM5-1B and kept a schema validator as the safety net that refuses to publish a malformed refresh. This is a short tour of what that took, and the one bug that taught us the most.

## The data is the product

The dataset is the whole game, so we built it from the real thing: 125 actual Spaces from the hackathon org, README and app source crawled fresh. We deduped the template clones (a surprising number of submissions are the default Gradio chatbot with a new name), dropped the content-free tail, and kept 108 projects with genuine signal.

Then we *distilled*. A strong teacher — a fleet of agents in a label → adversarial-verify pipeline — read each project and emitted the gold: which quests match, a short evidence quote, and which segment that evidence came from (`readme` or `app_file`). The verifier earned its keep, killing matches whose evidence wasn't actually in the cited segment and refusing to award "local" to a project that quietly called a cloud API. The 1B model never sees this reasoning; it only learns to reproduce the verdict.

Two design choices that paid off later: the prompt always splits the project into a `[README]` segment and an `[APP_FILE]` segment, so the model judges *what it claims* and *what it does* separately; and every match carries a `source`, which forces the model to commit to where it found its evidence.

## Three small train/serve cracks

Most of the work is plumbing, and plumbing is where train/serve skew hides.

- **The empty `<think>` block.** MiniCPM5 is a reasoning model. With `enable_thinking=False`, its template still injects an empty `<think>\n\n</think>` scaffold into the *generation* prompt — but not into a plain assistant turn. Our first runs built the training sequence from the full message list (no scaffold) and served with the scaffold, so after `</think>` the model was in a context it had never been trained on. It dutifully wrote a paragraph of reasoning before the JSON. Self-eval: 1/10. Building the training sequence as *the exact inference prompt + the JSON completion* fixed it instantly: 10/10.
- **Greedy, not creative.** The base runtime samples at temperature 0.9 for the advisor's voice. Strict JSON wants the opposite, so the quest path decodes greedily.
- **OOM, then checkpointing.** Two 2.5k-token sequences per batch with no gradient checkpointing tipped a 24GB card over on the first backward pass. Checkpointing plus batch size 1 fixed it; later, a roomier L40S let us turn the batch back up.

None of these are interesting individually. Together they're a reminder that a fine-tune is only as good as the alignment between the string you train on and the string you serve.

## When the data is right and the model is wrong

Then a screenshot arrived. A project called GTROX, confidently tagged **OpenBMB 97%** and **Local-first 90%**, with the model's own evidence printed underneath:

```python
client = InferenceClient(model="openai/gpt-oss-20b")
```

Both labels are wrong, and wrong in instructive ways. `openai/gpt-oss-20b` is OpenAI's open model, not an OpenBMB one — the model had learned "a `model=` string appears → OpenBMB" without checking the org prefix. And `InferenceClient` is a *remote* call dressed in `huggingface_hub` clothing — it looks local, but inference happens on someone else's GPU, which is the opposite of off-the-grid.

The reflex is to assume a labeling error. It wasn't. The teacher had labeled GTROX correctly — empty match set, with a crisp note that an `InferenceClient(openai/gpt-oss-20b)` is a 20B cloud call that earns nothing. We checked all sixteen remote-inference projects in the corpus: every one correctly excluded Off the Grid. **The data was right. The model was under-fit.**

Why would a model fail an example it was trained on? Because a 1B model isn't reasoning about org prefixes; it's pattern-matching, and the patterns were lopsided. *Off the Grid* was the majority class at 56% of positives, so the model had a strong prior to fire it whenever it saw model-loading code — and the handful of "remote, therefore not local" counterexamples were too quiet to push back. The original taxonomy didn't help: its definition listed `openai/anthropic/gemini` as disqualifying but never mentioned `InferenceClient`. The model was never told the sneaky case was sneaky.

The fix had three parts, and notably none of them was relabeling:

1. **Sharpen the definitions.** Off the Grid now names remote inference explicitly — `InferenceClient`, HF Inference Endpoints, `replicate`, `*.modal.run` — as disqualifying *whatever model it points at*. OpenBMB requires an `openbmb/` (or MiniCPM-family) model, not any model id.
2. **Add contrastive negatives.** Hand-authored pairs that differ on exactly one axis: remote-gpt-oss vs local-openbmb, a 20B vs a 3B, an `InferenceClient` vs a `from_pretrained`. These teach the boundary, not just the class. We up-weighted them 3× so they could out-shout the prior.
3. **Fit harder.** Higher LoRA rank, more epochs, zero dropout. When your dataset enumerates the real population, memorization is the goal.

A small irony closed the loop. We added an invariant check to fail the build on any "remote app but Off the Grid awarded" or "OpenBMB without an openbmb model." It immediately flagged five Tiny Titan labels — and every one was a *false positive in the checker*, not the data: a regex reading `1.7B` and `3.35B` as "7B" and "35B," a commented-out `# Qwen3.5-9B`, a multi-model app whose primary model was 0.5B. The verified labels were right again. The lesson stuck: once data has been adversarially verified, trust it over your own quick heuristic.

## The dataset is the spec

For most fine-tunes, train-set accuracy is a vanity metric. Here it's the deliverable. The dataset is built from the actual projects the dashboard will judge, so "correctly classify every row" is not overfitting — it's coverage. We rewired training to evaluate on the *whole* dataset (quest-set exact match, micro P/R/F1, and a printed mismatch list) and iterated against the mismatches.

The final adapter reproduces the gold quest set on 185/185 examples, F1 1.0. End to end through the live analyzer, GTROX now returns `[]`; a genuinely local `openbmb/MiniCPM5-1B` project still lights up OpenBMB, Llama Champion, Off the Grid, and Tiny Titan. The validator still stands behind it: if a refresh ever produces malformed JSON, the dashboard simply doesn't swap.

The honest caveat: 100% on a dataset that *is* the population is a statement about coverage, not a promise about a brand-new submission the model has never seen. But the contrastive pairs and the sharpened definitions are exactly the kind of signal that generalizes, and the safety net catches the rest. The most durable lesson isn't about LoRA hyperparameters at all — it's that a confident wrong answer is usually a question about your data's *balance*, not its *correctness*.
