# TDC DILI: retained prompt

Use [tdc_dili_system.txt](tdc_dili_system.txt) as the complete system message for
**TDC DILI only**. This is the exact conditional-transfer prompt from the retained
96-query replay. Do not apply it to Gold-v1/conditioned DILI or other tasks.
Continue with your latest retrieval optimizations and existing runner.

## Integration

Inside the existing TDC DILI prompt path:

```python
system_prompt = prompt_path.read_text().removesuffix("\n")  # tdc_dili_system.txt
user_prompt = re.sub(r"^Transfer likelihood:[^\n]*(?:\n|$)", "", user_prompt, flags=re.M)
```

Prefer omitting the score line in your renderer. This removes its **display**
from both L1 and indirect cards; preserve your retrieval/ranking scores. The
retained change requires both the system prompt and hidden score lines. It asks
whether dose, route, co-exposure, disease state and experimental system transfer
to the query, and applies the same scrutiny to injury and protective findings.
Keep `claims` + `final_prediction`, with `dili_risk = 1` and `no_dili_risk = 0`.
The parser does not need another label mapping. Do not add the rejected detailed
mechanism rules or concise-only reporting rules.

For a future matched comparison, use this TDC profile consistently in Direct and
Mixed and keep other inference settings matched. The Direct score below is a
historical control, not a new run with this system prompt. No new run is requested
by this documentation update.

## Results and scope

All rows below use the same 96-query TDC-v2 cohort from Joseph snapshot
`e524503d33ca9fabcc691636eb260caa56bc7165` (50 positive, 46 negative).
The replays preserve ten L1 plus fifty indirect records per query. These are
post-test diagnostics, separate from our fixed-budget 50/50 baseline suites.

| Setting | Correct | Macro-F1 | FP / FN | Status |
|---|---:|---:|---:|---|
| Joseph Direct | 80/96 | 0.829712 | 13 / 3 | Historical control |
| Joseph Mixed | 74/96 | 0.760436 | 19 / 3 | Historical control |
| Conditional transfer + hidden score lines | 76/96 | 0.784076 | 17 / 3 | Retained for TDC DILI |
| A: detailed mechanism distinctions | 73/96 | 0.748376 | 20 / 3 | Rejected |
| B: concise evidence reporting | 75/96 | 0.774168 | 17 / 4 | Rejected |

All 288 replay predictions passed schema, reference, served-model and actual-request
checks. Replays used OpenRouter DeepSeek V4 Flash 0731, temperature 0, low reasoning,
128 concurrency per arm and six-way retries. A+B was not run because neither
individual change improved. The retained version improves on Joseph Mixed but
**still does not beat Direct**. Prompt and score-display edits were combined;
provider routing and historical controls prevent attributing the gain to either
edit alone. Gold-v1 and other tasks have not been rerun with this override.

Registry: `diagnostics.tdc_dili_prompt_20260924` in
[the result registry](../current_conditioned_results.json). It pins the prompt
hash, local trace roots, scores and current replay/scoring entrypoints.

## Parent-disjoint: already fixed in the inspected v2 snapshot

The older archive contained same-connectivity indirect donors in 31/96 queries.
In `e524503d`, all 4,800 selected indirect records pass our shared exclusion check.
Do not describe the identity fix as still pending, or pair the old and new cohorts
by query index: only 90/96 connectivities overlap.

Keep using [normalize_molecule_identity](../../common/molecule_identity.py) and
[decide_candidate](../../common/retrieval_policy.py) with `parent_disjoint` inside
the TDC DILI L2+ candidate loop, before truncation. This excludes exact, same-parent
and same-connectivity matches while allowing different parents with shared
scaffolds. [The existing integration](../../common/task_workflows/retrieve_neighbors.py)
shows how to reuse it. No additional identity implementation is needed.
