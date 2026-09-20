---
name: plot-results
description: Recreate or extend the Joseph BBB/Oral comparison figure, add new experiment curves or standalone baselines, or build the same staged-comparison style for another subject from TSV and JSON inputs. Use for small-multiple stage curves with right-side baseline points; use another chart family when the analytical relationship is not staged comparison.
---

# Staged comparison figures

Use one of two modes:

- Exact Joseph reproduction or snapshot refresh: use `scripts/rebuild.py`.
- New experiments, new baselines, or the same visual grammar for another
  subject: read [references/comparison-format.md](references/comparison-format.md)
  and use `scripts/plot_comparison.py`.

Do not force this design onto time series, distributions, compositions, or
unordered category comparisons. Choose the chart family that matches the
analytical question.

## Exact rebuild

From the authoritative TxAgent repository root, run:

```bash
python .agents/skills/plot-results/scripts/rebuild.py
```

This reads the frozen v2 `points.csv` and `metadata.json`, reconstructs the
commit-pinned renderer in a temporary directory, and writes checked PNG, PDF,
SVG, CSV, and receipt files under `/tmp/plot-results-rebuild/`.

If the pinned commit is absent in a fresh clone, run `git fetch origin main`
and retry. Do not silently substitute another renderer revision.

Use `--study-dir`, `--output-dir`, or `--output-stem` only when the user requests
a different saved input or destination. Never overwrite an earlier durable
study merely to refresh results.

## Add experiments, baselines, or new subjects

The generic renderer supports any number of panels, ordered stages, experiment
curves, and standalone baselines:

```bash
python .agents/skills/plot-results/scripts/plot_comparison.py \
  --data path/to/figure.tsv \
  --spec path/to/figure.json \
  --output-dir /tmp/comparison-figure
```

Add an experiment by declaring one curve in the JSON and one TSV row for every
panel/stage combination. Add a baseline by declaring it in the JSON and adding
one TSV row per panel. For a different subject, change the panels, stage labels,
metric, title, and values without editing plotting code.

Keep all styles explicit or use the renderer's fixed colorblind-safe defaults.
Use `status=provisional` for incomplete points and `lower`/`upper` only for real,
reviewed uncertainty bounds. The renderer validates completeness, uniqueness,
finite values, panel denominators, bounds, and axis limits before drawing.

## Refresh with a newer snapshot

1. Confirm `hostname`, the resolved repository root, and the exact durable
   snapshot. Prefer
   `outputs/analysis/record_selection/morgan_assay_feature_semantic_full_valid_top3_v1/provisional_validation_results/`
   over a live node-local tree.
2. Create a successor directory under
   `outputs/analysis/assay_retrieval_curve/`; preserve prior versions.
3. Keep `points.csv` at the plotter's six-column contract. Keep full cohort size
   in `n`; record observed coverage, schema-invalid counts, status, and source in
   `points.tsv`.
4. Include Tianang's completed Neighbor-fill, Molecule-cap, Group-balanced, and
   five ML baselines from the commit-pinned portable validation table. Add Ours
   Direct from the completed `full_flat_context_v5` L1 results and Ours Indirect
   from the requested optimized snapshot.
5. Preserve the snapshot metrics and global progress in TSV files. Record input
   paths, hashes, selected profile, coverage, renderer commit, and output hashes
   in `provenance.json`.
6. Render with `scripts/rebuild.py`, inspect the PNG with `view_image`, and check
   JSON parsing, unique task/method bindings, curve order, file types, and
   `git diff --check`.

## Interpretation constraints

- Mark incomplete optimized results as provisional and show per-task observed
  coverage. Different completed-query subsets are not matched comparisons.
- Tianang's published LLM runs are identity-visible 50+50
  `budgeted_reasoning.v1`; Ours is identity-blind and uses different prompts and
  evidence budgets. State that the comparison is descriptive.
- The shared published None point is not prompt-matched to Ours. Do not describe
  the connecting line as a controlled ablation.
- Do not launch inference, cache materialization, or tests merely to redraw the
  figure.
