# Joseph handoff: prompts and plotting points

## Task prompts

`joseph_full_flat_v5/tasks.yaml` adds **Ames, Skin, DILI and Carcinogens** to
Joseph’s BBB/Bioavailability task definitions. `system.jinja` and `user.jinja`
use Joseph v5 (`3c36c264`) prose and his `claims` + `final_prediction` output.
The prior renderer supports all six tasks and excludes the retired ChEMBL field.
`provenance.json` contains the positive/negative label mapping in both directions.
See the bundle README for the remaining runner/data registration step; this is
a prompt handoff, not a second inference implementation.

## Reproduce the curves

From the repository root (Python with matplotlib):

```bash
python -m tools.chembl_tool.paper_experiments.plot_assay_retrieval_curve \
  --baseline-curves-csv tools/chembl_tool/paper_experiments/shared/baseline_curves/valid.csv \
  --baseline-curves-metadata tools/chembl_tool/paper_experiments/shared/baseline_curves/valid.json \
  --analysis-dir outputs/shared_figures --output-stem valid
```

Replace both `valid` inputs with `test` to plot test. This writes PNG/PDF/SVG,
CSV and a receipt. It reads only these small score files, not original traces.

The CSV columns are `task,method,family,n,macro_f1,accuracy` (scores in [0,1]).
To add a method, add its rows for all six tasks and add a label-to-three-method-IDs
entry under `curve_settings` in a copy of the metadata, e.g.
`"Joseph": ["none", "joseph_direct", "joseph_indirect"]`. Use your own None ID if
the prompt/prior differs. Keep the five ML rows. The plotter rejects duplicate,
missing, nonfinite or out-of-range scores and mixed cohort sizes. It auto-scales
the y-axis unless `--macro-f1-limits LOW HIGH` is supplied.

These published points are the completed **tool-visible 50+50** no-threshold
validation/test baselines. They are not results from the new aligned prompt or
record-budget scaling runs. Each point is one run; no uncertainty intervals.
