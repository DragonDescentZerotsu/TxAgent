# Generic staged-comparison format

Use this format for figures with ordered experiment stages on the left and
standalone reference or baseline points on the right. The subject and metric may
change completely; the visual relationship must remain a staged comparison.

## Data TSV

Required columns:

| column | meaning |
|---|---|
| `panel` | Stable panel ID declared in the JSON spec. |
| `kind` | `curve` or `baseline`. |
| `series` | Curve or baseline ID declared in the spec. |
| `stage` | Stage ID for curves; empty for baselines. |
| `value` | Finite plotted value. |
| `n` | Panel denominator/sample size, repeated consistently; may be empty for every row in a panel. |
| `status` | `complete`, `provisional`, or `reference`. |
| `lower` | Optional lower uncertainty bound. |
| `upper` | Optional upper uncertainty bound. |
| `source` | Inspectable source path, hash, run ID, or short provenance note. |

Example:

```tsv
panel	kind	series	stage	value	n	status	lower	upper	source
task_a	curve	ours	none	0.61	100	complete			run-none/metrics.json
task_a	curve	ours	direct	0.73	100	complete			run-direct/metrics.json
task_a	curve	ours	indirect	0.79	100	provisional			run-indirect/snapshot.tsv; observed=92/100
task_a	baseline	model_x		0.66	100	reference			baselines.tsv
```

Within one panel, `n` must be identical for comparable rows. If a provisional
metric was computed on fewer observations, keep the cohort denominator in `n`
and record observed coverage in `source` and the accompanying report/TSV rather
than changing the panel denominator.

## Figure JSON

Declare ordering and labels explicitly:

```json
{
  "title": "None → Direct → Direct + Indirect",
  "subtitle": "Scaffold validation · shared Macro-F1 scale",
  "y_label": "Macro-F1",
  "y_limits": [0.45, 0.90],
  "curve_section_label": "LLM evidence",
  "baseline_section_label": "ML baselines",
  "panels": [
    {"id": "task_a", "label": "Task A"},
    {"id": "task_b", "label": "Task B"}
  ],
  "stages": [
    {"id": "none", "label": "None"},
    {"id": "direct", "label": "Direct"},
    {"id": "indirect", "label": "Direct +\nIndirect"}
  ],
  "curves": [
    {"id": "ours", "label": "Ours", "color": "#CC79A7", "marker": "D"}
  ],
  "baselines": [
    {"id": "model_x", "label": "Model X", "tick_label": "Model\nX", "color": "#D89C21", "marker": "D"}
  ],
  "figure_note": "Open markers denote provisional points."
}
```

`color`, `marker`, and `linestyle` are optional. Omitted styles use deterministic
colorblind-safe defaults and are recorded in the receipt. `y_limits` is optional;
automatic limits include every point and supplied uncertainty bound.

## Extension rules

- New curve: add one `curves` entry and exactly one row for every panel/stage.
- New baseline: add one `baselines` entry and exactly one row for every panel.
- New panel: add one `panels` entry and complete rows for every declared series.
- New subject: change panels, stages, labels, metric, and data; keep the schema.
- Missing results stay missing; do not encode them as zero. Split the figure or
  finish the comparison rather than silently dropping required combinations.
- Export the numeric source beside every durable figure and retain provenance.
