# Canonical assay-transfer contract

The policy for all six tasks is declared in
[`assay_transfer/canonical.json`](../assay_transfer/canonical.json). Shared semantics
live in `common`; each task explicitly initializes its numeric settings and the
scientific policies that differ. The standard-library loader in
[`assay_transfer/canonical.py`](../assay_transfer/canonical.py) validates the full
contract and returns a content-hashed, effective task contract.

The separate record and context-molecule CLIs execute this contract through split/evaluation/filtering, sampling, augmentation, targets and dataset stages. Historical builders and datasets are archived unchanged. Every new output is a distinct successor with a pinned source snapshot and verification receipt.

## Initial task settings

| Task | Record degree cap | Context direct degree | Context indirect degree cap | Maximum variance ratio | OOD bucket records |
| --- | ---: | ---: | ---: | ---: | --- |
| BBB Martins | 96 | 96 | 96 | 0.50 | 9–11 |
| Oral bioavailability | 6 | 48 | 2 | 0.50 | 9–11 |
| Skin Reaction | 96 | 96 | 96 | 0.65 | 8–75 |
| Ames | 96 | 96 | 96 | 0.65 | Disabled |
| Carcinogens | 96 | 96 | 96 | 0.65 | Disabled |
| DILI | 96 | 48 | 96 | 0.65 | Disabled |

BBB and Oral initialize from V24.1/V25 and their compatible V10.2 context
releases. Skin initializes from V27 Skin and its V10.3 context release. The three
safety tasks initialize from V28 safety and V10.3 safety context releases.

Record caps apply separately to query and reference roles within each level.
Context direct degree is a graph parameter: it gives degree/2 references per
training query. Oral context indirect construction uses degree two over the pinned V10
record universe with the initialized V25 sampling policy before selecting its context mixture; this differs deliberately
from the standalone Oral record caps. Selected mixtures and trimming may realize
fewer pairs than a cap permits.

OOD bounds are inclusive and refer to record counts, not parent counts. For the
BBB/Oral historical initialization, size qualification occurs before conflict
filtering and is checked again after filtering. Parent support and other
eligibility checks remain required. The safety indirect artifacts have no record
ranking evaluation or OOD bucket split; their evaluation comes from direct gold
context panels. Enabling OOD for them requires an explicit evaluation policy and
evaluation levels as well as bounds.

`record_levels: null` means use every eligible level in the pinned V10 level
mapping, then record the realized levels in the build manifest. This preserves
safety levels such as Carcinogens L0/L6/L7 and DILI L6/L7 rather than imposing the
BBB level list on them.

## Shared semantics and explicit task variation

- Inputs use V10 evidence and Gold-v1; a build pins the actual manifests and file
  hashes, including level mappings, eligibility, reviewed geometry, and Skin donor routing. Molecule identity is computed with the shipped normalizer and pinned RDKit version. A version name or `CURRENT` pointer is insufficient.
- Record pairs stay inside one task, exact assay bucket and level. Direct context
  pairs use the task's gold context records. Training excludes same-parent pairs.
- The variance ratio is pooled within-parent variance divided by the unweighted
  variance of parent means. Acceptance requires a finite ratio at or below the
  task maximum. Undefined ratios are rejected.
- Numeric transfer probability is
  `sigmoid((0.4 - standardized_distance) / 0.1)` initially, with a reviewed
  identity/log10 geometry and unclipped probabilities. Record binary targets are
  fitted from final numeric training pairs within level.
- Indirect evidence enters context training only. Context validation/test use
  direct gold panels; query measurements remain hidden in prompts.
- Test values cannot fit targets. Their participation in the variance eligibility
  gate is explicitly declared per task in `variance_gate_fit`.

Holdouts, calibration fit scopes, sampling and evaluation selection are explicit
task policies. They are not inferred from the degree, ratio or OOD settings:

- BBB derives validation/test parent memberships exclusively from pinned Gold splits and selects eligible V10 records for its per-level panels. Historical query-parent lists, fixed excluded buckets, and test-to-validation copies are archived.
- Oral retains official Gold validation/test parent memberships and its bucket-based record panels.
- Skin retains Gold holdouts, donor training levels, and native L2/L3 evaluation.
- Safety retains Gold validation/test parents as gate-only evidence and fits target calibration on training records only.

Both builders apply task-wide `--split-disjointness parent|scaffold`, defaulting to scaffold for every task. Parent overlap is always forbidden between training and heldouts. Scaffold mode additionally excludes training scaffolds shared with heldouts. Different record levels never exempt an overlap. Gold validation/test membership and labels stay fixed; training exclusions are stored separately. Gold-derived record validation/test memberships and frozen OOD identities computed from V10 records define the shared holdout index. Filtering happens before final calibration and pair construction; removed training measurements cannot enter calibration or variance fitting.

Source task policies describe the initialization; the effective contract adds the selected disjointness mode and task-wide scope under `assay_transfer.effective_contract.v2`. Context effective contracts additionally store the augmentation ratio. These settings participate in the contract hash before any stage is built.

For BBB/Oral ID buckets, numeric geometry calibration uses train plus validation;
binary domain calibration uses training only. OOD calibration uses the whole
non-test bucket. Skin uses non-test train-plus-validation calibration. Direct
context calibration uses retained Gold training plus fixed validation, after exclusions and before smoke sampling. These existing
differences are declared rather than silently changed by the common interface.

## Using the contract

```bash
python -m assay_transfer.canonical
python -m assay_transfer.canonical --task bioavailability_ma
python -m unittest discover -s tests
```

```python
from assay_transfer.canonical import degree_cap, task_contract

policy = task_contract("Bioavailability_Ma")
cap = degree_cap("Bioavailability_Ma", "L4")  # 6
ratio_maximum = policy["variance_ratio_maximum"]  # 0.5
ood_minimum, ood_maximum = policy["ood_bucket_records"]  # 9, 11
```

Edit each task's `record_degree_cap`, `record_degree_caps_by_level`,
`context_direct_degree_cap`, `context_indirect_degree_cap`,
`variance_ratio_maximum` and `ood_bucket_records` to instantiate its settings.
The loader accepts canonical task names and record-level slugs, rejects unknown
settings, validates even positive degree caps and ordered OOD bounds, and never
imports a historical builder.

Each output stage must store the complete effective task contract and its
`contract_sha256` alongside input hashes. Changing a task parameter requires a
distinct output release and regeneration of dependent stages. Changing another
task's settings does not change this task's effective hash; changing shared
semantics changes every task's effective hash. No build is complete until its
split audit, artifact hashes, coverage and verification receipt are present.

Historical code, datasets, scripts, and documentation are preserved in the original source checkout's `archive/historical-20261013/`; earlier archived artifacts remain in its `archive/unused-20261013/`. These archives are not included in the TxAgent copy. No historical release was regenerated or relabeled. See the root README for pinning, staging, and verification commands.

The direct context degree must be divisible by four. Scientific policy identifiers are supported explicitly; changing degree caps, numeric target parameters, variance thresholds or OOD sizes requires a fresh output root. Context augmentation defaults to one indirect training example per direct example. `--augmentation-ratio` changes this ratio; `--no-augment` or ratio zero produces Gold-only training without evidence-library dependencies. The integer budget is the floor of ratio times the direct pair count remaining after prompt cleanup. Skin preserves L1:L2:L3 source proportions 22,767:22,767:2,100, scaling quotas to that budget with deterministic integer rounding. Full builds fail on budget shortfalls; smoke coverage reports the achieved ratio and shortfall.

BBB uses L1/L2/L3/L4; Oral uses L2/L3/L4/L6. L5 records, evaluation panels, and indirect context pairs are excluded for both tasks. Gold direct construction uses its own builder and fixed Gold evaluation cohorts.

Record evaluation uses the top 50 Morgan-ranked records for ID queries and the complete same-bucket OOD record pool, excluding the query itself. Direct context evaluation uses the top 75 training parents and expands their contexts. Oral selects up to 500 total queries per level and split in equal bucket rounds, prioritizing unused parents. Full Oral builds require the full budget and every eligible bucket; smoke builds report shortfalls. Skin record training keeps all native L2/L3 pairs and adds an equal total donor budget, split across L1/L4/EXPOSURE using bucket/label rounds. The context builder prepares its own eligible indirect training pools with context degree caps; their empirical target calibration precedes mixture selection. It shares scientific helpers with the record builder and consumes no record release or pair-stage outputs.

## Stage ownership

`split.py` defines Gold memberships and task-wide disjointness. `evaluation.py` owns IID/OOD selection and evaluation queries/rankings. `filtering.py` prepares eligible calibrated pools, including the preliminary fit needed to freeze OOD decisions and the refit after overlap filtering. `sampling/` selects raw pairs with near/far classes; `augmentation.py` chooses mixtures. `targets.py` computes sigmoid probabilities and empirical binary targets from the full sampled population. `dataset.py` renders Jinja templates, creates Parquet datasets, and explicitly uploads verified full releases to a caller-selected HF repository.
