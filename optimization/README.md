# Optimization

`selection/` owns candidate pools, objective optimization and JSON grid execution.
`semantic_buckets/` owns bucket construction, refinement, sliding utility scoring,
physical membership ledgers and explicit publication. `dili_score/` is the unchanged
standalone deterministic record scorer. `bioavailability_score/` adds five-tier
Oral relevance and completeness scoring. `record_scoring.py` owns the shared BBB,
Ames and Carcinogens implementation. All five tasks share the offline cache command.

See [selection](selection/README.md) for direct/indirect policies and prepared L1 card
caches, and [semantic buckets](semantic_buckets/README.md) for construction and scoring.
Experiment settings are Bash recipes; Python owns scientific algorithms, validation,
execution and artifact handling. Prediction/evaluation remains in `predict/`.

## Offline BBB, Ames and Carcinogens scores

```python
from optimization.record_scoring import score_record

result = score_record({
    "endpoint": "bacterial reverse mutation",
    "biological_system": "Salmonella TA98, with S9",
    "reported_result": "negative",
    "support_text": "The source reports this assay result.",
}, task="ames")
assert result["score"] == 1.0
```

The API accepts `bbb_martins`, `ames`, and `carcinogens`. Rules are separately
versioned as `<task>_relevance_completeness.v2`. The score is relevance multiplied
by completeness; it measures information, not a molecule's outcome probability,
experimental reliability, or transferability. No datasets or services are loaded.

| Task | Direct, 1 | Strong proxy, 0.75 | Ambiguous/proxy, 0.5 | Weak proxy, 0.25 |
| --- | --- | --- | --- | --- |
| BBB | Penetration conclusions and brain–blood distribution | Identified barrier models/transport, brain or CSF exposure | Transport/exposure with missing site | Remote transport, descriptors, barrier-bypassing brain dosing |
| Ames | Identified bacterial reverse mutation/Ames outcomes | Other gene-mutation assays | DNA damage, chromosome damage, unspecified mutagenicity | Generic redox, metabolic and survival mechanisms |
| Carcinogens | Exposure-related carcinogenicity, cancer incidence and classifications | Transformation and identified genotoxic mechanisms | Growth/regulatory mechanisms or uncertain tumor context | Generic redox, reactivity and survival |

Explicit unrelated endpoints receive 0. Unfamiliar endpoints remain 0.5 with an
`unrecognized_endpoint` flag. Positive and negative outcomes have equal relevance.
Cancer names alone do not establish carcinogenicity. Therapeutic tumor responses
and implanted-tumor growth do not count as direct carcinogenicity; xenograft
formation by transformed cells is a transformation proxy. Predicted results,
uncertain task scope, and inseparable mixed endpoints are capped at 0.5.
Independent endpoint fields are considered separately. Source IDs, molecule names,
supporting prose and benchmark labels cannot confer relevance.
Unrecognized aliases retain diagnostic flags but cannot promote an identified
positive-relevance endpoint to the ambiguous fallback tier.

V2 reads structured transporter identifiers and requires a named BBB transporter
before interpreting a generic potency endpoint as transport evidence. Explicitly
distant potency assays remain weak proxies. Negated/assumed BBB models cannot establish assay scope.
Target descriptions cannot supply an anatomical site or experimental assay model.
Ames distinguishes nonbacterial reverse-mutation and protective anti-mutagenicity
readouts; carcinogenicity requires more than a tumor name in a generic cohort.
Rule IDs distinguish CSF exposure, cell transformation and genotoxicity proxies.
An unqualified endpoint alias cannot erase explicit predicted or unmeasured status.
Negative outcomes remain relevant; negating a prediction does not imply a prediction.
Named prediction software and predictive models remain capped, while an experimental
assay described as predictive (such as measured PAMPA) is not automatically computational.
These remain five-tier rules, not calibrated probabilities.

Completeness is one quarter each for a structured endpoint, structured result,
support passage and assay/system information. Native task conclusions can identify
their endpoint even without a numeric value. Numeric zero is present information;
route/dose alone is not assay information. The existing DILI result-content screen
distinguishes `reported`, `unresolved`, `description_only`, and `missing`.
Unresolved result prose retains presence credit; a high score does not certify it.

Native and canonical mappings, `raw_record_json`/`reviewed_record_json`, and prepared
`input.source_fields` wrappers are supported. Explicit reviewed aliases, including
nulls, replace stale canonical equivalents. Malformed wrappers raise an error and
inputs are never mutated. Outputs include components, matched rules and fields,
reasons, ambiguity flags, result status, and missing information.

```sh
PYTHON=python bash experiments/record_scoring/tasks.sh bbb_martins
PYTHON=python bash experiments/record_scoring/tasks.sh ames
PYTHON=python bash experiments/record_scoring/tasks.sh carcinogens
```

Each recipe uses the named `scaffold_disjoint_parent100_v2` cache set, checks its
evidence projection against CURRENT and reads the exact pinned canonical table.
Override `CACHE_SET`, `PYTHON`, or `OUTPUT_ROOT` explicitly. The shared cache CLI
also accepts `--release-index`, `--canonical-records`, and `--output-root`.
New Gold/TDC cache builds target `data/caches/record_scores/<task>/v2/`.
They cover unique validation/test indirect candidate UIDs at all available levels,
bind inputs and scorer dependencies, retain diagnostic columns, and refuse
populated outputs. DILI caches retain their existing contract.
Published v1 payloads remain frozen; the current reader rejects their older rules
and fingerprints. Use a fresh v2 build for the revised scoring rules. DILI is unchanged.

These three scorers remain offline. Building caches does not activate them in
selection or change evidence, semantic ledgers, predictions or query priors.
BBB eligibility is separate from these information scores: prediction-only L2+
records are excluded by default in optimization and inference, even if their
offline relevance/completeness score is positive. Published score caches remain frozen.
Regex inventory coverage and sampled record review are not independent accuracy
validation; unknown terminology and contextual interpretation remain limitations.
