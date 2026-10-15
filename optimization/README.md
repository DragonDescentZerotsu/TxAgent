# Optimization

`selection/` owns candidate pools, objective optimization and JSON grid execution.
`semantic_buckets/` owns bucket construction, refinement, sliding utility scoring,
physical membership ledgers and explicit publication. `dili_score/` is the unchanged
standalone deterministic record scorer. `bioavailability_score/` adds task-specific
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
versioned as `<task>_relevance_completeness.v3`. The score is relevance multiplied
by completeness; it measures information, not a molecule's outcome probability,
experimental reliability, or transferability. No datasets or services are loaded.

The number of levels follows distinct evidence relationships in the current
endpoint inventories. Species, outcome polarity and assay branding do not create
extra levels. Tier names are defined by `RELEVANCE_TIERS` in each scorer.

### BBB

| Relevance | Evidence relationship |
| ---: | --- |
| 1 | Direct BBB penetration conclusion or brain–blood distribution measurement |
| 0.85 | Compound permeability in an identified BBB model |
| 0.70 | Brain exposure without a blood comparator, or identified barrier transport mechanisms |
| 0.60 | CSF exposure, a different compartment from brain tissue |
| 0.50 | Missing/ambiguous site, target or endpoint; prediction/uncertainty cap |
| 0.40 | Brain barrier integrity or leakage markers rather than the compound's penetration |
| 0.25 | Distant transport systems, descriptors, or brain dosing that bypasses the barrier |
| 0 | Explicitly unrelated endpoint |

BBB rates, brain exposure and intra-brain distribution are distinct measurements;
our choice to group direct brain–blood measurements at 1 and grade proxies separately
is a task utility policy. See the experimental [brain disposition study](https://pubmed.ncbi.nlm.nih.gov/24623476/).

### Oral

| Relevance | Evidence relationship |
| ---: | --- |
| 1 | Direct oral bioavailability F |
| 0.90 | Fa/Fg/Fh component fraction or explicit hepatic/gut extraction ratio |
| 0.80 | Intestinal absorption/permeability, identified first-pass disposition, or oral AUC |
| 0.70 | Other oral concentration measurements, including Cmax |
| 0.60 | GI dissolution, solubility or stability in an identified GI system/medium |
| 0.50 | Unknown route/site, unfamiliar endpoint, or prediction/uncertainty cap |
| 0.40 | Absorption timing/rate, including Tmax; AUC ending at Tmax remains exposure |
| 0.25 | Descriptors, remote assays or explicit nonoral exposure |
| 0 | Explicitly unrelated endpoint |

Component fractions identify quantities in the oral availability decomposition;
plain metabolic activity does not become a measured component fraction. The
[oral absorption model](https://pmc.ncbi.nlm.nih.gov/articles/PMC2691459/) separates
Fa, Fg and Fh. Our weights for component fractions, exposure and mechanistic proxies
are heuristic, not values prescribed by that study.

### Ames

| Relevance | Evidence relationship |
| ---: | --- |
| 1 | Bacterial reverse mutation / Ames |
| 0.85 | Other identified bacterial gene-mutation assays |
| 0.75 | Gene mutation in other organisms/systems |
| 0.60 | Physical DNA lesions/adducts or strand-break measurements |
| 0.50 | Chromosome-level damage, unspecified mutation/genotoxicity, or ambiguity |
| 0.35 | Damage-response, repair, or reporter/pathway assays |
| 0.25 | Generic stress, metabolism, survival or protective modulation |
| 0 | Explicitly unrelated endpoint |

The assay distinctions follow [OECD 471](https://www.oecd.org/en/publications/test-no-471-bacterial-reverse-mutation-test_9789264071247-en.html),
[OECD 476](https://www.oecd.org/en/publications/test-no-476-in-vitro-mammalian-cell-gene-mutation-tests-using-the-hprt-and-xprt-genes_9789264243088-en.html)
and [OECD 487](https://www.oecd.org/en/publications/test-no-487-in-vitro-mammalian-cell-micronucleus-test_9789264264861-en.html).
Ordering adjacent proxies is our utility hypothesis. Mammalian S9 activation never
changes a bacterial assay into a mammalian assay; repair-reporter reversion is not
promoted to a fixed mutation assay.

### Carcinogens

| Relevance | Evidence relationship |
| ---: | --- |
| 1 | Exposure-related cancer outcomes, carcinogenicity bioassays or classifications |
| 0.85 | Transformation, initiation/promotion or tumor formation by transformed cells |
| 0.75 | Fixed gene mutation / bacterial reverse mutation |
| 0.65 | Physical DNA or chromosome damage |
| 0.50 | Uncertain tumor context or unspecified genotoxic mechanism |
| 0.40 | Growth, epigenetic, inflammatory, repair or damage-response regulation |
| 0.25 | Generic redox, reactivity, cytotoxicity or protective modulation |
| 0 | Unrelated endpoints, including therapeutic tumor regression |

Long-term neoplastic outcomes and cell transformation are distinct assays in
[OECD 451](https://www.oecd.org/en/publications/test-no-451-carcinogenicity-studies_9789264071186-en.html)
and the [OECD cell-transformation review](https://www.oecd.org/en/publications/detailed-review-paper-on-cell-transformation-assays-for-detection-of-chemical-carcinogens_8b8ef5ba-en.html).
The numerical separation of transformation, mutation, damage and downstream
responses is a heuristic interpretation of their proximity to the task outcome.

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

The scorer reads structured transporter identifiers and requires a named BBB transporter
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
The v3 rubrics below use task-specific tiers. Numeric values are heuristic utility weights, not calibrated probabilities or externally validated distances between assays.

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
New Gold/TDC cache builds target `data/caches/record_scores/<task>/v3/`.
They cover unique validation/test indirect candidate UIDs at all available levels,
bind inputs and scorer dependencies, retain diagnostic columns, and refuse
populated outputs. DILI caches retain their existing contract.
Published v1/v2 payloads and reports remain frozen; the current reader rejects their older rules
and fingerprints. Use a fresh v3 build for the revised scoring rules. DILI is unchanged.

These three scorers remain offline. Building caches does not activate them in
selection or change evidence, semantic ledgers, predictions or query priors.
BBB eligibility is separate from these information scores: prediction-only L2+
records are excluded by default in optimization and inference, even if their
offline relevance/completeness score is positive. Published score caches remain frozen.
Regex inventory coverage and sampled record review are not independent accuracy
validation; unknown terminology and contextual interpretation remain limitations.
