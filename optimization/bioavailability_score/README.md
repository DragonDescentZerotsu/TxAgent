# Bioavailability record score

A deterministic information score for an evidence record:

`score = relevance × completeness`

It measures neither the molecule's probability of good bioavailability nor
experimental reliability or transferability to another molecule. Rules are
`bioavailability_relevance_completeness.v2`. Python 3.10+ and the TxAgent checkout
are sufficient; the API loads no datasets and makes no model or service calls.
The existing standalone DILI wheel is unchanged.

```python
from optimization.bioavailability_score import score_bioavailability_record

result = score_bioavailability_record({
    "endpoint": "intestinal permeability",
    "biological_system": "Caco-2 monolayer",
    "reported_result": "Papp = 12 × 10^-6 cm/s",
    "support_text": "The source reports this permeability measurement.",
})
assert result["components"] == {"relevance": 0.75, "completeness": 1.0}
```

## Relevance rules

| Tier | Meaning | Examples |
| --- | --- | --- |
| 1 | Explicit direct oral bioavailability | Oral F; absolute/relative/corrected bioavailability with oral context |
| 0.75 | Identified mechanistic or oral-exposure proxy | Intestinal absorption/transport, hepatic clearance, GI dissolution, oral AUC |
| 0.5 | Incomplete context or ambiguous evidence | Generic PAMPA, solubility in water, unspecified metabolic site/PK route, unfamiliar endpoints |
| 0.25 | Explicit weak or distant proxy | logP/logD/pKa, protein binding, IV-only exposure, off-site transporter activity |
| 0 | Explicitly unrelated readout | Nephrotoxicity, tumor response, tissue-specific delivery without an oral ADME readout |

Endpoint and system/route patterns are separate. A transporter identity does not
establish its anatomical site. Caco-2/MDCK/PAMPA umbrella labels do not establish
which assay was actually used. Negated, presumed or unestablished tissue context
cannot confer scope. Predicted bioavailability is capped at 0.5. A mixed endpoint
that inseparably combines relevant and unrelated readouts is capped at 0.5;
a separately identified relevant endpoint is not vetoed by another field.
Anatomical terms in patient populations do not identify a tested organ. IV
reference parameters and co-agent conditions do not establish the tested
molecule's administration route. Positive and negative outcomes have the same relevance. Supporting paragraphs
and source IDs cannot independently supply relevance. Every result includes
`matched_rules` with rule IDs, endpoint fields, context fields and pattern families,
plus `ambiguity_flags` and `reasons`.

V2 distinguishes optical absorption, dissolved oxygen and environmental/radiation
exposure from ADME endpoints. It distinguishes a negative bioavailability outcome
from an absent measurement, and a negated prediction from a predicted observation.
Unqualified endpoint aliases cannot erase explicit predicted/unmeasured status;
unknown aliases cannot promote an identified weak proxy. Intestinal and hepatic
disposition have separate rule IDs within the same 0.75 tier. Supporting passages
still supply presence credit only; these rules do not verify passage entailment.

The patterns cover the five source schemas (oral exposure, Fa, Fg, Fh and direct
oral bioavailability), their 219 reviewed endpoint mappings and the current 121
canonical endpoint concepts. This is inventory coverage, not independent accuracy
validation or an exhaustive biological ontology. New or unfamiliar terms remain
explicitly ambiguous. Regex precedence and context remain meaningful limitations.

## Input and completeness

Pass a mapping of native/canonical fields, a row containing `raw_record_json` and
optional `reviewed_record_json` objects/JSON strings, or a prepared
`{"input": {"source_fields": {...}, "support_text": "..."}}` object. Explicit source
and reviewed aliases override stale canonical equivalents, including nulls.
Invalid wrappers or JSON raise an error. The input is never modified.

The field groups in `scoring.py` define accepted aliases, including
`parameter_value`, `measured_value`, `reported_value`, `oral_bioavailability_value`,
and canonical measurement fields. Bare ambiguous `F` is not a direct-endpoint alias.

Completeness is one quarter each for a structured endpoint, structured result,
source passage and assay/system information. Dose/route-only study context does
not satisfy the assay/system check. Prose does not fill missing structured fields.
Zero is a valid reported value. The existing DILI result-content screen is reused,
with its source files included in cache provenance. Its four statuses are:

- `reported`: recognized numerical/qualitative outcome or categorical call;
- `unresolved`: unfamiliar result prose retains presence credit;
- `description_only`: methods/placeholders/figure-only pointers receive no result credit;
- `missing`: no usable structured result.

A complete record with unresolved prose can therefore still receive a high
information score. Inspect the flags; the scorer does not certify source truth.
No query-condition matching or benchmark labels enter this version.

## Offline caches

```sh
PYTHON=python bash experiments/record_scoring/bioavailability.sh
```

The recipe resolves Oral CURRENT once, uses the named
`scaffold_disjoint_parent100_v2` catalog, and writes separate Gold/TDC caches under
`data/caches/record_scores/bioavailability_ma/v2/`. Override `PYTHON`, `CACHE_SET`
and `OUTPUT_ROOT` explicitly when needed. Existing populated outputs are refused.

The shared `python -m optimization.selection.record_scores` command also accepts
explicit `--release-index`, `--canonical-records`, and `--output-root` inputs.
It infers the task, reads available indirect levels (currently L2–L6), unions
validation/test physical candidate UIDs, and verifies complete unique coverage.
The canonical-record digest must match the retrieval projection's pinned input.

Each new cache contains `scores.parquet` and a complete `manifest.json`, binding
the score rules and their dependency files, retrieval index, canonical records
and payload. Rows include components, result status, rule IDs, ambiguity flags,
source identity and missing information. Unknown source commits stay empty;
source-file digests identify the exact rules. Existing DILI caches still load.
Published v1 caches are unchanged and are rejected by the current reader because
their scientific rules differ. New rules require a fresh v2 output directory.

These are offline derived artifacts. The recipe does not activate them in context
optimization or change retrieval, evidence, selections, priors or predictions.
