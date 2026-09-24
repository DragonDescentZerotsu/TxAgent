# DILI record score

A small, offline Python API for **DILI relevance × structured information
completeness**. Give it one record and receive a score, component scores, missing
fields, and result-content flags. Runtime dependencies: Python 3.10+ only.
No model, API key, RDKit, database, or downloaded evidence library is needed.

Package: `txagent-dili-score` 0.1.0. Scoring rules:
`dili_relevance_completeness.v5`. Package versions and scientific-rule versions
are separate; this release packages v5 without changing valid-record scores.

## Install and call

In an existing TxAgent checkout:

```sh
python -m pip install .
```

This installs only `dili_score`; it does not install the experiment framework or
include datasets in the wheel. For a lightweight checkout that avoids the large
data files:

```sh
git clone --filter=blob:none --sparse https://github.com/DragonDescentZerotsu/TxAgent.git
cd TxAgent
git sparse-checkout set dili_score
python -m pip install .
```

Repository access is required because TxAgent is private. Use your normal GitHub
credentials; do not put tokens in scripts. An existing checkout can also build a
small distributable wheel with `python -m pip wheel --no-deps . -w dist`.

```python
from dili_score import score_dili_record

record = {
    "endpoint": "ROS production",
    "biological_system": "human hepatocytes",
    "reported_result": "ROS decreased versus control",
    "support_text": "Treatment decreased ROS in human hepatocytes.",
}

result = score_dili_record(record)
print(result["score"])                  # 1.0
print(result["components"])             # {'relevance': 1.0, 'completeness': 1.0}
print(result["result_content_status"])  # reported
```

If you only need a number, use `score_dili_record(record)["score"]`. For multiple
records, use `[score_dili_record(record) for record in records]`. The function
does not mutate its input and returns a JSON-serializable dictionary.

## Input contract

Pass a Python mapping, typically the object returned by `json.loads`.
Ordinary field values are strings, numbers, booleans or `None`. There is no
required identifier, SMILES, source, family, level or benchmark label.

| Information | Simple field | Also accepted |
| --- | --- | --- |
| Endpoint | `endpoint` | `endpoint_name`, `endpoint_category`, `canonical_endpoint_name`, task-native endpoint fields |
| Result | `reported_result` | `result_value`, `quantitative_value`, `reported_value`, `effect_direction`, `canonical_measurement_text` |
| Model or assay | `biological_system` | `biological_model_context`, `experimental_system`, `assay_method`, canonical assay/species fields |
| Supporting passage | `support_text` | Supply the actual source passage when available |

The complete accepted aliases live in [_schema.py](_schema.py) and the explicit
result/system field lists in [scoring.py](scoring.py). Unknown field names are not
automatically mapped: for example, rename your own `result` key to
`reported_result`. Missing fields reduce completeness; an empty mapping scores 0.

The API also accepts:

- TxAgent canonical records, including `key=value` fragments in canonical fields.
- Native records inside `raw_record_json` and optional `reviewed_record_json`,
  each supplied as an object or JSON string. Reviewed fields, including explicit
  null corrections, take precedence. Original records are preferred.
- Prepared records shaped as
  `{"input": {"source_fields": {...}, "support_text": "..."}}`. Fields already
  omitted by an old preparation cache cannot be recovered.

A non-mapping input or malformed prepared wrapper raises `TypeError`. Invalid
JSON raises a JSON decoding error; malformed conditions raise `ValueError`.

## Output and interpretation

`score = relevance * completeness`, in the range 0–1.

- Relevance: explicit liver endpoints or recognized mechanisms in hepatic
  systems receive 1; unknown relevance receives 0.5; explicit off-task readouts
  receive 0. Mixed scopes and plant systems are limited.
- Completeness: one quarter each for structured endpoint, result, support,
  and assay/system information. It does **not** require complete dose, duration,
  controls, replication or numerical magnitude.

Inspect `completeness_checks`, `missing_fields`, `completeness_evidence` and
`reasons` when explaining a score. `result_content_status` has four values:

| Status | Meaning | Result credit |
| --- | --- | --- |
| `reported` | A rule recognized a numeric/qualitative outcome or structured call | Yes |
| `unresolved` | Unfamiliar result prose is present | Yes, presence only |
| `description_only` | Only recognized methods, placeholders or result pointers remain | No |
| `missing` | No usable structured result field | No |

`excluded_result_descriptions` and `unresolved_result_descriptions` preserve the
corresponding text. A high score with `unresolved` does not establish that an
outcome was reported. Even `reported` does not verify the truth or attribution
of a source statement. When independent fields differ, a recognized reported
result takes precedence in the aggregate status; per-field flags remain visible.

An optional condition is reported separately and never changes the main score:

```python
result = score_dili_record(
    {**record, "age": 71},
    condition="age_group=older_than_65",
)
assert result["condition"]["status"] == "matched"
```

Supported condition axes are `age_group`, `exposure`, `population_context`,
`regimen`, `genotype`, and `co_treatment`; join multiple axes with `+`.
They use exact source values and bounded age/exposure rules, not a general
medical ontology. Missing information stays unknown. Positive, negative and
protective effects do not change relevance simply because their directions differ.

## Limits and maintenance

This is a coarse screening/ordering aid, **not experimental reliability, a DILI
probability, or transferability to another molecule**. Supporting prose does not
fill missing structured endpoint/result fields. Many complete records tie;
preserve the retrieval order for ties.

The v5 field audits repaired known result-screening errors, but all review rounds
also informed edits, so they are not independent accuracy estimates. A known
boundary remains: peripheral readouts in a hepatic-disease model, such as serum
ET-1 in hepatopulmonary syndrome, can receive overly strong hepatic relevance.

There is one scoring implementation in [scoring.py](scoring.py), with private
field adapters and optional condition rules beside it. Change scientific rules
only with a rule-version bump and regression evidence. Packaging-only changes
must preserve the full response for valid existing records. The release's parity
checks are recorded in [validation.json](validation.json).

Portable tests use synthetic/minimal examples and do not load project data:

```sh
git sparse-checkout add tests/dili_score
python -m pip install '.[test]'
python -m pytest tests/dili_score -q
```

The release was also installed and smoke-tested in a fresh Python environment,
with the checkout excluded from the import path.
