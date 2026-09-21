# V10 unit reconciliation

## Build selection contract

All six V10 tasks resolve measurement, exact-unit, and applicable auxiliary
mappings through their task-owned `mapping_registry.v1.json`. Release builds use
the selected existing maps by default; `--use_existing` and `--use-existing` are
equivalent explicit spellings. A selected file must pass its task, schema, hash,
dependency, and review checks before the build lock is opened. Files retained
under `reference_mappings` are provenance only and are never fallback inputs.

Map generation remains a separate reviewed workflow. Unit successors use this
module, while non-unit columns use
`shared/v2/clustered_auxiliary_mapping.py` through `auxiliary-plan` and
`auxiliary-run`. A generated draft must be consolidated with decisions and a
hash-pinned review-completion receipt, then selected by a registry successor;
the evidence build itself never launches paid model work.

BBB and Oral `v10_main_universe_v2` are validation-only successors. Their
Stage-1, Stage-2, pruning, and Stage-3 scientific bytes are unchanged from v1;
the exact mappings used by v1 were promoted from node-local working paths into
durable task-owned assets before activation.

## Endpoint-aware V3 successor

AMES, DILI, and Carcinogens V3 starts from the exact endpoint/unit pairs that can
be consumed by Stage 1: deterministic rows whose measurement-resolution route is
`accept`, plus units from LLM resolution rows whose status is `ok`. Raw units used
only by `relative`, `unsure`, or `unavailable` rows are excluded. V2 remains an
immutable historical publication; V3 does not rebuild Stage 1 or update `CURRENT`.

Both V3 passes fit task-level character TF-IDF (`char_wb`, 2-5 grams) so the same
label has one vector across endpoint occurrences. Within each canonical endpoint,
proper K-means uses `k=ceil(n/50)`, seed `20260801`, `n_init=10`, and Lloyd's
algorithm. Oversized clusters are recursively split by K-means; only an
unsplittable identical-vector cluster falls back to stable lexical chunks. Each
cluster is one request and partial clusters are never combined across endpoints.
Pass B applies the same procedure to endpoint-local provisional labels from Pass A.

The model must return a verbatim `unit_span` and a concise `canonical_unit` for
every ID. It may strip analyte, assay, study, outcome, and explanatory prose while
retaining the physical unit and any reference basis. Percent outcome descriptions
such as yield, inhibition, loss, repair, viability, and depletion canonicalize to
`%`. Control, baseline, initial/total, population, denominator, ratio, fraction,
fold, and numeric-scale bases remain protected. Every published mapping uses
`scale: "1"`; retry exhaustion is an identity mapping rather than a dropped unit.

Formula subscripts and inverse-unit exponents are interpreted separately from
scale-bearing numbers. Thus `H2O2` may be removed as analyte prose without treating
its `2` characters as a magnitude change, and `mg-1` or `mg^-1` may canonicalize to
`/mg`. Scientific factors such as `10^-7`, explicit magnitudes, and population or
sample denominators remain protected.

V3.1 does not require every descriptive word in a proposed canonical unit to occur
in the source span. That lexical subset rule rejected reasonable abbreviations and
synonyms. Parsed physical scale or dimension changes, protected numeric factors,
qualifiers, and endpoint-defined bases remain guarded.

V3.2 removes the broad numeric-token equality guard because explanatory numbers in
otherwise valid unit prose caused whole-cluster rejection. Parseable physical scale
or dimension changes remain hard failures. The prompt explicitly requires percent,
per or population, ratio or fraction, inhibition, control-relative, fold, count,
and other named qualifiers to remain in both the selected unit span and canonical
unit; the deterministic qualifier and endpoint-basis guards remain active.

V3.3 states that lexical co-clustering never authorizes scale unification. Every ID
is mapped independently, and SI-prefixed units such as `mmol/L`, `µmol/L`, `nmol/L`,
and `pmol/L` (or `mM`, `µM`, `nM`, and `pM`) must remain distinct. The parsed
physical scale/dimension guard enforces this instruction.

V3.4 adds reviewed qualifier examples for percent, inhibition, control-relative,
ratio, population counts, fold, and per-denominator units. Standalone `x` beside a
numeric factor is multiplication rather than fold semantics, while slash and inverse
exponent forms such as `/mg`, `mg-1`, and `mg^-1` all express a per-denominator.

V3.5 treats words such as yield, inhibition, loss, repair, viability, survival,
incidence, frequency, and depletion as measurement semantics when attached to a
percent unit, allowing their canonical unit to be `%`. The V3 validator still
rejects removal of reference bases, population or sample denominators, ratios,
folds, counts, and numeric or scientific-notation scales. Raw source units and
record context remain unchanged.

V3 preflights the authorized local DeepSeek endpoints `dgx020:50002`,
`dgx005:50001`, and `dgx011:50001`. Each selected endpoint permits 256 in-flight
requests. Requests use `reasoning_effort=high`, a 262,144-token ceiling, and three
cross-endpoint attempts. A selected endpoint is quarantined after three consecutive
transport or server failures, but a long read timeout fails only that request and
does not permanently quarantine a service that remains reachable. No endpoint is
added after the pass begins. Recovery may seed only exact cluster-compatible
`resolved` terminal events by hash; identity fallbacks are always requeued.

An active local run may snapshot only its complete terminal identity events into a
separate immutable prefix-hashed artifact and retry them independently with
`gpt-5.4-mini`. This OpenAI fallback loads `OPENAI_API_KEY_ONE` through the shared
LLM client, uses high reasoning and the model's 128,000 completion-token maximum,
and never writes into the active local terminal log.

Publication keys are `(task, canonical_endpoint, input_unit)`. Endpoint-specific
differences are retained; identical input-to-target rules may coalesce only by
listing all endpoints that share the rule. Endpoint-local lexical conflicts require
complete, hash-pinned task-agent decisions. Conflict packets, decisions, completion
receipts, manifests, and the exact mapping are preserved in the immutable V3 bundle.
Canonical labels longer than 80 characters are reported but do not block
publication.

```bash
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation prepare-v3 \
  --run-root /local/$USER/unit-reconciliation/<run-id>
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation run-v3 \
  --run-root /local/$USER/unit-reconciliation/<run-id> --pass a
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation prepare-v3-pass-b \
  --run-root /local/$USER/unit-reconciliation/<run-id>
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation run-v3 \
  --run-root /local/$USER/unit-reconciliation/<run-id> --pass b
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation prepare-v3-conflicts \
  --run-root /vast/path/to/closed-run --output-dir /vast/path/to/review
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation consolidate-v3 \
  --task ames --run-root /vast/path/to/closed-run --review-dir /vast/path/to/review
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation validate-v3 \
  --task ames --run-root /vast/path/to/closed-run
```

## Historical V2 protocol

AMES, DILI, and Carcinogens use the lossless single-pass protocol implemented by
`canonical_reconciliation.py`. The output is an exact unit map; it does not
modify Stage 1 or implement Carcinogens Stage 2.

The frozen input universe is the union of every nonempty raw `unit_text`, every
deterministic `measurement_resolution_exact_unit`, and every unit emitted by an
`ok` Stage-1 LLM measurement assignment. The canonical script relates units
through the task measurement module's explicit
`UNIT_RECONCILIATION_CONTEXT_FIELDS`, canonical endpoint, and source. Each unit
is anchored to its rarest declared context and deterministic groups of at most
50 are sent through one LLM pass. No context column is inferred.

The pass preserves magnitude and scientific notation. Typography aliases such
as `uM` and `µM` may be reconciled, but `nM`, `µM`, `mM`, and
`10^-6 mol/L` remain distinct. Percent, control-relative, inhibition, ratio,
fold, count, and denominator qualifiers are preserved. Every published rule is
an `action: map` rule with `scale: "1"`. An uncertain unit remains its own
canonical unit; no unit is excluded.

Each partition uses this exact retry sequence with `reasoning_effort=high`:

1. `gpt-5.4-mini` through `OPENAI_API_KEY_TWO`, 262,144 output tokens;
2. the same OpenAI model with validation feedback, 262,144 output tokens;
3. local `deepseek-ai/DeepSeek-V4-Flash-0731`, 262,144 output tokens.

OpenAI work is capped at 256 concurrent requests. A prior terminal cache may be
supplied with `--seed-terminal-cache`; only exact input-compatible `resolved`
partitions are inherited. Retry-exhausted identity partitions are deliberately
retried. Transient OpenAI rate limits back off, while `insufficient_quota` opens
the OpenAI circuit and routes subsequent attempts to the local fallback.

If all three attempts fail structurally, the entire partition is identity-mapped
with `identity_after_retry_exhaustion`. Partitions are never recursively split.
The append-only terminal cache, pass mappings, manifests, and final publication
manifest retain request routing, token usage, retry outcomes, inherited cache
hashes, and coverage counts. A subsequent Codex review covers every pre-review
canonical label. It may add only guarded, task-local aliases to existing labels;
all other proposals remain identity mappings.

Context-graph batching is unit-only. Non-unit canonical columns use the existing
single-pass embedding implementation in
`shared/v2/clustered_auxiliary_mapping.py`, exposed through the canonical CLI's
`auxiliary-plan` and `auxiliary-run` commands.

High-churn work should be staged on node-local storage. Once its writer closes,
copy the closed run tree to `outputs/analysis/evidence_library/unit_reconciliation/`
on `/vast` and validate it. Final task-owned bundles live under
`data_processing/canonicalization_v10/unit_reconciliation_v2/` and contain
`mapping.json`, `manifest.json`, `agent_review.decisions.jsonl`, and `report.md`.
A future task sequence is:

```bash
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation prepare \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id>
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation run \
  --task ames --run-root /local/$USER/unit-reconciliation/<run-id> \
  --workers 512
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation consolidate-draft \
  --task ames --run-root /vast/path/to/closed-run \
  --output /vast/path/to/review-draft/mapping.json
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation prepare-review \
  --task ames --mapping /vast/path/to/review-draft/mapping.json \
  --output-dir /vast/path/to/review-packets
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation consolidate \
  --task ames --run-root /vast/path/to/closed-run \
  --output /vast/path/to/reviewed-bundle/mapping.json \
  --decisions /local/path/agent_review.decisions.jsonl \
  --review-completion /local/path/review_completion.json
python -m data.processing.evidence_library.versions.v10.canonical_reconciliation validate \
  --task ames --run-root /vast/path/to/closed-run \
  --mapping /vast/path/to/reviewed-bundle/mapping.json
```

`consolidate-draft` can write only to an explicit nonpublication path. The
publication command requires both the decisions and their hash-pinned completion
receipt, and normal validation rejects draft bundles. Registry-selected unit and
auxiliary maps likewise require either their exact frozen predecessor hash or a
reviewed successor receipt; changing the selected bytes cannot bypass review.

The 2026-09-18 AMES, DILI, and Carcinogens publication deliberately reuses each
completed historical Pass A as the single provisional pass. Those batches were
lexically ordered rather than context-grouped; manifests retain that limitation.
Partial Pass B output is not a publication input.

| Task | Input units | Reviewed aliases | Final canonical labels |
|---|---:|---:|---:|
| AMES | 69,818 | 1,878 | 67,362 |
| DILI | 17,187 | 651 | 15,910 |
| Carcinogens | 326,038 | 4,833 | 320,224 |

## Skin Reaction successor

Skin Reaction uses its task-owned reviewed successor at
`data/caches/evidence_library/skin_reaction/v10_main_universe_v4/unit_reconciliation/`.
Its inventory is the endpoint-independent union of source and successful LLM
units required by the active main-tree source universe, plus one raw replay unit
that is absent after Stage-1 deduplication. The exact map contains 6,322 input
labels: 6,259 map rules and 63 reviewed non-unit exclusions. All rules use the
wildcard endpoint. The final `canonical_unit` cardinality is 3,134.

The final manual review covers every pre-merge canonical label and contains 163
accepted conservative merge decisions, reducing 3,359 labels to 3,136. A
deterministic denominator guard then produces 3,134 final labels and validates
189 explicit population-denominator rules with exact `1/N` scale. Exponents such
as `h^(1/2)` are not interpreted as population denominators. The release does not
append the endpoint-aware V3 rules.
