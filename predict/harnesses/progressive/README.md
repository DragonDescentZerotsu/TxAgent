# Progressive harness

The progressive harness has two explicit presentation profiles. They share the
append-only response state, JSON validation, model transport, checkpoints,
summaries, and trace writer. They do not share retrieval or card semantics.

## Standard profile

`runner.py --profile standard` is the established molecule-card experiment.
`retrieval.py`, `state.py`, `molecule_card.yaml`, and `progressive.jinja` own its
cumulative mechanism-family retrieval, bounded molecule/card deltas, analog
comparison tools, and prompt.

## Context-record profile

`runner.py --profile context_records` runs the two-level raw-record experiment
owned by `context_records/`:

1. The current `data/gold_labels/<Task>/CURRENT` release defines train and valid.
2. The frozen V9 cache selects exact ranks 0-9 from the full Morgan top-100
   training-context pool for each valid query. No structure-score threshold is
   applied and no structure score is shown to the model.
3. L1 maps each selected gold context's immutable provenance back to physical
   normalized V7 rows and samples a bounded number of records deterministically.
4. L2 appends the same bounded number of other V7 records from the task's gold
   source domain with the same normalized parent and exact `condition_group`.
5. The query property prior is reused. Query-to-reference comparison tools are
   disabled, so the prompt contains no per-reference structure score or
   structure-distance label.

The profile has two explicitly versioned front ends:

- `context_records/card_v1.yaml` and `prompt_v1.jinja` reproduce
  `progressive.assay_transfer.v1`.
- `context_records/card_v2.yaml` and `prompt_v2.jinja` define
  `progressive.assay_transfer.v2`: source-native record values only, plus
  task-specific direct-versus-indirect guidance for BBB, oral bioavailability,
  and skin sensitization.
- `context_records/prompt_v3.jinja` keeps the V2 source-native card contract and
  task guidance, but restores V1-style calibration of `transfer_likelihood` as
  one relevance signal rather than the primary transfer weight.
- `context_records/card_v4.yaml` and `prompt_v4.jinja` keep V3 retrieval and
  reasoning while showing normalized V7 record fields without source-native or
  resolved-value fallback. Verbatim `support_text` remains visible because it
  has no canonical replacement.
- The cumulative V4 ablation starts from V1: V4.1 adds only V4's relaxed
  transfer-score instruction, V4.2 adds the detailed task guidance while
  retaining V1's compact card, and V4.3 switches to V4's 29-field canonical
  card. These variants isolate one model-visible change at a time.
- V5 introduces a symmetric context-level decision procedure with V1's compact
  card. V6 uses the identical prompt with V4's canonical card, isolating the
  effect of record representation.
- V7 combines V2/V3's source-native card with V1's concise prompt. It tests the
  remaining card/prompt combination without adding new reasoning instructions.
- V8 uses a bioavailability-focused hybrid card with V3-style reasoning. V8.1
  adds only an explicit parent-condition aggregation and prior tie-break rule;
  V8.2 keeps that prompt and restores the richer source-native V2 card to
  isolate the card surface.

The YAML controls the ordered JSON fields shown for every context and record;
blank optional fields are omitted. The Jinja file controls the system prompt.
Select them with
`--assay-transfer-prompt-version v1|v2|v3|v4|v4.1|v4.2|v4.3|v5|v6|v7|v8|v8.1|v8.2`.
Each run manifest
pins the profile name, card-contract hash, and prompt-template hash.
`--record-limit-per-context-level` controls the L1 per-context bound (default
10). `--l2-record-limit-per-context` controls newly appended L2 records and
defaults to the L1 cap. `--indirect-record-limit-per-level` controls the new
records appended at each L3+ family (default 50). Sampling is nested: a smaller
cap is the exact prefix of a larger cap.
`context_records/profile.py` owns provenance mapping, sampling, card projection,
and prompt assembly. The runner only orchestrates those operations.

`retrieval_source_id` is not used to select or classify either level. In
particular, `direct_vote`, `direct_residual`, and `indirect` are V7 audit labels,
not definitions of current gold membership. This matters because active-gold
constituents can cross those V7 labels after later source-policy or structure
normalization changes.

Task source provenance is resolved as follows:

- BBB source row IDs resolve by immutable source index.
- Bioavailability canonical claim IDs expand to their physical HF/local source
  rows through the hash-checked canonical direct-claim artifact.
- Skin external reviewed rows resolve by source-local row index. Frozen
  molecule-only rows resolve by source record ID plus PMID; when an aggregate
  gold row makes that pair non-unique, the most specific gold provenance group
  is used and parent identity breaks remaining ties.

The internal preparation artifact keeps hashes and record counts for auditing.
The model sees only the fields projected by the YAML contract. Gold labels,
vote fractions, V7 source-policy tags, provenance IDs, and selection metadata
never enter the prompt.

The optional cache-backed continuation appends independently ranked V19.1 Stage
3 records per configured family (50 by default). BBB continues through L5;
Bioavailability ends at L6 with hepatic-clearance/metabolic-stability evidence;
Skin uses L3 sensitization-AOP evidence and ends at L4 with the distinct
phototoxicity/irritation/local-damage family. Skin can reuse its frozen V6 L1/L2
states when the original 10/10 caps are selected.

## Historical BBB L1-L5 mapping

The historical source-purity-v5 levels are an audit lineage, not the
context-record profile. Exact source-identity mapping from current V7 rows to
that old catalog produced:

| current V7 audit tag | historical L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| `direct_vote` | 8,536 | 5 | 0 | 0 | 0 |
| `direct_residual` | 5,478 | 254,134 | 11,014 | 20,109 | 3,552 |

Those residual rows appear across all historical levels because the old levels
were assigned by the source-purity-v5 family catalog, whereas V7's later
`retrieval_source_id` was assigned by a different direct-voting policy. The two
columns answer different questions. Do not reconstruct historical L1/L2 by
splitting on the V7 audit tag, and do not use the historical mapping to define
the new context-record levels.
