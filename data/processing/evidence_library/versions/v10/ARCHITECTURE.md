# V10 evidence-library construction order

V10 has one ordered normalization pipeline and one prebuilt-mapping registry per
task. The registry is the sole inventory of mapping assets for that task: it
records the stage, source coverage, output fields, dependencies, path, and hash
for every prebuilt mapping. Task code may implement how a mapping is applied, but
must not hide an additional unregistered mapping.

Canonical registries:

- `tasks/bbb_martins/data_processing/mapping_registry.v1.json`
- `tasks/bioavailability_ma/data_processing/mapping_registry.v1.json`

## Ordered stages

1. **Stage 0 — immutable source identity.** Load source-native rows, attach a
   stable `source_row_uid`, and record source paths, revisions, and hashes.
2. **Stage 1a — reviewed source repair.** Apply reviewed field repairs, reviewed
   drops, source-SMILES mappings, and name/SMILES decisions. The repaired
   `canonical_smiles` remains record-level; parent normalization is not used to
   decide whether two measurements are duplicates.
3. **Stage 1b — canonical endpoint name.** Attach `canonical_endpoint_name`
   before resolving the main measurement. This is the semantic endpoint name
   used to select and interpret the row's principal measurement.
4. **Stage 1c — canonical measurement and unit.** Apply measurement resolution,
   then exact unit reconciliation, producing `canonical_measurement_text` and
   `canonical_unit_text`. These are canonical measurements, not endpoint names
   or endpoint concepts. Before extraction, every source is validated against
   `stage1_source_roles.v1`: exactly one endpoint field or constant, one raw
   measurement field, and one unit field, constant, or reviewed exception. The
   declaration is hash-bound into the request cache and copied into the final
   mapping manifest.
5. **Stage 1d — exact physical-row deduplication.** A duplicate candidate must
   have present, literally identical repaired SMILES, PMID, canonical
   measurement, and canonical unit. Within-source comparisons additionally
   require every configured scientific field to match; cross-source comparisons
   require the shared configured scientific fields to match. Provenance and
   prose alone do not create separate measurements. Every physical UID in the
   active Gold-v1 voter contract survives, even when two protected rows are exact
   duplicates. Nonprotected duplicate groups retain the smallest UID, and every
   discarded UID points to its retained UID in the audit.
6. **Stage 2 — canonical endpoint concept and auxiliary semantics.** Attach the
   broader `canonical_endpoint_concept`, auxiliary context, and reference
   semantics after the Stage-1 record identity is frozen. Endpoint concepts are
   used for downstream pair buckets and family semantics; they are not part of
   Stage-1 exact duplicate identity.
7. **Stage 3 — pair buckets and downstream policy.** Build pair buckets,
   record-level assay-transfer calibration exclusions, and level mappings.
   Pruning is record-level calibration exclusion only; it never deletes a pair
   bucket or removes a record from canonical/retrieval libraries. Pair-bucket
   identity contains only intrinsic evidence metadata: source, canonical
   endpoint, canonical unit, and source-specific canonical assay dimensions.
   Gold membership, benchmark condition/context, split, voter identity, parent
   assignment, and values or hashes derived from them remain separate record
   metadata and must never enter the pair key or canonical pair fields.

## Failure-prevention gates

These gates apply to every task as it advances through the corresponding stage,
including future Skin, DILI, AMES, and Carcinogens builds.

| Failure seen during BBB/Oral rebuilding | Required guard |
|---|---|
| Protected rows were sourced from an evidence-library vote table rather than the pinned main source universe. | Instantiate Stage 0 from the pinned source universe first; use Gold voter membership only to mark physical UIDs for protection. |
| Historical voter SMILES or later reviewed repairs changed protected structures. | Preserve each protected UID's pinned source structure. Apply reviewed structure repairs only to nonvoters; non-structure corrections may still apply to every row. |
| Gold canonical-claim or card membership leaked into evidence deduplication. | Keep evidence rows physical and independent. Canonical claims aggregate votes only; they never join or deduplicate evidence rows. |
| Parent disagreement under a Gold card was treated as a construction error. | Parent is display/retrieval metadata, not Gold membership authority. Accept mixed-parent cards and never mutate records merely to make a card chemically uniform. |
| Final Stage 3 silently published without reviewed pruning. | Build unpruned Stage 3 only to generate candidates, freeze reviewed UID decisions, then rebuild final Stage 3 with the pruning-manifest hash. Missing or stale pruning must fail closed. |
| A refresh silently changed previously reviewed exclusions. | Treat predecessor prunes as the immutable minimum. A refresh may add prunes; reversal requires a separately reviewed successor. Save the carried decisions in reviews, decisions, and manifest—not only in a patched output table. |
| Stage 3 changed while the level map, eligibility sidecar, global index, or test pins stayed stale. | Rebuild all four from the final Stage-3 hash and validate exact UID coverage, exact physical-voter L1 membership, artifact hashes, and row counts before publication. |
| New semantic atoms were added to an already selected universe. | Publish a separate immutable semantic generation. Never modify the selected map, ranking, manifest, pair records, or running cache in place. |

Publication is one transaction: build and validate in staging, retain paid/review
caches, omit build locks and incomplete markers, archive the prior active tree,
atomically install the successor, refresh indexes, and only then change `CURRENT`.
The global level-mapping index is a locator with hashes and counts; it is not a
scientific mapping source and must never derive or rewrite levels.

## Creating mappings for a fresh dataset

A new dataset is not expected to start with prebuilt semantic maps. Build and
register them in dependency order:

1. Inventory raw endpoint strings and generate `canonical_endpoint_name` with
   the existing two-stage process: deterministic/high-confidence clustering,
   followed by reviewed semantic reconciliation for unresolved clusters.
2. Freeze the endpoint-name artifact and its receipt. Generate measurement
   candidates using those names, review ambiguous main measurements, then freeze
   the measurement-resolution and exact-unit artifacts. If a compatible reviewed
   predecessor exists, extend it in a new immutable generation and review only
   uncovered inputs; do not restart from scratch or mutate the selected map.
3. Run Stage 1 through exact deduplication.
4. Generate `canonical_endpoint_concept` with the same two-stage clustering and
   review protocol, using the frozen endpoint names as inputs. Current legacy BBB
   and Oral endpoint-concept assets are explicitly grandfathered because their
   original two-stage receipts cannot be reconstructed; new datasets may not use
   that exception.
5. Generate auxiliary/species/reference mappings, add every asset to the task's
   single mapping-registry JSON, pin hashes, and run registry validation before a
   full build.

## Active Gold-v1 boundary

Gold membership is downstream of record acquisition and identifies the physical
rows protected from Stage-1 structure repair and exact deduplication. BBB and Oral
load the frozen contract at
`data/gold_labels/<Task>/v1/scaffold/voter_membership.parquet`. Only `task` and
`source_row_uid` define that boundary: historical `canonical_smiles` and
`source_sha256` columns are audit payload, not structure authority. Each protected
UID must exist with a valid structure from the pinned main source universe. Its
structure is preserved while non-structure reviewed corrections still apply. All
frozen structure mappings and reviewed repairs apply to nonprotected rows.

`vote_units.parquet` preserves the original Gold-v1 aggregation unit and
`gold_label_record_index.parquet` stores the nested card -> vote unit -> physical
UID mapping. BBB has one physical member per vote. Oral expands each frozen
canonical claim to its physical HF/local members: all members remain separate L1
records, but the shared `vote_id` contributes exactly once to the card voter mean.
The 70% null-condition and 60% reviewed external-condition decisions are frozen
in the published v1 cards; the V10 rebuild does not reaggregate them.

Gold condition mapping may control voter membership, aggregation, retrieval
filtering, and assay-transfer eligibility. It never changes Stage-3 pair-bucket
identity; records with different Gold contexts but identical intrinsic assay
semantics share the same pair bucket.

`voter_membership.parquet` is the L1 membership authority. Parent SMILES groups
records for molecule cards, but never substitutes for physical UID membership or
changes which source row voted. Records outside published-card membership remain
retrievable at their nondirect level rather than voting at L1.

Record-level assay-transfer suitability is published separately from the level
map. It retains continuous absolute or endpoint-defined-ratio scalars and complete
controlled categorical values, subject to resolved structure/endpoint/unit and
reviewed record exclusions. Minimum support, variance, distribution, and distance
calibration are downstream whole-bucket decisions and never alter this record
mark or remove records from retrieval.

## Stage-1 source roles

| Task/source | Endpoint role | Measurement role | Unit role |
|---|---|---|---|
| Oral `oral_exposure` | `exposure_measure` | `parameter_value` | `parameter_units` |
| Oral `fa` | `endpoint_category` | `reported_value` | `reported_units` |
| Oral `fg` | `gut_wall_process` | `measured_value` | `embedded_in_measurement` exception |
| Oral `fh` | `metric_type` | `reported_value` | `reported_units` |
| Oral `hf_bioavailability` | constant `oral_bioavailability` | `oral_bioavailability_value` | `embedded_in_measurement` exception |
| AMES `mutagenicity_outcomes` | `assay_family` | `mutagenicity_result` | `unitless_categorical` exception |
| AMES `fixed_mutation` | `assay_family` | `response_value` | `response_unit` |
| AMES `premutagenic_damage` | `assay_family` | `response_value` | `response_unit` |
| AMES `mutagenicity_mechanism` | `assay_family` | `quantitative_readout_value` | `quantitative_readout_unit` |
| DILI `dili_base` | constant `human_dili_relation` | `causal_status` | `unitless_categorical` exception |
| DILI `dili_v1` | `endpoint_category` | `result_value` | `result_unit` |
| DILI `dili_v2` | `assay_category` | `reported_result` | `embedded_in_measurement` exception |
| DILI `dili_v3` | `assay_category` | `result_value` | `result_unit` |
| DILI `dili_v4` | `assay_family` | `result_value` | `result_unit` |
| DILI `dili_v5` | `endpoint_class` | `quantitative_value` | `quantitative_unit` |
| Carcinogens `carcinogens_base` | `cancer_or_tumor` | `carcinogenicity_conclusion` | `unitless_categorical` exception |
| Carcinogens `carcinogens_v1` | `assay_family` | `result_value` | `measurement_unit` |
| Carcinogens `carcinogens_v2` | `assay_family` | `response_value` | `response_unit` |
| Carcinogens `carcinogens_v3` | `endpoint_category` | `reported_result` | `measurement_unit` |
| Carcinogens `carcinogens_v4` | `assay_domain` | `endpoint_result` | `measurement_unit` |
| Carcinogens `carcinogens_v5` | `phenotype_domain` | `endpoint_result` | `endpoint_unit` |
| Skin `direct_skin_reaction` | `reaction_type` | `effect_metric` | `embedded_in_measurement` exception |
| Skin `sensitization_aop` | `endpoint_or_target` | `result_value` | `result_unit` |
| Skin `phototoxicity_irritation_local_damage` | `evidence_endpoint` | `observed_effect` | `embedded_in_measurement` exception |
| Skin `skin_exposure` | `evidence_type` | `result_value` | `result_unit` |

The exception rationale lives with each task schema. No other missing unit field
is accepted implicitly. Exact-unit behavior changes require a successor component
version; Skin's expanded percent-endpoint mapping is v3, while its published v2
asset remains byte-identical to the earlier contract.
