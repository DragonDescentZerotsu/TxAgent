# AGENTS.md: ChEMBL BBB Assay Screening Implementation Notes

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

## Env instructions

If you are on `node002`, default to the `vllm` conda environment when you need RDKit or the local project dependencies. conda is at: /data1/tianang/anaconda3/condabin/conda

## Goal

Implement a reusable script that screens assays from the ChEMBL SQLite/database to help determine whether a molecule can cross the BBB, and ranks them by evidence strength.

Current status notes:

```text
This file mainly retains early assay screening plans and rule descriptions.
The current official implementation has moved to tools/chembl_tool/tasks/bbb_martins/.
Phase 2 has implemented molecule-level evidence library, neighbor retrieval, and reasoning pipeline.
```

Current key entry points:

```text
tools/chembl_tool/tasks/bbb_martins/screen_assays.py
tools/chembl_tool/tasks/bbb_martins/rescore_outputs.py
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
```

Current role of ChEMBL neighbor retrieval:

```text
Not a callable tool for DeepSeek.
Not a current FastAPI service tool.
It is evidence prefetch / context assembly executed by run_reasoning_pipeline.py before LLM calls.
Group-level DeepSeek can only call mmp_structure_compare and properties_compare.
Single-molecule DeepSeek can only call molecule_properties.
```

Raw screening output for `screen_assays.py`:

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_candidates.csv
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_candidates.jsonl
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_assay_report.md
```

Current official BBB_Martins task outputs are archived by pipeline stage:

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/
outputs/chembl_tool/tasks/bbb_martins/evidence_library/
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/
```

Do not train models. Only perform deterministic assay retrieval, scoring, tiering, and export.

## Input

Prefer ChEMBL SQLite, for example:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.screen_assays \
  --chembl-sqlite /data1/tianang/Projects/TxAgent/tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000
```

If the current project already has ChEMBL access tools, reuse them preferentially; do not reinvent complex ORMs.

---

## ChEMBL tables to read

At minimum use:

```text
assays
activities
target_dictionary
target_components
component_sequences
component_synonyms
molecule_dictionary
compound_structures
```

Additionally recommended:

```text
docs
```

`docs.title` / `docs.abstract` are only weak recall aids and cannot alone decide to retain an assay. They are used to discover assays in BBB papers, but ultimately the assay description, activity endpoint, cell model, or target annotation must support retention.

Core fields:

```text
assays.assay_id
assays.doc_id
assays.chembl_id
assays.description
assays.assay_type
assays.assay_test_type
assays.assay_category
assays.assay_cell_type
assays.assay_tissue
assays.assay_organism
assays.confidence_score
assays.relationship_type
assays.tid

activities.assay_id
activities.standard_type
activities.standard_relation
activities.standard_value
activities.standard_units
activities.molregno
activities.pchembl_value
activities.data_validity_comment
activities.potential_duplicate
activities.activity_comment

target_dictionary.tid
target_dictionary.chembl_id
target_dictionary.pref_name
target_dictionary.target_type
target_dictionary.organism

component_sequences.component_id
component_sequences.accession
component_sequences.description
component_synonyms.component_id
component_synonyms.component_synonym
component_synonyms.syn_type

docs.doc_id
docs.title
docs.abstract
docs.pubmed_id
docs.doi
```

Gracefully fall back when fields are missing; do not crash.

---

## Output fields

`bbb_assay_candidates.csv/jsonl` must include at least:

```text
assay_chembl_id
assay_id
tier
score
assay_type
description
target_chembl_id
target_pref_name
target_genes
target_synonyms
organism
confidence_score
relationship_type
assay_cell_type
assay_tissue
n_activities
n_unique_molecules
standard_types
matched_keywords
matched_endpoints
matched_targets
negative_flags
weak_context_flags
reason
keep_for_bbb_reasoning
```

`reason` must explain in one English sentence why the assay is retained, for example:

```text
Matches brain/plasma and the logBB endpoint, so it is direct brain-exposure evidence.
```

---

## Text normalization

Normalize before all keyword matching:

```python
text = text.lower()
text = text.replace("-", " ").replace("/", " ")
text = text.replace(",", " ")
text = collapse_multiple_spaces(text)
```

Also keep the original description for output.

Be compatible with these spellings:

```text
blood-brain barrier / blood brain barrier
P-gp / P gp / P glycoprotein / P-glycoprotein
Caco-2 / Caco2
MDCK-MDR1 / MDR1-MDCK / MDCKII-MDR1 / MDCK2-MDR1
B-A/A-B / BA/AB / basolateral to apical / apical to basolateral
Kp,uu / Kpuu / Kp uu / K(p,uu,brain)
brain/plasma / brain to plasma / brain:blood / brain blood
CSF/plasma / CSF to plasma / cerebrospinal fluid plasma
```

Prefer token/phrase matching or precompiled regexes to avoid false positives from single characters or overly short tokens.

---

## Evidence tiering

### Tier 1: Direct BBB / brain exposure, highest priority

This tier indicates the molecule already has brain exposure, brain/plasma, brain/blood, CSF, or BBB penetration-related experimental readings.

Keywords:

```text
blood brain barrier
bbb permeability
brain penetration
brain uptake
brain exposure
brain plasma
brain to plasma
brain blood
brain to blood
brain concentration
brain level
brain auc
logbb
kp uu brain
kpuu brain
k p uu brain
unbound brain
unbound plasma
fraction unbound brain
fu brain
brain binding
in situ brain perfusion
brain perfusion
brain uptake index
brain penetration index
bui
bpi
permeability surface area
ps product
csf plasma
csf to plasma
cerebrospinal fluid plasma
```

Endpoints:

```text
logbb
k(p,uu,brain)
kp,uu,brain
kp uu brain
kpuu brain
brain/plasma
brain plasma
brain/plasma ratio
brain to plasma ratio
brain/blood
brain blood
brain to blood ratio
brain uptake
brain uptake index
brain penetration index
brain concentration
brain level
csf/plasma
csf plasma
csf to plasma ratio
kin
ps product
```

Base score: `100`

Note:

```text
kp
ps
kin
csf
```

These short words must not trigger Tier 1 alone. They must co-occur with brain, BBB, CSF/plasma, perfusion, unbound brain, or similar context, or the standard endpoint must be an explicit `K(p,uu,brain)` / `brain/plasma` / `logBB`.

---

### Tier 2: Passive permeability / barrier model permeability

This tier indicates the molecule has BBB-related in vitro barrier models, PAMPA-BBB, brain endothelial cell models, MDCK/Caco-2 permeability, or Papp evidence. Caco-2 is more intestinal absorption, but can serve as general permeability/P-gp auxiliary evidence.

Keywords:

```text
pampa bbb
bbb pampa
parallel artificial membrane
mdck mdr1
mdr1 mdck
mdckii mdr1
mdck2 mdr1
mdck-mdr1
mdck
mdckii
mdck2
caco 2
caco2
hcmec d3
bend 3
bmec
bbmec
rbec
brain endothelial
brain microvessel endothelial
transwell
apical to basolateral
basolateral to apical
papp
logpapp
apparent permeability
permeability coefficient
```

Endpoints:

```text
papp
logpapp
apparent permeability
permeability coefficient
caco-2 papp
caco-2 permeability
efflux ratio
```

Base score: `70`

Note:

```text
pampa
permeability
pe
a b
b a
```

These words must not trigger Tier 2 alone. `pampa` requires simultaneous BBB, PAMPA-BBB, or parallel artificial membrane; `permeability` must co-occur with Caco-2, MDCK, PAMPA, BBB, brain endothelial, transwell, Papp, or efflux ratio; `a b` / `b a` are used only in basolateral/apical or Papp contexts.

---

### Tier 3: Active efflux / efflux transporter

This tier indicates the molecule may be a BBB efflux transporter substrate, inhibitor, or have bidirectional transport / efflux ratio evidence.

Key transporters:

```text
ABCB1, MDR1, P-gp, P glycoprotein, P-glycoprotein
ABCG2, BCRP, breast cancer resistance protein
ABCC1, MRP1
ABCC2, MRP2
ABCC4, MRP4
ABCC5, MRP5
```

Keywords:

```text
p gp
p glycoprotein
pglycoprotein
mdr1
abcb1
bcrp
abcg2
breast cancer resistance protein
mrp1
mrp2
mrp4
mrp5
abcc1
abcc2
abcc4
abcc5
efflux
efflux ratio
substrate
bidirectional
basolateral to apical
apical to basolateral
rhodamine 123
calcein am
digoxin
prazosin
vinblastine
tariquidar
elacridar
verapamil
cyclosporin a
ko143
atpase
```

Endpoints:

```text
efflux ratio
substrate
transport
papp
atpase
ic50
ki
```

Base score: `85`

Note:

`P-gp/BCRP substrate`, `bidirectional Papp`, `efflux ratio` are more important than mere inhibitor `IC50`. `transport`, `substrate`, `IC50`, `Ki` must not trigger Tier 3 alone; they must co-occur with ABCB1/ABCG2/ABCC gene symbols, P-gp/BCRP/MRP targets, efflux ratio, bidirectional assays, or typical probe/inhibitor context.

---

### Tier 4: Active uptake / influx transporter

This tier indicates the molecule may enter the brain via BBB-related uptake transporters.

Key transporters:

```text
SLC7A5, LAT1
SLC2A1, GLUT1
SLCO1A2, OATP1A2
SLC22A8, OAT3
SLC16A1, MCT1
TFRC, transferrin receptor
```

Keywords:

```text
lat1
slc7a5
glut1
slc2a1
oatp1a2
slco1a2
oat3
slc22a8
mct1
slc16a1
transferrin receptor
tfrc
influx
uptake
transport
```

Endpoints:

```text
uptake
transport
substrate
```

Base score: `55`

Note:

`uptake` and `transport` are weak words and must not trigger Tier 4 alone. They must co-occur with a Tier 4 transporter target/gene/synonym, or the description must clearly be a LAT1/GLUT1/OATP1A2/OAT3/MCT1/TFRC-related assay.

---

## Scoring rules

For each assay, compute `score`:

```text
score = tier_base_score
      + keyword_bonus
      + endpoint_bonus
      + target_bonus
      + quality_bonus
      - weak_context_penalty
      - negative_penalty
```

Suggested rules:

```text
Each strong keyword +3, max +20
Key endpoint hit +15
Multiple Tier 1 direct endpoints +10
ABCB1/ABCG2 single-protein target hit +20
ABCC/SLC BBB-relevant transporter target hit +10
Transporter substrate/efflux ratio/bidirectional Papp hit +20
confidence_score >= 8 +10
relationship_type == 'D' +5
n_unique_molecules >= 20 +5
n_unique_molecules >= 100 +10
Only weak words but insufficient context -50
```

Weak words include:

```text
kp
ps
kin
csf
pampa
permeability
pe
a b
b a
transport
uptake
substrate
ic50
ki
```

Weak words must satisfy the context requirements in each tier of this file; otherwise, do not use them as retention evidence.

Negative rules:

```text
If only CNS receptor binding, without BBB/permeability/transport keywords, -80
If only cytotoxicity/cell viability/tumor proliferation, -60
If only generic kinase/receptor/enzyme inhibition, -60
If only generic uptake, e.g., thymidine/glucose/oxygen/tumor uptake, -60
If all activities are invalid or data_validity_comment is severely abnormal, -30
```

Negative keywords:

```text
cytotoxicity
cell viability
proliferation
tumor
cancer cell line
dopamine receptor
serotonin receptor
gaba receptor
muscarinic receptor
histamine receptor
opioid receptor
cannabinoid receptor
kinase inhibition
thymidine uptake
glucose uptake
oxygen uptake
tumor uptake
```

However, if strong evidence such as direct BBB, PAMPA-BBB, MDCK-MDR1, Caco-2 Papp, ABCB1, ABCG2 is also hit, do not discard solely because receptor/cell line appears.

---

## Retention rules

Default retention:

```text
score >= min_score
```

And at least one of:

```text
Hit Tier 1 direct brain/BBB exposure keyword or endpoint
Hit Tier 2 BBB-relevant permeability/cell model keyword or endpoint
Hit Tier 3 efflux transporter target/keyword/endpoint
Hit Tier 4 influx transporter target/keyword/endpoint
```

For transporter target assays:

```text
If the target is a single protein such as ABCB1/ABCG2/ABCC/SLC, prefer requiring confidence_score >= 8 or relationship_type == 'D'
If confidence_score is low, but description/endpoint clearly indicates efflux ratio, substrate, or bidirectional Papp, it can also be retained
```

For cell-based assays:

```text
Caco-2, MDCK, hCMEC/D3, bEnd.3, BMEC, BBMEC, RBEC should not be discarded solely because of low confidence_score
But must have permeability, Papp, efflux ratio, transwell, or BBB/barrier context
```

---

## Implementation structure

Current `tools/chembl_tool` is already implemented as a reusable Python package structure. Do not add a new sibling `tools/chembl_bbb/`, otherwise each future ChEMBL task will form a set of fragmented tools.

Current structure should remain: `common/` holds reusable SQLite/schema/assay aggregation logic for all tasks, `tasks/` holds specific tasks. BBB_Martins is just a task.

Current core structure:

```text
tools/chembl_tool/
  __init__.py
  common/
    __init__.py
    sqlite.py
    schema.py
    assay_loader.py
    text.py
    export.py
  tasks/
    __init__.py
    bbb_martins/
      __init__.py
      endpoint_groups.py
      build_evidence_library.py
      retrieve_neighbors.py
      run_reasoning_pipeline.py
      screen_assays.py
      rules.py
      scoring.py
      report.py
  utils/
    BBB_Martins/
      AGENTS.md
tests/
  chembl_tool/
    common/
      test_text.py
      test_schema.py
      test_assay_loader.py
    tasks/
      bbb_martins/
        test_rule_matching.py
        test_scoring.py
```

Directory responsibilities:

```text
common/sqlite.py: Connect to SQLite, execute queries, stream reads, check table existence
common/schema.py: Field existence checks, graceful fallback, ChEMBL version/path detection
common/assay_loader.py: Read and aggregate assay metadata, activity summaries, target annotations, doc annotations
common/text.py: General text normalization, phrase/token matching helpers
common/export.py: CSV/JSONL export helpers

tasks/bbb_martins/rules.py: BBB_Martins-specific rules, tier definitions, target gene lists, weak word context rules
tasks/bbb_martins/scoring.py: BBB_Martins-specific match/score/keep logic
tasks/bbb_martins/report.py: BBB_Martins reports
tasks/bbb_martins/screen_assays.py: CLI entry point, only parameter parsing, calling common loader, calling task scoring, exporting results
```

This way, when adding other ChEMBL tasks later, you only need to add:

```text
tools/chembl_tool/tasks/<task_name>/
  __init__.py
  screen_assays.py or run.py
  rules.py
  scoring.py
  report.py
```

Do not copy `assay_loader.py`, SQLite connections, target synonym aggregation, text normalization, and other common logic into new tasks.

---

## Main functions

### `common/sqlite.py`

```python
connect_sqlite(path: str) -> sqlite3.Connection
table_exists(conn, table_name: str) -> bool
get_table_columns(conn, table_name: str) -> set[str]
read_sql(conn, query: str, params: dict | None = None) -> DataFrame
```

### `common/schema.py`

```python
require_any_columns(conn, table: str, candidates: list[str]) -> list[str]
build_select_columns(conn, table: str, requested: dict[str, str | None]) -> list[str]
```

When fields do not exist, perform a unified graceful fallback here; do not scatter `PRAGMA table_info` throughout the task code.

### `common/assay_loader.py`

Responsible for reading and aggregating assay information:

```python
load_assay_metadata(conn) -> DataFrame
load_activity_summary(conn) -> DataFrame
load_target_annotations(conn) -> DataFrame
load_doc_annotations(conn) -> DataFrame
merge_assay_table(...) -> DataFrame
```

For each assay aggregation:

```text
standard_types: deduplicated list
n_activities: number of activities
n_unique_molecules: count of distinct molregno
target_genes / target_synonyms: deduplicated lists
doc_title / doc_abstract: optional weak recall fields
```

### `common/text.py`

```python
normalize_text(text: str) -> str
contains_phrase(text: str, phrase: str) -> bool
match_phrases(text: str, phrases: Iterable[str]) -> list[str]
```

### `tasks/bbb_martins/rules.py`

```python
BBB_RULES = {
    "direct_bbb": {...},
    "passive_permeability": {...},
    "efflux": {...},
    "influx": {...},
}

NEGATIVE_RULES = {...}
WEAK_CONTEXT_RULES = {...}
TRANSPORTER_TARGETS = {...}
```

### `tasks/bbb_martins/scoring.py`

```python
match_rules(row) -> MatchResult
score_assay(row, matches) -> ScoredAssay
should_keep_assay(row, scored) -> bool
```

### `tasks/bbb_martins/report.py`

Generate markdown report:

```text
total assay count
candidate assay count
counts per tier
Top 50 direct BBB assays
Top 50 passive permeability assays
Top 50 efflux assays
Top 50 influx assays
examples of weak phrase hits that were filtered
examples of negative rule filtering
```

### `tasks/bbb_martins/screen_assays.py`

CLI entry point:

```python
main(argv: list[str] | None = None) -> int
```

Responsibilities:

```text
parse arguments
connect to ChEMBL SQLite
call common.assay_loader to construct assay-level table
call tasks.bbb_martins.scoring for scoring and filtering
export CSV/JSONL/report
optionally export activity evidence for candidate assays
output scan progress according to `--progress-every`
```

---

## SQL Query Recommendations

Do not pull all activity details at once for the final table. Aggregate first:

```sql
SELECT
  assay_id,
  COUNT(*) AS n_activities,
  COUNT(DISTINCT molregno) AS n_unique_molecules,
  GROUP_CONCAT(DISTINCT LOWER(standard_type)) AS standard_types
FROM activities
WHERE standard_type IS NOT NULL
GROUP BY assay_id;
```

Assay metadata:

```sql
SELECT
  a.assay_id,
  a.chembl_id AS assay_chembl_id,
  a.description,
  a.assay_type,
  a.assay_test_type,
  a.assay_category,
  a.assay_cell_type,
  a.assay_tissue,
  a.assay_organism,
  a.confidence_score,
  a.relationship_type,
  a.tid,
  a.doc_id,
  td.chembl_id AS target_chembl_id,
  td.pref_name AS target_pref_name,
  td.target_type,
  td.organism AS target_organism
FROM assays a
LEFT JOIN target_dictionary td ON a.tid = td.tid;
```

Component/synonym information should be aggregated separately and then merged according to `tid`. Gene symbols must preferentially come from `component_synonyms.syn_type = 'GENE_SYMBOL'`; synonyms can also retain `UNIPROT` / `GENE_SYMBOL_OTHER` etc.

Doc metadata is optional:

```sql
SELECT
  doc_id,
  title,
  abstract,
  pubmed_id,
  doi
FROM docs;
```

---

## Optional: Export Activity Evidence

Add parameter:

```bash
--export-activities
```

If enabled, additionally export for candidate assays:

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw/bbb_activity_evidence.csv
```

Fields:

```text
assay_chembl_id
molecule_chembl_id
canonical_smiles
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
```

Only export activities corresponding to candidate assays to avoid overly large full-database exports.

---

## Current Implementation: Similar Molecule Evidence Retrieval

Filtering out BBB-relevant assays is only the first step. A molecule-level BBB evidence library has already been built to retrieve similar molecules with BBB-related experimental readings for a given new molecule, and use these analog evidence to assist in determining whether the query might cross the BBB.

### Evidence Library

Based on activity evidence from candidate assays, the molecule-level evidence table is currently constructed by `build_evidence_library.py`:

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_tier
assay_description
target_chembl_id
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
evidence_direction
evidence_strength
evidence_reason
```

`evidence_direction` is used to distinguish evidence meaning:

```text
supports_bbb_crossing
argues_against_bbb_crossing
efflux_risk
influx_support
context_dependent
unknown_direction
```

Do not simply mix and average different endpoints. The meanings of `logBB`, `Kp,uu,brain`, brain/plasma, CSF/plasma, Papp, efflux ratio, transporter substrate/inhibition differ; they must be interpreted separately before aggregation.

### Query-time Retrieval

Given a query molecule:

```text
1. Standardize the query molecule, generate canonical SMILES / InChIKey / Morgan fingerprint
2. Exclude the same molecule from the evidence library
3. Use Morgan fingerprint Tanimoto to retrieve similar neighbors that are not the same molecule
4. Aggregate evidence by similarity bucket, assay tier, endpoint type
5. Output analog evidence summary, not an absolute judgment directly
```

Must avoid directly retrieving the query's ground truth:

```text
exclude same molecule
exclude same full standard InChIKey
exclude same InChIKey connectivity layer (first segment of InChIKey)
exclude same canonical standardized SMILES
```

Do not exclude high-similarity analogs by default. As long as it is not the same molecule, neighbors of `Tanimoto >= 0.95` should be retained because they are usually the most valuable analog evidence.

Suggested similarity stratification:

```text
very_close_analog: Tanimoto >= 0.95, retain and mark
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

Do not use `0.40` as a default hard cutoff. Each group can return top 3 non-identical neighbors; even if similarity is very low, retain and mark the similarity bucket. The subsequent group-level reasoning LLM should judge whether these analogs are transferable. `distant_analog` and `very_distant_analog` cannot serve as strong positive or negative evidence unless there is a strong medicinal chemistry rationale for shared scaffold and assay mechanism.

If after excluding the same molecule there are no retrievable BBB evidence neighbors, explicitly return:

```text
no reliable analog evidence found
```

Do not make a hard judgment on whether the BBB can be crossed when evidence is sparse.

### Current Output and Entry Points

The `retrieve_analogs` CLI from the early plan was not adopted. The currently implemented entry point is:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors \
  --query-smiles "CCN(CC)..."
```

End-to-end reasoning entry point:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --model deepseek-v4-pro \
  --run-id <run_id>
```

Only rerun final summary:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>
```

Current reasoning run outputs:

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/retrieval.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/single_molecule_reasoning_output.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/group_reasoning_outputs.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/final_reasoning_output.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/trace_messages.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/runs/<run_id>/manifest.json
```

---

## Testing Requirements

Minimal unit tests are required:

```text
P-gp, P gp, P glycoprotein, P-glycoprotein can be normalized to efflux
Caco-2, Caco2 both can match passive permeability
blood-brain barrier, blood brain barrier both can match direct BBB
K(p,uu,brain), Kp,uu brain, Kpuu brain all can match direct BBB
MDCK-MDR1 matches both passive and efflux, but the final tier should lean toward efflux/passive high score
Only dopamine receptor binding should not be retained as BBB evidence
ABCB1 target + substrate endpoint should be retained with high score
Only generic permeability should not be retained
glucose uptake can only serve as influx evidence in the context of GLUT1/SLC2A1 target
```

Tests should not depend on the full ChEMBL database; small DataFrames or mock SQLite are sufficient.

---

## Acceptance Criteria

After completion, the following should be possible:

```bash
python -m tools.chembl_tool.tasks.bbb_martins.screen_assays \
  --chembl-sqlite /data1/tianang/Projects/TxAgent/tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000
```

Successfully generate:

```text
bbb_assay_candidates.csv
bbb_assay_candidates.jsonl
bbb_assay_report.md
```

The report must show candidates in the following categories:

```text
Direct BBB / brain exposure
Passive permeability / PAMPA-BBB / Caco-2 / MDCK / brain endothelial model
Efflux transporter / ABCB1 / ABCG2 / P-gp / BCRP / efflux ratio
Influx transporter / LAT1 / GLUT1 / OATP / MCT1
```

## Notes

1. Do not treat CNS receptor binding as evidence of BBB crossing.
2. P-gp inhibitor is not equivalent to P-gp substrate; substrate/efflux ratio carries higher weight.
3. PAMPA-BBB only indicates passive permeability, not the absence of efflux.
4. Caco-2 leans more toward intestinal absorption but can serve as auxiliary evidence for general permeability/P-gp.
5. MDCK-MDR1, bidirectional Papp, and efflux ratio are valuable for BBB reasoning.
6. Direct brain/plasma, logBB, Kp,uu,brain, brain perfusion, and CSF/plasma evidence have the highest priority.
7. All rules should be placed in editable configuration or `rules.py`, not hardcoded scattered throughout scripts.
