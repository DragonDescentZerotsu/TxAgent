# Starling BBB / Skin_Reaction import audit (2026-07-22)

## Transfer integrity

Seven parquet files were copied from the configured `Mac` SSH host. Their local SHA-256 values match the remote
files exactly. Per-file source paths, row counts, and hashes are recorded in:

- `bbb_martins/SOURCE_MANIFEST.json`
- `skin_reaction/SOURCE_MANIFEST.json`

Each mechanism acquisition copied `extraction_guidance.json`, `retrieval_spec.json`, and
`pmids_to_process.json` for the transfer-integrity check. Git retains the parquet plus the compact guidance and
retrieval specification. The PMID work queues are reproducible acquisition intermediates, are not read by the
evidence builders, and are excluded by `.gitignore`; the largest queue is 122 MB and exceeds GitHub's per-file
limit. The standalone Skin_Reaction direct source contained only the parquet.

## Ingestion contract

All new parquet sources use the shared profile-driven Starling reader and the `minimal_evidence.v1` contract. The
source column is `SMILES` (uppercase). Rows are aggregated to molecule level within each family, and the stable
canonical-structure ID merges the same standardized molecule across families before fingerprint indexing.

Only extraction-guidance schema values are retained for the declared endpoint field. For the two ambiguous Skin
local-damage endpoints (`inflammatory_response` and `local_tissue_injury`), a source-local context gate additionally
requires explicit skin/dermal/cutaneous/epidermal/topical/rash/irritation/phototoxicity context. This removed 45,585
rows that otherwise included systemic, cardiac, ocular, or unrelated tissue evidence. This filter does not use
benchmark labels or performance.

## Frozen index statistics

### BBB_Martins

- Output: `outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full/`
- Molecule-family evidence rows: 47,419
- Unique index molecules: 37,451
- Direct rows inherited from the existing frozen Starling BBB source: 29,731
- Passive permeability: 2,386 molecule-level rows
- Efflux transport: 13,157 molecule-level rows
- Influx transport: 2,145 molecule-level rows

### Skin_Reaction

- Output: `outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full/`
- Molecule-family evidence rows: 25,525
- Unique index molecules: 17,546
- Direct skin reaction: 3,557 molecule-level rows
- Sensitization AOP: 3,492 molecule-level rows
- Phototoxicity / irritation / local damage: 12,174 molecule-level rows
- Skin exposure: 6,302 molecule-level rows

## Full-test retrieval gate

Settings: Morgan radius 2 / 2048 bits, `top_k_per_group=3`, `min_similarity=0.30`, operational exact-record
exclusion.

- BBB_Martins: 381/392 queries have at least one Starling family neighbor (97.1939%). Family coverage is direct
  380/392, passive 220/392, efflux 332/392, and influx 264/392.
- Skin_Reaction: 77/82 queries have at least one Starling family neighbor (93.9024%). Family coverage is direct
  56/82, sensitization 60/82, local damage 69/82, and exposure 65/82.
- `full_flat` versus `full_mechanism` evidence-union mismatches: 0 for both tasks.
- Exact-query neighbor violations: 0.
- Per-family top-k violations: 0.

The five newly available BBB/Skin Starling conditions subsequently completed on test in identity-blind and
deployment-visible operational modes. All five also completed on valid in identity-blind, matched-prefetch,
deployment-visible operational, and deployment-visible parent-disjoint modes. Every completed condition has
`n_failed_runs == 0`; the valid matched-prefetch audit covers 2,713/2,713 sample-condition pairs, and the valid
parent-disjoint matrix covers 2,275 sample-condition pairs with no final neighbor-identity conflict. Canonical
performance and audit results live in `tools/chembl_tool/paper_experiments/RESULTS.md`; this file remains an import
and ingestion audit rather than a second result table.
