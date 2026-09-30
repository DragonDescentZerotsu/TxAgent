# ClinTox base raw source v1

This directory contains the exact files supplied in
`external_source/clintox_base`. The Parquet is tracked
directly so downstream work can start from the original 338,780 source rows.
No cleaning or normalization has been applied to either raw file.

| File | Size | SHA-256 |
|---|---:|---|
| `extractions.parquet` | 88,093,222 bytes | `472b5239e38c59f91d1a60eece7caf7c37a83f3b00b785f9ead59a460b83c285` |
| `extraction_guidance.json` | 7,206 bytes | `94b520acf0c58979580ba685e32a48d5c9f2853f7640e452e6c75adacd8dcb88` |

## Columns

The Parquet has 16 columns:

| Column | Type | Meaning |
|---|---|---|
| `paragraph_idx` | int64 | Source passage index within the PMID. |
| `support_text` | string | Passage text supporting the extraction. |
| `molecule_name` | string | Molecule or clinical-product name used in the passage. |
| `toxicity_outcome` | string | Reported human toxicity, safety outcome, or explicit absence claim. |
| `toxicity_category` | string | Broad toxicity category assigned by the extractor. |
| `outcome_measure` | string | Reported rate, count, grade, comparison, threshold, or other quantitative detail. |
| `clinical_context` | string | Trial, population, indication, formulation, route, or regimen context. |
| `dose_or_exposure` | string | Dose, schedule, concentration, or exposure tied to the claim. |
| `fda_approval_status` | string | Explicitly reported FDA or United States regulatory status. |
| `approved_indication` | string | Explicit FDA-approved indication or population. |
| `extra_details` | string | Additional extracted context not represented elsewhere. |
| `confidence` | float64 | Extractor-provided confidence score. |
| `needs_more_context` | bool | Whether the extractor marked the passage as context-incomplete. |
| `pmid` | string | PubMed identifier. |
| `extraction_id` | string | Extractor-local identifier; not unique across the Parquet. |
| `SMILES` | string | Source-provided molecular structure. |

The extraction guide defines 12 controlled toxicity categories and six FDA
status values, but the raw file also contains off-schema variants. Derived
gold logic must reject rather than silently rewrite those variants.

## Important source limitation

The guidance says that unusual, interpretation-changing modifiers should be
stored in `qualifying_conditions`. That column is absent from the delivered
Parquet. It is therefore **unavailable**, not empty after null cleaning. Any
benchmark derived from this source must disclose that missing semantic gate
and remain pending source-record QA until the affected assumptions are
reviewed.

`extraction_id` is also reused across passages. Downstream record identities
must include the zero-based Parquet row number to remain unique.

See `SOURCE_MANIFEST.json` for the machine-readable schema, completeness
counts, source hashes, and known provenance limitations.
