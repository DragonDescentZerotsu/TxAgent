# Pass/fail encoding check (2026-09-24)

Source: [completed-v2 archive at 4b4489c1](https://github.com/DragonDescentZerotsu/TxAgent/tree/4b4489c1598ba0fc428308b930d0ca71a90f2c84/outputs/paper/assay_transfer_harness/joseph/trace_archives/final_test_20260923_completed_v2).
We screened all 3,147 successful Gold-v1 outputs in its nine Ames, Carcinogens
and Skin result leaves, including alternate profiles: 548 Ames, 1,876
Carcinogens and 723 Skin outputs. These are predictions, not unique molecules.
Original and recovery finals were joined to the scored labels. Archive-part
and selected-file SHA-256 checks passed. This is keyword-assisted screening
with manual inspection of candidate conclusions, raw labels and mappings,
not an exhaustive semantic review or an estimate of error prevalence.

## Confirmed final inversions

All query indices below are zero-based. These three examples are **Direct
recovery** outputs, not evidence that the problem is specific to Indirect.
The task instructions correctly say pass = positive; the model uses the
opposite meaning in its final reasoning, and its raw final label retains it.

| Task / profile | Query | Final reasoning excerpt | Raw → parsed | Gold |
|---|---:|---|---|---|
| Ames / `ga050_mc010_label010` | 115 | `pass` (negative for mutagenicity) | pass → positive | negative |
| Carcinogens / `ga075_mc000_label000` | 290 | `fail` meaning the molecule is predicted to be carcinogenic | fail → negative | positive |
| Carcinogens / `ga100_mc010_label025` | 271 | `pass` (i.e., not a carcinogen) | pass → positive | negative |

These establish that the DILI failure mode also occurs in Ames and Carcinogens.
They are confirmed examples, not total error counts. The parser follows its
contract; globally reversing that mapping would corrupt correctly encoded cases.

## Skin: no final inversion confirmed by this screening

Skin Mixed `ga075_mc010_sr050_sd010_ld025` #107 initially considers
`pass (no risk)`, repeatedly questions the unusual naming, then explicitly
resolves it: `"fail" = non-sensitizer. This is unusual but explicit.` Its final
conclusion is no risk and its output is fail → no_risk. That prediction is
wrong against gold positive, but the final label encoding is consistent;
it must not be counted as a final encoding error.
The 23 Skin recovery conclusions were also inspected; no final inversion was
confirmed there. This does not certify every Skin trace as semantically correct.

## What to change

Use the small [DILI patch](DILI_HANDOFF.md) now supplied in the shared bundle.
For Ames and Carcinogens, the analogous follow-up is native `positive` /
`negative` labels with matching allowed values and identity mappings. Skin's
native pair is `risk` / `no_risk`; the shared naming risk exists, but this screen
has not established a final Skin inversion. Only DILI is changed in this patch.
Keep the two-field JSON schema and existing parser; use a fresh prompt identity
and output root. Do not rewrite historical scores by parsing reasoning text.

This issue alone does not explain an Indirect-vs-Direct performance gap. The
confirmed examples above occur in Direct, while DILI affects both arms. Endpoint
transfer, evidence concentration and retrieval differences remain separate.
No LLM rerun or causal performance improvement is claimed.

The compact [audit receipt](label_encoding_audit_20260924.json) records exact
archive paths, profiles, file hashes, label mappings and excerpts, plus the
DILI offline validation result. It also lists every screened result leaf, so
collaborators can locate these traces without relying on query index alone.
