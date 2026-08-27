# Pair-bucket variance audit

This directory keeps the reproducible analysis used to decide whether normalized-v7 pair buckets are too broad or too narrow.

Run it from the TxAgent repository root:

```bash
/usr/bin/python tools/chembl_tool/common/starling/pair_bucket_variance_audit/analyze.py \
  --output tools/chembl_tool/common/starling/pair_bucket_variance_audit/RESULTS.md
```

The script is read-only with respect to evidence libraries. It reads the current BBB, Bioavailability, and Skin Stage-03 artifacts, including Bioavailability's canonical V7 exact-dose fields. Regenerate Bioavailability Stage 02/03 after the shared exact-unit map is finalized before rerunning this audit. The script aborts if an input artifact changes while being read.

`RESULTS.md` contains the generated cross-source tables, definitions, input hashes, and oral exact-dose comparison.

## Four P1 review findings, reassessed

### 1. Direct-family `group_id` rewriting: confirmed P1

The collapse stage now keeps the configured paper-facing `group_id` unchanged and uses `retrieval_source_id` for direct vote/residual partition membership. A regression test prevents reintroducing the old prefix behavior.

### 2. Canonical aggregate values in collapsed prompts: reviewer fix rejected

The reviewer asked to restore source measurement text and source units. That is correct for an uncollapsed source record, but not for a median or mode that did not exist in any one source row. After collapse, the truthful display is the canonical aggregate value, canonical unit, aggregation method, source-record count, and range or category counts.

The shared task contract should be clarified: source projections remain source-faithful before collapse; post-collapse projections expose an explicitly labeled normalized aggregate. Replacing a multi-record canonical median with one representative source value would be incorrect.

### 3. Preserve every context-value pairing: reframe, not a generic P1

The intended contract is to discard non-key context after records are grouped by `(pair_bucket_key, canonical_smiles)`. Keeping every source context-value pairing would undo the collapse.

The real risks are narrower:

1. A material context must be promoted into the pair-bucket key before collapse. The audit shows normalized exact dose materially changes Cmax and AUC variation, so dose-sensitive oral endpoints need an explicit dose decision.
2. A collapsed deterministic row must not retain one representative's non-key context and present it as if it described the aggregate. Downstream rendering should expose only pair-bucket context plus aggregate statistics; representative-only residual fields should be omitted.

Thus the review identified a real misattribution hazard, but retaining all context-value pairings is not the desired fix.

### 4. Skin canonical direct/AOP partition: confirmed P1

The canonical partition audit contains:

| Raw source | Direct | AOP | Rejected |
|---|---:|---:|---:|
| `direct_skin_reaction` | 42,986 | 544 | 23,067 |
| `sensitization_aop` | 12,095 | 11,219 | 22,671 |

The normalized routing and direct adapter now apply this partition to both raw sources before condition/vote handling. This admits the canonical-direct final outcomes from `sensitization_aop`, routes the 544 mechanistic rows from `direct_skin_reaction` to AOP, and excludes partition-rejected rows from retrieval.
