# Retired evidence-library construction inputs

This directory keeps compact lineage needed to understand or reproduce a
superseded construction decision. Active V7 and V8 builders do not import it.

- `bbb_martins/measurement_resolution_v8/` preserves the prior Luna mapping and
  its two one-off preparation scripts.
- `prompts/measurement_resolution/` preserves prompts no active task selects.
- `prompts/semantic_record_aggregation/` preserves superseded prompt revisions;
  active releases retain only the revision they select.

Do not add request queues, review packets, progress logs, or duplicate exports.
Reusable provider caches remain under the documented artifact/cache location.
