# dili: final reviewed retrieval release

The reviewed source is frozen in the single current source bundle, including the
13-record incremental `../trace_review_20260913/` repair. That review preserves
raw acquisition and frozen gold; it is not a full-library certification.
`publication.json` binds the source, both indices, benchmark hashes and card-link
validation. `source_changes.jsonl` records every applied field change against the
original gold_v4 source; raw source records and gold voting membership are preserved.
`source_validation.json` verifies the full row comparison.

Restore and rebuild through the shared commands in
`tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md`.
Final source records and large audit JSONL files are restored from ordinary Git
archive parts. No model or network identity calls are needed for restoration.

Historical predictions retain their original prepared-input lineage. This final
source consolidation does not itself authorize reuse under changed retrieval inputs.
