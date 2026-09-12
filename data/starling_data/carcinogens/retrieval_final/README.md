# carcinogens: final reviewed retrieval release

The closed cleaning cycle is frozen in the single current source bundle.
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
