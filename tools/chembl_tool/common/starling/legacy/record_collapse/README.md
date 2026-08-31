# Record collapse

Historical implementation: `tools/chembl_tool/common/starling/record_collapse.py`
and the former `06_collapsed_records` artifacts. It merged records within a
pair bucket and used semantic aggregation/informativeness review. The active
core retains deduplicated source records and does not call this machinery.

