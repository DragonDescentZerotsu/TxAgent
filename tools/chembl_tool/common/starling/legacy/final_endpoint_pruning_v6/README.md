# Final endpoint pruning v6

Historical implementation: `tools/chembl_tool/common/starling/final_endpoint_pruning.py`
and task output directories named `final_endpoint_pruning_v*`. These artifacts
are preserved for frozen lineage but are incompatible with the new pair-key
version and are not read by the three-stage core. Unreviewed Stage 3 buckets
are usable when they pass deterministic calibration checks.

