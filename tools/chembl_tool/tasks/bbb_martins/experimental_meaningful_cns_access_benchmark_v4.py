"""BBB gold v4 with conservative reviewed recovery of missing directions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.benchmark_dataset import LabelDecision

from .experimental_meaningful_cns_access_benchmark import (
    SOURCE_REVISION,
    classify_scope,
    label_decisions_from_rows,
)
from .experimental_meaningful_cns_access_benchmark_v3 import (
    label_record as label_record_v3,
)
from .experimental_metric_direction_review import (
    MANUAL_SOURCE_EXCLUSIONS,
    REVIEW_VERSION,
    REVIEWED_PROPOSED_SOURCE_INDICES,
    SOURCE_ARROW_SHA256,
    adjudicate_missing_direction,
)


CONTRACT_VERSION = "bbb_experimental_meaningful_cns_access_gold.v4"


def label_record(
    record: Mapping[str, Any],
    *,
    source_index: int | None = None,
    allow_conditioned_context: bool = False,
) -> tuple[int | None, str]:
    """Apply v3, then recover only frozen manually approved directions."""

    label, method = label_record_v3(
        record,
        source_index=source_index,
        allow_conditioned_context=allow_conditioned_context,
    )
    if label is not None or method != "no_explicit_binary_permeability_label":
        return label, method
    if source_index is None:
        return None, "missing_source_index_for_metric_direction_review"

    review = adjudicate_missing_direction(record, source_index=source_index)
    if review.label is None:
        return None, f"metric_direction_review:{review.rule_id}"
    scope = classify_scope(
        record,
        allow_conditioned_context=allow_conditioned_context,
    )
    if scope.rejection_reason is not None:
        raise RuntimeError("v4 review escaped the frozen v3 experimental scope")
    return (
        review.label,
        f"{CONTRACT_VERSION}:{scope.endpoint_family}:{scope.basis}:"
        f"{review.rule_id}",
    )


def load_label_decisions_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    resolved_revision: str = SOURCE_REVISION,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    decisions, metadata = label_decisions_from_rows(
        rows,
        resolved_revision=resolved_revision,
        label_decider=label_record,
        contract_version=CONTRACT_VERSION,
    )
    metadata["gold_contract"]["v4_change_from_v3"] = (
        "recover only source-index-reviewed unambiguous qualitative directions "
        "from v3 rows rejected solely for a null bbb_permeability_label"
    )
    metadata["gold_contract"]["metric_direction_review_version"] = REVIEW_VERSION
    metadata["gold_contract"]["label_rule"] = (
        "use an explicit source-native bbb_permeability_label or a frozen "
        "source-index-reviewed unambiguous qualitative direction after the same "
        "experimental scope checks; do not threshold heterogeneous numeric endpoints"
    )
    metadata["gold_contract"]["metric_direction_review"] = {
        "proposed_rows_manually_reviewed": len(REVIEWED_PROPOSED_SOURCE_INDICES),
        "approved_new_voting_rows": (
            len(REVIEWED_PROPOSED_SOURCE_INDICES) - len(MANUAL_SOURCE_EXCLUSIONS)
        ),
        "manually_excluded_rows": len(MANUAL_SOURCE_EXCLUSIONS),
        "review_artifact": (
            "data/starling_data/bbb_martins/"
            "experimental_metric_direction_review_v1/summary.json"
        ),
    }
    return decisions, metadata


def load_label_decisions_from_arrow(
    path: str | Path,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    from datasets import Dataset

    source_path = Path(path)
    hasher = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != SOURCE_ARROW_SHA256:
        raise ValueError("local BBB Arrow source does not match the reviewed copy")
    rows = Dataset.from_file(str(source_path))
    decisions, metadata = load_label_decisions_from_rows(rows)
    metadata["local_source_arrow"] = str(source_path.resolve())
    metadata["local_source_arrow_sha256"] = digest
    return decisions, metadata
