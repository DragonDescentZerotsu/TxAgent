"""TDC-compatible binary-label adapter for Starling oral bioavailability data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

from tools.chembl_tool.common.starling.benchmark_dataset import (
    LabelDecision,
    LabeledSourceRecord,
    accepted,
    classify_interval,
    has_reported_text,
    normalize_numeric_text,
    parse_numeric_interval,
    rejected,
    sha256_file,
)
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    CANONICAL_VERSION,
    DIRECT_CLAIMS_PATH,
    DIRECT_REPORT_TYPES,
    MANIFEST_PATH,
)

TDC_BIOAVAILABILITY_THRESHOLD_PERCENT = 20.0
FROZEN_GOLD_QUALITATIVE_POLICY_VERSION = (
    "bioavailability_gold_qualitative_substring.v1"
)

# Frozen benchmark lineage.  These patterns deliberately remain private to
# the gold adapter and must not be reused by evidence normalization.  Their
# behavior is preserved byte-for-byte at the decision level so the published
# benchmark does not change while the v7 evidence parser becomes stricter.
POSITIVE_QUALITATIVE_PATTERNS = (
    r"\bhigh\b",
    r"\bgood\b",
    r"\bexcellent\b",
    r"\bcomplete(?:ly)?\b",
    r"\bnear(?:ly)? complete\b",
    r"\balmost complete\b",
)
NEGATIVE_QUALITATIVE_PATTERNS = (
    r"\bvery low\b",
    r"\blow\b",
    r"\bpoor\b",
    r"\bnegligible\b",
    r"\bminimal\b",
)
AMBIGUOUS_QUALITATIVE_PATTERNS = (
    r"\bmoderate\b",
    r"\bvariable\b",
    r"\bunpredictable\b",
    r"\bintermediate\b",
    r"\borally bioavailable\b",
    r"\borally available\b",
)

NON_HUMAN_PATTERN = re.compile(
    r"\b(?:rat|rats|mouse|mice|dog|dogs|canine|beagle|pig|pigs|swine|monkey|monkeys|"
    r"macaque|primate|rabbit|rabbits|guinea pig|hamster|sheep|goat|horse|horses|"
    r"bovine|cow|cattle|chicken|fish|zebrafish|rodent|animal|non-human|nonhuman)\b",
    flags=re.IGNORECASE,
)
HUMAN_PATTERN = re.compile(
    r"\b(?:human|humans|subject|subjects|participant|participants|volunteer|volunteers|"
    r"patient|patients|men|women|male adults|female adults|healthy adults|elderly|"
    r"pediatric|paediatric|children|adolescents|infants)\b",
    flags=re.IGNORECASE,
)


def load_label_decisions(
    *,
    source_path: str | Path = DIRECT_CLAIMS_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    max_rows: int = 0,
) -> tuple[Iterable[LabelDecision], dict[str, Any]]:
    """Load one vote per canonical direct claim shared with agent evidence."""
    import pandas as pd

    path = Path(source_path)
    frame = pd.read_parquet(path)
    if max_rows:
        frame = frame.head(max_rows)
    records = frame.to_dict(orient="records")
    merge_manifest_path = Path(manifest_path)
    merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))

    metadata = {
        "canonical_source": {
            "contract_version": CANONICAL_VERSION,
            "path": str(path),
            "sha256": sha256_file(path),
            "manifest_path": str(merge_manifest_path),
            "manifest_sha256": sha256_file(merge_manifest_path),
            "upstream_hf": merge_manifest.get("hf_source", {}),
            "upstream_local": merge_manifest.get("local_source", {}),
            "deduplication_policy": merge_manifest.get("deduplication_policy", {}),
        },
        "tdc_compatibility": {
            "population_scope": "human only",
            "positive_label": "human oral bioavailability F >= 20%",
            "negative_label": "human oral bioavailability F < 20%",
            "numeric_units": "percent and unitless fraction normalized to percent",
            "qualitative_policy": "frozen historical substring policy",
            "qualitative_policy_version": FROZEN_GOLD_QUALITATIVE_POLICY_VERSION,
        },
    }
    return (
        (_label_canonical_record(index, row, source_path=path) for index, row in enumerate(records)),
        metadata,
    )


def label_bioavailability_value(value: Any) -> tuple[int | None, str]:
    """Map a direct oral-F value to the TDC threshold without hiding ambiguity."""
    text = normalize_numeric_text(value)
    if not text:
        return None, "missing_bioavailability_value"
    lowered = text.lower()
    if re.search(
        r"\b(?:fold|times|relative)\b|\b(?:increase|increased|decrease|decreased)\s+by\b|"
        r"\b(?:higher|lower)\s+than\b",
        lowered,
    ):
        return None, "relative_not_absolute_bioavailability"

    if any(re.search(pattern, lowered) for pattern in AMBIGUOUS_QUALITATIVE_PATTERNS):
        if not re.search(r"\d", text):
            return None, "qualitative_value_not_threshold_anchored"
    if not re.search(r"\d", text):
        positive = any(
            re.search(pattern, lowered)
            for pattern in POSITIVE_QUALITATIVE_PATTERNS
        )
        negative = any(
            re.search(pattern, lowered)
            for pattern in NEGATIVE_QUALITATIVE_PATTERNS
        )
        if positive and not negative:
            return 1, "explicit_qualitative_high"
        if negative and not positive:
            return 0, "explicit_qualitative_low"
        return None, "unmapped_or_ambiguous_qualitative_value"

    interval = parse_numeric_interval(text, fraction_to_percent=True)
    if interval is None:
        return None, "unparseable_numeric_value"
    if interval.lower != -math.inf and interval.lower < 0:
        return None, "numeric_value_out_of_percent_range"
    if interval.upper != math.inf and interval.upper > 100:
        return None, "numeric_value_out_of_percent_range"
    label = classify_interval(interval, threshold=TDC_BIOAVAILABILITY_THRESHOLD_PERCENT)
    if label is None:
        return None, "numeric_interval_crosses_20_percent_threshold"
    return label, f"numeric_20_percent_threshold:{interval.method}"


def is_human_context(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or NON_HUMAN_PATTERN.search(text):
        return False
    return HUMAN_PATTERN.search(text) is not None


def _label_canonical_record(
    index: int,
    row: Mapping[str, Any],
    *,
    source_path: Path = DIRECT_CLAIMS_PATH,
) -> LabelDecision:
    """Map a deduplicated canonical claim to one auditable benchmark vote."""
    source_id = f"canonical:{source_path}"
    report_type = str(row.get("bioavailability_report_type") or "").strip().lower()
    if report_type not in DIRECT_REPORT_TYPES:
        return rejected(
            "non_direct_oral_bioavailability_report_type",
            source_id=source_id,
            source_index=index,
            report_type=report_type,
        )
    population = row.get("species_or_population")
    if not is_human_context(population):
        return rejected(
            "nonhuman_or_unresolved_population",
            source_id=source_id,
            source_index=index,
            population=population,
        )
    if has_reported_text(row.get("qualifying_conditions")):
        return rejected(
            "interpretation_altering_qualifying_conditions",
            source_id=source_id,
            source_index=index,
            qualifying_conditions=row.get("qualifying_conditions"),
        )
    label, method = label_bioavailability_value(row.get("oral_bioavailability_value"))
    if label is None:
        return rejected(
            method,
            source_id=source_id,
            source_index=index,
            value=row.get("oral_bioavailability_value"),
        )
    smiles = str(row.get("smiles") or "").strip()
    if not smiles:
        return rejected("missing_smiles", source_id=source_id, source_index=index)
    return accepted(
        LabeledSourceRecord(
            smiles=smiles,
            label=label,
            source_id=source_id,
            source_record_id=str(row.get("canonical_claim_id") or f"row:{index}"),
            pmid=str(row.get("pmid") or ""),
            label_method=f"canonical:{method}",
            raw_value=str(row.get("oral_bioavailability_value") or ""),
            context=str(population or ""),
        )
    )
