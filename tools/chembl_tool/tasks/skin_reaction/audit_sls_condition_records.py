"""Audit manually cleaned SLS qualifying-condition records and promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.tasks.skin_reaction.manual_condition_audit import (
    ManualConditionAuditConfig,
    audit_condition,
)


REVIEW_ROOT = Path("data/artifacts/starling/skin_reaction/source_reviews/context_conditioned_review_v1")
FROZEN_ROOT = Path("data/gold_labels/legacy/processed_starling_record_supported_v2/Skin_Reaction/scaffold")


def audit(
    *,
    review_root: Path = REVIEW_ROOT,
    frozen_root: Path = FROZEN_ROOT,
    manual_review_path: Path = REVIEW_ROOT / "sls_manual_review_v1.json",
    audit_path: Path = REVIEW_ROOT / "sls_record_decisions_v1.jsonl",
    summary_path: Path = REVIEW_ROOT / "sls_record_decisions_v1_summary.json",
) -> dict:
    return audit_condition(
        ManualConditionAuditConfig(
            condition_atom="coexposure=sls_barrier_enhancement",
            pure_condition_group="coexposure=sls_barrier_enhancement",
            audit_version="skin_sls_record_audit.v1",
            manual_review_path=manual_review_path,
            audit_path=audit_path,
            summary_path=summary_path,
            review_root=review_root,
            frozen_root=frozen_root,
        )
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
