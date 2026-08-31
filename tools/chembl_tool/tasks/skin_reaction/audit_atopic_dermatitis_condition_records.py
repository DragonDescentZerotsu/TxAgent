"""Audit the manually cleaned atopic-dermatitis condition group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.tasks.skin_reaction.manual_condition_audit import (
    ManualConditionAuditConfig,
    audit_condition,
)


REVIEW_ROOT = Path("data/starling_data/skin_reaction/context_conditioned_review_v1")
FROZEN_ROOT = Path("data/processed_starling_record_supported_v2/Skin_Reaction/scaffold")
CONDITIONED_ROOT = task_root("skin_reaction")


def audit(
    *,
    review_root: Path = REVIEW_ROOT,
    frozen_root: Path = FROZEN_ROOT,
    conditioned_root: Path = CONDITIONED_ROOT,
    manual_review_path: Path = REVIEW_ROOT / "atopic_dermatitis_manual_review_v1.json",
    audit_path: Path = REVIEW_ROOT / "atopic_dermatitis_record_decisions_v1.jsonl",
    summary_path: Path = REVIEW_ROOT / "atopic_dermatitis_record_decisions_v1_summary.json",
) -> dict:
    return audit_condition(
        ManualConditionAuditConfig(
            condition_atom="disease=atopic_dermatitis",
            pure_condition_group="disease=atopic_dermatitis",
            audit_version="skin_atopic_dermatitis_record_audit.v1",
            manual_review_path=manual_review_path,
            audit_path=audit_path,
            summary_path=summary_path,
            review_root=review_root,
            frozen_root=frozen_root,
            conditioned_root=conditioned_root,
        )
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
