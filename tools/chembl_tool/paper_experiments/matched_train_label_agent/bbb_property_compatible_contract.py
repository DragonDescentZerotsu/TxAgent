"""Frozen paths and settings for the E16 BBB selector candidate."""

from dataclasses import replace
from pathlib import Path

from .contract import TASK_SPECS


SCHEMA_VERSION = "bbb_property_compatible_matched_agent.v1"
CONDITION = "bbb_martins__matched_train_label_property_compatible_direct"
SIMILARITY_BUDGET = 0.02
TOP_N = 20
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/matched_train_label_direct_agent_bbb_property_compatible_v1_"
    "scaffold_valid_gpt_oss_120b"
)
AVAILABILITY_ROOT = Path(
    "outputs/paper/bbb_property_compatibility_availability_"
    "experimental_meaningful_cns_access_v2_valid"
)
BASELINE_V3_ROOT = Path(
    "outputs/paper/matched_train_label_direct_agent_meaningful_cns_adjudication_"
    "v3_scaffold_valid_gpt_oss_120b_formal"
)
BASELINE_V3_RUN_ROOT = BASELINE_V3_ROOT / "runs_identity_blind_parent_disjoint"
SPEC = replace(TASK_SPECS["bbb_martins"], condition=CONDITION)
