"""Prepare and build the reviewed Skin sensitization conditioned benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.condition_review import (
    attach_parent_identity,
    merge_terminal_verdicts,
    write_review_queue,
)
from tools.chembl_tool.common.starling.external_condition import (
    AtomRule,
    ConditionAtom,
    payload_sha256,
    propose_pattern_condition,
)
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    build_reviewed_conditioned_benchmark,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import BUILD_ROOT, CONTRACT
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import DIRECT_RECORDS_PATH
from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import label_record


PROPOSAL_VERSION = "skin_external_condition_proposal.v2"
LINEAGE = "skin_sensitization_context_conditioned_v1"
SELECTED_LINEAGE = CONTRACT
FROZEN_ROOT = Path("data/processed_starling_record_supported_v2/Skin_Reaction/scaffold")
REVIEW_ROOT = Path("data/starling_data/skin_reaction/context_conditioned_review_v1")
OUTPUT_ROOT = Path("data/processed_starling_context_conditioned_reviewed_v1/Skin_Reaction/scaffold")
SELECTED_OUTPUT_ROOT = BUILD_ROOT / "Skin_Reaction/scaffold"
SELECTED_EXTERNAL_CONDITION_GROUPS = ("disease=atopic_dermatitis",)


RULES = (
    AtomRule.make("application", "semi_occluded", r"semi[- ]?occlu"),
    AtomRule.make("application", "open_or_unoccluded", r"without occlusion|avoid occlusion|no occlusion|non[- ]?occlu|unocclu|open application|repeated open application|open epicutaneous"),
    AtomRule.make("application", "occluded", r"\bocclu(?:ded|sion|sive)\b"),
    AtomRule.make("barrier", "abraded_or_stripped", r"damaged skin|abraded skin|skin abrasion|tape[- ]?strip|stripped skin|compromised skin barrier|barrier[- ]?disrupt"),
    AtomRule.make("coexposure", "sls_barrier_enhancement", r"sodium lauryl sul(?:fate|phate)|sodium dodecyl sul(?:fate|phate)|\bsls\b|\bsds\b"),
    AtomRule.make("disease", "atopic_dermatitis", r"atopic dermatitis|atopic eczema"),
    AtomRule.make("disease", "psoriasis", r"psoriasis|psoriatic"),
    AtomRule.make(
        "age_group",
        "pediatric",
        r"p[ae]diatric|\bchildren\b|\bchild\b|\binfants?\b|adolescen",
    ),
    AtomRule.make(
        "age_group",
        "elderly",
        r"\belderly\b|older (?:adults?|patients?|subjects?|men|women|people)|"
        r"\b(?:6[5-9]|[7-9]\d)[- ]?year[- ]old\b|aged (?:65|[7-9]\d) (?:years? )?(?:or older|and older)",
    ),
    AtomRule.make(
        "population_state",
        "healthy_volunteers",
        r"healthy (?:human )?(?:volunteers?|subjects?|participants?)|normal (?:volunteers?|subjects?|controls?)",
    ),
    AtomRule.make("vehicle", "petrolatum", r"petrolatum|vaseline|\bpet\.?\b"),
    AtomRule.make("vehicle", "dmso", r"dimethyl sulfoxide|\bdmso\b"),
    AtomRule.make(
        "vehicle",
        "ethanol",
        r"\d+(?:\.\d+)?\s*%\s*(?:in\s+)?(?:ethanol|ethyl alcohol)|"
        r"\b(?:ethanol|ethyl alcohol)(?:ic)?\s+(?:solution|vehicle)|\beth\.?\b",
    ),
    AtomRule.make("vehicle", "aqueous", r"aqueous|water vehicle|in water|\baq\.?\b"),
)

_SELF_CONDITION_SMILES = {
    ConditionAtom("vehicle", "ethanol"): {"CCO"},
    ConditionAtom("vehicle", "dmso"): {"CS(C)=O", "C[S+](C)[O-]"},
    ConditionAtom("coexposure", "sls_barrier_enhancement"): {
        "CCCCCCCCCCCCOS(=O)(=O)O"
    },
}


def _suppress(atoms: set[ConditionAtom]) -> set[ConditionAtom]:
    output = set(atoms)
    if ConditionAtom("application", "semi_occluded") in output:
        output.discard(ConditionAtom("application", "occluded"))
    if ConditionAtom("application", "open_or_unoccluded") in output:
        output.discard(ConditionAtom("application", "occluded"))
    return output


def _remove_query_material_conditions(
    atoms: tuple[ConditionAtom, ...], parent_smiles: str
) -> tuple[ConditionAtom, ...]:
    return tuple(
        atom
        for atom in atoms
        if parent_smiles not in _SELF_CONDITION_SMILES.get(atom, set())
    )


def _explicit_human(value: object) -> bool:
    text = str(value or "").lower()
    return bool(
        re.search(
            r"human|patient|volunteer|worker|hairdresser|child|adult|subject|participant",
            text,
        )
    )


def prepare_review_queue() -> dict[str, Any]:
    frame = pd.read_parquet(DIRECT_RECORDS_PATH)
    candidates = []
    proposal_audit = []
    for index, row in frame.iterrows():
        record_id = str(row["source_record_id"])
        if not _explicit_human(row.get("species_or_population")):
            proposal_audit.append(
                {
                    "source_record_id": record_id,
                    "source_index": int(index),
                    "queue_status": "not_queued",
                    "proposal_reason": "not_explicit_human_context",
                }
            )
            continue
        label, method = label_record(row)
        if label is None:
            proposal_audit.append(
                {
                    "source_record_id": record_id,
                    "source_index": int(index),
                    "queue_status": "not_queued",
                    "proposal_reason": f"endpoint_ineligible:{method}",
                }
            )
            continue
        context_fields = (
            row.get("assay_or_test"),
            row.get("species_or_population"),
            row.get("dose_or_concentration"),
            row.get("extra_details"),
            row.get("support_text"),
        )
        context = " | ".join(str(value or "") for value in context_fields)
        proposal = propose_pattern_condition(
            context,
            rules=RULES,
            incompatible_families={
                "application", "vehicle", "disease", "age_group", "population_state"
            },
            suppress=_suppress,
        )
        audit = {
            "source_record_id": record_id,
            "source_index": int(index),
            "condition_text": context,
            "proposal_status": proposal.status,
            "proposal_reason": proposal.reason,
            "proposed_condition_group": proposal.signature or "",
        }
        if not proposal.atoms:
            proposal_audit.append({**audit, "queue_status": "not_queued"})
            continue
        if proposal.signature is None:
            proposal_audit.append(
                {
                    **audit,
                    "queue_status": "not_queued",
                    "proposal_reason": proposal.reason,
                }
            )
            continue
        identity = attach_parent_identity({}, row.get("SMILES"))
        atoms = _remove_query_material_conditions(
            proposal.atoms, str(identity.get("drug") or "")
        )
        if not atoms:
            proposal_audit.append(
                {
                    **audit,
                    "queue_status": "not_queued",
                    "proposal_reason": "condition_material_is_query_molecule",
                }
            )
            continue
        signature = "+".join(atom.key for atom in atoms)
        candidate = {
            **identity,
            **{
                "source_record_id": record_id,
                "source_payload_sha256": payload_sha256(
                    (PROPOSAL_VERSION, signature, int(label), record_id, row.get("SMILES"), row.get("outcome_label"), *context_fields)
                ),
                "source_index": int(index),
                "pmid": str(row.get("pmid") or ""),
                "condition_text": context,
                "support_text": str(row.get("support_text") or ""),
                "raw_value": str(row.get("outcome_label") or ""),
                "proposed_condition_group": signature,
                "proposed_condition_atoms": [atom.key for atom in atoms],
                "proposal_reason": proposal.reason,
                "Y": int(label),
                "label_method": method,
            },
        }
        if candidate["molecule_identity_key"]:
            candidates.append(candidate)
            proposal_audit.append(
                {**audit, "queue_status": "queued", "proposed_condition_group": signature}
            )
        else:
            proposal_audit.append(
                {**audit, "queue_status": "not_queued", "proposal_reason": "invalid_parent_identity"}
            )
    return write_review_queue(
        rows=candidates,
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        manifest_path=REVIEW_ROOT / "review_queue_manifest.json",
        task_name="Skin_Reaction",
        source_artifacts=(DIRECT_RECORDS_PATH,),
        proposal_version=PROPOSAL_VERSION,
        proposal_audit_rows=proposal_audit,
    )


def _apply_selected_manual_review(
    reviewed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions_path = REVIEW_ROOT / "atopic_dermatitis_record_decisions_v1.jsonl"
    decisions = {
        row["source_record_id"]: row
        for row in (
            json.loads(line)
            for line in decisions_path.open(encoding="utf-8")
            if line.strip()
        )
        if row["exact_pure_signature"]
    }
    output = []
    seen = set()
    for source in reviewed:
        row = dict(source)
        decision = decisions.get(row["source_record_id"])
        if decision is not None:
            seen.add(row["source_record_id"])
            row.update(
                {
                    "review_status": (
                        "accepted"
                        if decision["clean_status"] == "eligible"
                        else "rejected"
                    ),
                    "reviewer": "manual_atopic_dermatitis_review_v1",
                    "review_reason": decision["clean_reason"],
                    "review_notes": decision["manual_note"],
                }
            )
            if decision["clean_status"] == "eligible":
                row["condition_group"] = "disease=atopic_dermatitis"
                row["condition_atoms"] = ["disease=atopic_dermatitis"]
        output.append(row)
    if seen != set(decisions):
        raise ValueError(
            "Selected Skin build did not consume every pure AD manual decision: "
            f"missing={sorted(set(decisions) - seen)}"
        )
    return output


def build(*, selected: bool = False) -> dict[str, Any]:
    if selected:
        # Revalidate the manual manifest, exact pure-signature coverage, and
        # source hashes before consuming its record decisions.
        from tools.chembl_tool.tasks.skin_reaction.audit_atopic_dermatitis_condition_records import (
            audit as audit_atopic_dermatitis,
        )

        audit_atopic_dermatitis()
    reviewed = merge_terminal_verdicts(
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        verdict_path=REVIEW_ROOT / "review_verdicts.jsonl",
    )
    if selected:
        reviewed = _apply_selected_manual_review(reviewed)
    return build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="Skin_Reaction",
            lineage=SELECTED_LINEAGE if selected else LINEAGE,
            protocol_version=(
                "skin_context_conditioned_selected.v1"
                if selected
                else "skin_context_conditioned_benchmark.v1"
            ),
            frozen_root=FROZEN_ROOT,
            output_root=SELECTED_OUTPUT_ROOT if selected else OUTPUT_ROOT,
            source_artifacts=(
                DIRECT_RECORDS_PATH,
                REVIEW_ROOT / "review_queue.jsonl",
                REVIEW_ROOT / "review_queue_manifest.json",
                REVIEW_ROOT / "model_prereview.jsonl",
                REVIEW_ROOT / "model_prereview_round2.jsonl",
                REVIEW_ROOT / "review_payload_rebind_receipt.json",
                REVIEW_ROOT / "review_comparison_summary.json",
                REVIEW_ROOT / "review_verdicts.jsonl",
                *(
                    (
                        REVIEW_ROOT / "atopic_dermatitis_manual_review_v1.json",
                        REVIEW_ROOT / "atopic_dermatitis_record_decisions_v1.jsonl",
                        REVIEW_ROOT / "atopic_dermatitis_record_decisions_v1_summary.json",
                    )
                    if selected
                    else ()
                ),
            ),
            row_id_prefix="SKINCTXSEL" if selected else "SKINCTX",
            frozen_lineage="record_supported_v2",
            allowed_condition_groups=(
                SELECTED_EXTERNAL_CONDITION_GROUPS if selected else None
            ),
            publish_allowed_group_audit_only=selected,
            review_policy=(
                "two semantic prereviews plus exhaustive manual adjudication of "
                "the pure atopic-dermatitis signature; relative-only, duplicate, "
                "hidden-composite, and attribution-ambiguous records are rejected"
                if selected
                else ConditionedBenchmarkConfig.review_policy
            ),
        ),
        review_rows=reviewed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare-review", "build", "build-selected")
    )
    args = parser.parse_args(argv)
    result = (
        prepare_review_queue()
        if args.action == "prepare-review"
        else build(selected=args.action == "build-selected")
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
