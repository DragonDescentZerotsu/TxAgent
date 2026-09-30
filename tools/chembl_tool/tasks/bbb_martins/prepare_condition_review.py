"""Prepare the frozen semantic-review queue for BBB external conditions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from tools.chembl_tool.common.starling.condition_review import attach_parent_identity, write_review_queue
from tools.chembl_tool.common.starling.external_condition import (
    AtomRule,
    ConditionAtom,
    payload_sha256,
    propose_pattern_condition,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    SOURCE_DATASET,
    SOURCE_REVISION,
    label_record,
)


PROPOSAL_VERSION = "bbb_external_condition_proposal.v2"
REVIEW_ROOT = Path("data/starling_data/bbb_martins/context_conditioned_review_v1")
FROZEN_CANDIDATE_SOURCE = REVIEW_ROOT / "frozen_candidate_source.parquet"

SELECTED_EXTERNAL_CONDITION_GROUPS = (
    "barrier_state=disrupted",
    "co_treatment=cyclosporine",
    "disease=meningitis_unspecified",
    "disease=bacterial_meningitis",
    "disease=pneumococcal_meningitis",
    "disease=tuberculous_meningitis",
    "disease=brain_tumor_or_glioma",
    "disease=cerebral_ischemia",
)


RULES = (
    AtomRule.make("route", "oral", r"\boral(?:ly)?\b|\bpo\b"),
    AtomRule.make("route", "intravenous", r"\bintravenous\b|(?<!stage )(?<!phase )\bi\.?v\.?\b"),
    AtomRule.make("route", "intraperitoneal", r"\bintraperitoneal\b|\bi\.?p\.?\b"),
    AtomRule.make("route", "subcutaneous", r"\bsubcutaneous\b|\bs\.?c\.?\b"),
    AtomRule.make("barrier_state", "intact", r"intact (?:blood[- ]brain barrier|bbb)|normal (?:blood[- ]brain barrier|bbb)|uninflamed meninges|non[- ]?inflamed meninges|absence of meningeal inflammation|normal meninges"),
    AtomRule.make("barrier_state", "disrupted", r"bbb disruption|disrupted (?:blood[- ]brain barrier|bbb)|damaged (?:blood[- ]brain barrier|bbb)|injured (?:blood[- ]brain barrier|bbb)|osmotic (?:bbb )?(?:opening|disruption)|focused ultrasound"),
    AtomRule.make("disease", "bacterial_meningitis", r"bacterial meningitis"),
    AtomRule.make("disease", "pneumococcal_meningitis", r"pneumococcal meningitis"),
    AtomRule.make("disease", "tuberculous_meningitis", r"tuberculous meningitis|tb meningitis"),
    AtomRule.make("disease", "meningitis_unspecified", r"\bmeningitis\b|inflamed meninges|meningeal inflammation"),
    AtomRule.make("disease", "hiv_infection", r"hiv[- ]infect|\bhiv\b|\baids\b"),
    AtomRule.make("disease", "siv_infection", r"siv infection|siv[- ]infect"),
    AtomRule.make("disease", "alzheimers_disease", r"alzheimer|\bad patients?\b|ad[- ]like"),
    AtomRule.make("disease", "parkinsons_disease", r"parkinson|alpha[- ]syn|α[- ]syn"),
    AtomRule.make("disease", "eae", r"experimental autoimmune encephalomyelitis|\beae\b"),
    AtomRule.make("disease", "brain_tumor_or_glioma", r"brain tumou?r|intracranial tumou?r|\bglioma\b|glioblastoma"),
    AtomRule.make("disease", "cerebral_ischemia", r"cerebral ischemi|focal ischemi|ischemic stroke"),
    AtomRule.make("disease", "chronic_liver_failure", r"chronic liver failure|hepatic failure"),
    AtomRule.make("age_group", "fetal", r"\bfetal\b|\bfoetal\b"),
    AtomRule.make("age_group", "neonatal", r"\bneonat|\bnewborn|immature blood[- ]brain barrier"),
    AtomRule.make("age_group", "pediatric", r"\bchildren\b|p[ae]diatric|\binfants?\b|adolescen"),
    AtomRule.make("age_group", "adult", r"\badults?\b|adult brain"),
    AtomRule.make("anesthesia", "unspecified", r"anestheti[sz]ed|anaestheti[sz]ed"),
    AtomRule.make("anesthesia", "pentobarbital", r"pentobarbital[- ]anestheti[sz]ed|pentobarbital[- ]anaestheti[sz]ed"),
    AtomRule.make("host_genotype", "abcb1_knockout", r"p[- ]?gp knockout|mdr1 knockout|abcb1 knockout|mdr1a/b\s*\(-/-\)"),
    AtomRule.make("formulation", "nanoparticle", r"\bnanoparticle|nanocarrier|nanosuspension"),
    AtomRule.make("formulation", "liposomal", r"\bliposom"),
    AtomRule.make("co_treatment", "tariquidar", r"\btariquidar\b"),
    AtomRule.make("co_treatment", "elacridar", r"\belacridar\b"),
    AtomRule.make("co_treatment", "cyclosporine", r"\bcyclosporin(?:e)?\b"),
    AtomRule.make("co_treatment", "probenecid", r"\bprobenecid\b"),
)

_SELF_CONDITION_SMILES = {
    ConditionAtom("co_treatment", "probenecid"): {
        "CCCN(CCC)S(=O)(=O)c1ccc(C(=O)O)cc1"
    },
    ConditionAtom("co_treatment", "tariquidar"): {
        "COc1cc2c(cc1OC)CN(CCc1ccc(NC(=O)c3cc(OC)c(OC)cc3NC(=O)c3cnc4ccccc4c3)cc1)CC2"
    },
}


def _suppress(atoms: set[ConditionAtom]) -> set[ConditionAtom]:
    output = set(atoms)
    for specific in ("bacterial_meningitis", "pneumococcal_meningitis", "tuberculous_meningitis"):
        if ConditionAtom("disease", specific) in output:
            output.discard(ConditionAtom("disease", "meningitis_unspecified"))
    if ConditionAtom("anesthesia", "pentobarbital") in output:
        output.discard(ConditionAtom("anesthesia", "unspecified"))
    return output


def _remove_non_conditions(
    atoms: tuple[ConditionAtom, ...], parent_smiles: str, condition_text: str
) -> tuple[ConditionAtom, ...]:
    output = []
    lowered = condition_text.lower()
    for atom in atoms:
        if parent_smiles in _SELF_CONDITION_SMILES.get(atom, set()):
            continue
        if atom.family == "co_treatment" and re.search(
            rf"\b(?:without|absence of)\s+{re.escape(atom.value)}\b|"
            rf"\bbefore (?:the )?(?:administration of )?{re.escape(atom.value)}\b|"
            rf"\bbaseline \(before (?:administration of )?{re.escape(atom.value)}\b",
            lowered,
        ):
            continue
        output.append(atom)
    return tuple(output)


def _load_source() -> pd.DataFrame:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    arrow = (
        cache_root
        / "datasets/starling-labs___bbb/default/0.0.0"
        / SOURCE_REVISION
        / "bbb-train.arrow"
    )
    if arrow.exists():
        return Dataset.from_file(str(arrow)).to_pandas()
    return load_dataset(SOURCE_DATASET, split="train", revision=SOURCE_REVISION).to_pandas()


def prepare_review_queue() -> dict[str, Any]:
    frame = _load_source()
    candidates = []
    source_rows = []
    proposal_audit = []
    for index, row in frame.iterrows():
        condition_text = str(row.get("qualifying_conditions") or "").strip()
        if not condition_text:
            continue
        record_id = f"starling-labs/BBB:row:{index}"
        proposal = propose_pattern_condition(
            condition_text,
            rules=RULES,
            incompatible_families={"route", "barrier_state", "age_group", "anesthesia", "formulation", "disease"},
            suppress=_suppress,
        )
        audit = {
            "source_record_id": record_id,
            "source_index": int(index),
            "condition_text": condition_text,
            "proposal_status": proposal.status,
            "proposal_reason": proposal.reason,
            "proposed_condition_group": proposal.signature or "",
        }
        if not proposal.atoms:
            proposal_audit.append({**audit, "queue_status": "not_queued"})
            continue
        label, method = label_record(
            row,
            source_index=int(index),
            allow_conditioned_context=True,
        )
        if label is None:
            proposal_audit.append(
                {
                    **audit,
                    "queue_status": "not_queued",
                    "proposal_reason": f"endpoint_ineligible:{method}",
                }
            )
            continue
        identity = attach_parent_identity({}, row.get("smiles"))
        atoms = _remove_non_conditions(
            proposal.atoms, str(identity.get("drug") or ""), condition_text
        )
        if not atoms:
            proposal_audit.append(
                {
                    **audit,
                    "queue_status": "not_queued",
                    "proposal_reason": "matched_material_is_absent_or_query_molecule",
                }
            )
            continue
        signature = "+".join(atom.key for atom in atoms)
        payload_fields = (
            PROPOSAL_VERSION,
            signature,
            int(label),
            record_id,
            row.get("smiles"),
            row.get("bbb_permeability_label"),
            row.get("quant_metric"),
            row.get("quant_value"),
            row.get("assay_model"),
            row.get("species"),
            condition_text,
            row.get("support_text"),
            row.get("extra_details"),
        )
        candidate = {
            **identity,
            **{
                "source_record_id": record_id,
                "source_payload_sha256": payload_sha256(payload_fields),
                "source_index": int(index),
                "pmid": str(row.get("pmid") or ""),
                "condition_text": condition_text,
                "support_text": str(row.get("support_text") or ""),
                "raw_value": " | ".join(
                    str(row.get(field) or "")
                    for field in ("bbb_permeability_label", "quant_metric", "quant_value", "quant_units")
                ),
                "assay_model": str(row.get("assay_model") or ""),
                "species": str(row.get("species") or ""),
                "extra_details": str(row.get("extra_details") or ""),
                "proposed_condition_group": signature,
                "proposed_condition_atoms": [atom.key for atom in atoms],
                "proposal_reason": proposal.reason,
                "Y": int(label),
                "label_method": method,
            },
        }
        if candidate["molecule_identity_key"]:
            candidates.append(candidate)
            source_rows.append({**row.to_dict(), "source_index": int(index)})
            proposal_audit.append(
                {**audit, "queue_status": "queued", "proposed_condition_group": signature}
            )
        else:
            proposal_audit.append(
                {**audit, "queue_status": "not_queued", "proposal_reason": "invalid_parent_identity"}
            )
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_parquet(FROZEN_CANDIDATE_SOURCE, index=False)
    return write_review_queue(
        rows=candidates,
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        manifest_path=REVIEW_ROOT / "review_queue_manifest.json",
        task_name="BBB_Martins",
        source_artifacts=(FROZEN_CANDIDATE_SOURCE,),
        proposal_version=PROPOSAL_VERSION,
        proposal_audit_rows=proposal_audit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = prepare_review_queue()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
