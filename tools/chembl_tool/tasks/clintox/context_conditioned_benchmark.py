"""Build a reviewed AACT-source ClinTox context-conditioned extension."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.condition_review import (
    attach_parent_identity,
    merge_terminal_verdicts,
    write_review_queue,
)
from tools.chembl_tool.common.starling.external_condition import (
    AtomRule,
    payload_sha256,
    propose_pattern_condition,
)
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    build_reviewed_conditioned_benchmark,
)
from tools.chembl_tool.tasks.clintox.clinical_trial_failure_benchmark import (
    AACT_SOURCE_PATH,
    SOURCE_ROOT,
)


PROPOSAL_VERSION = "clintox_aact_external_condition_proposal.v3"
LINEAGE = "clintox_clinical_trial_failure_context_conditioned_v1"
FROZEN_ROOT = Path("data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold")
AACT_ARCHIVE = SOURCE_ROOT / "AACT201603_pipe_delimited.tar.gz"
SMILES_CACHE = SOURCE_ROOT / "smiles_cache.csv.gz"
REVIEW_ROOT = Path("data/starling_data/clintox/context_conditioned_aact_review_v1")
FROZEN_CANDIDATE_SOURCE = REVIEW_ROOT / "frozen_candidate_source.parquet"
OUTPUT_ROOT = Path("data/processed_starling_context_conditioned_reviewed_v1/ClinTox/scaffold")


RULES = (
    AtomRule.make("route", "oral", r"\boral(?:ly)?\b|\bby mouth\b"),
    AtomRule.make("route", "intravenous", r"\bintravenous\b|(?<!stage )(?<!phase )\bi\.?v\.?\b|\binfusion\b"),
    AtomRule.make("route", "subcutaneous", r"\bsubcutaneous\b"),
    AtomRule.make("route", "inhaled", r"\binhal(?:ed|ation)\b"),
    AtomRule.make("disease", "nsclc", r"non[- ]?small[- ]?cell lung|\bnsclc\b"),
    AtomRule.make("disease", "small_cell_lung_cancer", r"(?<!non[- ])small[- ]?cell lung|\bsclc\b"),
    AtomRule.make("disease", "breast_cancer", r"breast (?:cancer|carcinoma)"),
    AtomRule.make("disease", "ovarian_cancer", r"ovarian (?:cancer|carcinoma)"),
    AtomRule.make("disease", "prostate_cancer", r"prostate (?:cancer|carcinoma)"),
    AtomRule.make("disease", "colorectal_cancer", r"colorectal|colon cancer|rectal cancer"),
    AtomRule.make("disease", "glioma", r"\bglioma\b|glioblastoma"),
    AtomRule.make("disease", "lymphoma", r"\blymphoma\b"),
    AtomRule.make("disease", "leukemia", r"leuk[ae]mia"),
    AtomRule.make("disease", "melanoma", r"\bmelanoma\b"),
    AtomRule.make("disease", "renal_cell_carcinoma", r"renal[- ]cell carcinoma|kidney cancer"),
    AtomRule.make("disease", "cervical_cancer", r"cervical (?:cancer|carcinoma)"),
    AtomRule.make("disease", "urothelial_cancer", r"urothelial|bladder (?:cancer|carcinoma)"),
    AtomRule.make("disease", "soft_tissue_sarcoma", r"soft[- ]tissue sarcoma"),
    AtomRule.make("disease", "advanced_solid_tumor_unspecified", r"advanced.{0,30}solid tumou?r|advanced solid malign"),
    AtomRule.make("disease", "type_2_diabetes", r"type[- ]?2 diabet|\bt2dm\b"),
    AtomRule.make("disease", "hiv", r"\bhiv\b|\baids\b"),
    AtomRule.make("disease", "hepatitis_b", r"hepatitis b|\bhbv\b"),
    AtomRule.make("disease", "hepatitis_c", r"hepatitis c|\bhcv\b"),
    AtomRule.make("disease", "crohns_disease", r"crohn'?s disease"),
    AtomRule.make("disease", "parkinsons_disease", r"parkinson'?s? disease"),
    AtomRule.make("disease", "alzheimers_disease", r"alzheimer'?s? disease"),
    AtomRule.make("population", "healthy_volunteers", r"healthy (?:human )?(?:volunteers?|subjects?|participants?)"),
    AtomRule.make("age_group", "elderly", r"\belderly\b|older adults|aged 65|geriatric"),
    AtomRule.make("age_group", "pediatric", r"p[ae]diatric|\bchildren\b|adolescen"),
    AtomRule.make("transplant", "liver", r"liver transplant"),
    AtomRule.make("release_profile", "modified_release", r"extended[- ]release|delayed[- ]release|controlled[- ]release|sustained[- ]release"),
    AtomRule.make("formulation", "liposomal", r"\bliposom"),
)

TOXICITY_CANDIDATE = re.compile(
    r"toxic|safety|adverse|side effect|serious event|death|mortality|morbidity|risk|tolerability",
    re.IGNORECASE,
)
FAILED_STATUS = {"terminated", "suspended", "withdrawn"}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _read_archive_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv.field_size_limit(20_000_000)
    with tarfile.open(AACT_ARCHIVE) as archive:
        interventions = pd.read_csv(
            archive.extractfile("AACT201603_pipe_delimited/interventions.txt"),
            sep="|", dtype=str, keep_default_na=False,
        )
        other_names = pd.read_csv(
            archive.extractfile("AACT201603_pipe_delimited/intervention_other_names.txt"),
            sep="|", dtype=str, keep_default_na=False,
        )
        stream = io.TextIOWrapper(
            archive.extractfile("AACT201603_pipe_delimited/clinical_study_noclob.txt"),
            encoding="utf-8", errors="replace", newline="",
        )
        reader = csv.reader(stream, delimiter="|")
        header = next(reader)
        fields = (
            "NCT_ID", "BRIEF_TITLE", "OFFICIAL_TITLE", "OVERALL_STATUS", "WHY_STOPPED",
            "PHASE", "GENDER", "MINIMUM_AGE", "MAXIMUM_AGE", "HEALTHY_VOLUNTEERS", "STUDY_POP",
        )
        positions = {field: header.index(field) for field in fields}
        studies = pd.DataFrame(
            ({field: row[index] if index < len(row) else "" for field, index in positions.items()} for row in reader)
        )
    return interventions, other_names, studies


def _source_trial_links() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_csv(SMILES_CACHE, header=None, names=("source_name", "smiles"))
    positives = pd.read_csv(AACT_SOURCE_PATH)
    if set(cache["smiles"]) != set(positives["smiles"]):
        raise RuntimeError("SMILES cache does not reconcile to frozen aacttox positives")
    cache["normalized_name"] = cache["source_name"].map(_norm)
    interventions, other_names, studies = _read_archive_tables()
    aliases = pd.concat(
        [
            interventions[["INTERVENTION_ID", "NCT_ID", "INTERVENTION_NAME"]].assign(
                alias=lambda frame: frame["INTERVENTION_NAME"]
            ),
            other_names[["INTERVENTION_ID", "NCT_ID", "OTHER_NAME"]]
            .rename(columns={"OTHER_NAME": "alias"})
            .merge(
                interventions[["INTERVENTION_ID", "NCT_ID", "INTERVENTION_NAME"]],
                on=["INTERVENTION_ID", "NCT_ID"], how="left",
            ),
        ],
        ignore_index=True,
    )
    aliases["normalized_name"] = aliases["alias"].map(_norm)
    links = cache.merge(aliases, on="normalized_name", how="left").merge(studies, on="NCT_ID", how="left")
    # One frozen positive structure in one AACT trial is one source claim even
    # when the same intervention has multiple aliases or arm-local rows.
    links = links.drop_duplicates(["smiles", "NCT_ID"])
    interventions_by_trial = (
        interventions[interventions["INTERVENTION_TYPE"].str.lower() == "drug"]
        .groupby("NCT_ID")["INTERVENTION_NAME"].agg(list).to_dict()
    )
    links["all_drug_interventions"] = links["NCT_ID"].map(interventions_by_trial).map(
        lambda value: value if isinstance(value, list) else []
    )
    return links, cache


def prepare_review_queue() -> dict[str, Any]:
    links, cache = _source_trial_links()
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    source_rows, proposal_audit = [], []
    for _, row in links.iterrows():
        record_id = f"aacttox_trial:{_slug(row.get('source_name'))}:{row.get('NCT_ID')}"
        why_stopped = str(row.get("WHY_STOPPED") or "")
        status = str(row.get("OVERALL_STATUS") or "").lower()
        audit = {
            "source_record_id": record_id,
            "source_name": str(row.get("source_name") or ""),
            "nct_id": str(row.get("NCT_ID") or ""),
            "overall_status": str(row.get("OVERALL_STATUS") or ""),
            "why_stopped": why_stopped,
        }
        if not row.get("NCT_ID") or status not in FAILED_STATUS or not TOXICITY_CANDIDATE.search(why_stopped):
            proposal_audit.append({**audit, "queue_status": "not_queued", "proposal_reason": "not_failed_trial_with_toxicity_candidate_reason"})
            continue
        context_fields = (
            row.get("BRIEF_TITLE"), row.get("OFFICIAL_TITLE"), row.get("STUDY_POP"),
            row.get("INTERVENTION_NAME"), " + ".join(row.get("all_drug_interventions") or []),
            row.get("PHASE"), row.get("GENDER"), row.get("MINIMUM_AGE"), row.get("MAXIMUM_AGE"),
            row.get("HEALTHY_VOLUNTEERS"), why_stopped,
        )
        context = " | ".join(str(value or "") for value in context_fields)
        proposal = propose_pattern_condition(context, rules=RULES)
        # The frozen AACT extract has no arm mapping.  Other interventions are
        # shown to the reviewer but must not be inferred to be co-treatments.
        # Explicit combinations that this finite ontology cannot represent are
        # rejected during semantic review instead of collapsed into disease-only.
        atoms = sorted(proposal.atoms)
        if not atoms:
            proposal_audit.append({**audit, "queue_status": "not_queued", "proposal_reason": "no_supported_external_condition"})
            continue
        signature = "+".join(atom.key for atom in atoms)
        candidate = attach_parent_identity(
            {
                "source_record_id": record_id,
                "source_payload_sha256": payload_sha256(
                    (PROPOSAL_VERSION, signature, 1, record_id, row.get("smiles"), *context_fields)
                ),
                "source_index": -1,
                "nct_id": str(row.get("NCT_ID") or ""),
                "pmid": "",
                "molecule_name": str(row.get("source_name") or ""),
                "condition_text": context,
                "support_text": f"AACT status={row.get('OVERALL_STATUS')}; why_stopped={why_stopped}",
                "raw_value": why_stopped,
                "proposed_condition_group": signature,
                "proposed_condition_atoms": [atom.key for atom in atoms],
                "proposal_reason": "aact_trial_metadata_condition_proposal",
                "Y": 1,
                "label_method": "frozen_aacttox_positive_trial_link",
            },
            row.get("smiles"),
        )
        if candidate["molecule_identity_key"]:
            parent_trial = (candidate["molecule_identity_key"], candidate["nct_id"])
            if parent_trial in candidates_by_key:
                proposal_audit.append(
                    {
                        **audit,
                        "queue_status": "not_queued",
                        "proposal_reason": "duplicate_parent_trial_alias",
                        "proposed_condition_group": signature,
                    }
                )
                continue
            canonical_id = f"aacttox_trial:{candidate['nct_id']}:{candidate['molecule_identity_key']}"
            candidate["source_record_id"] = canonical_id
            candidate["source_payload_sha256"] = payload_sha256(
                (PROPOSAL_VERSION, signature, 1, canonical_id, row.get("smiles"), *context_fields)
            )
            candidates_by_key[parent_trial] = candidate
            source_rows.append(row.to_dict())
            proposal_audit.append(
                {
                    **audit,
                    "source_record_id": canonical_id,
                    "queue_status": "queued",
                    "proposed_condition_group": signature,
                }
            )
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_parquet(FROZEN_CANDIDATE_SOURCE, index=False)
    return write_review_queue(
        rows=list(candidates_by_key.values()),
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        manifest_path=REVIEW_ROOT / "review_queue_manifest.json",
        task_name="ClinTox",
        source_artifacts=(AACT_SOURCE_PATH, AACT_ARCHIVE, SMILES_CACHE, FROZEN_CANDIDATE_SOURCE),
        proposal_version=PROPOSAL_VERSION,
        proposal_audit_rows=proposal_audit,
    )


def build() -> dict[str, Any]:
    reviewed = merge_terminal_verdicts(
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        verdict_path=REVIEW_ROOT / "review_verdicts.jsonl",
    )
    return build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="ClinTox", lineage=LINEAGE,
            protocol_version="clintox_context_conditioned_benchmark.v1",
            frozen_root=FROZEN_ROOT, output_root=OUTPUT_ROOT,
            source_artifacts=(
                AACT_SOURCE_PATH,
                AACT_ARCHIVE,
                SMILES_CACHE,
                FROZEN_CANDIDATE_SOURCE,
                REVIEW_ROOT / "review_queue.jsonl",
                REVIEW_ROOT / "review_queue_manifest.json",
                REVIEW_ROOT / "model_prereview.jsonl",
                REVIEW_ROOT / "model_prereview_round2.jsonl",
                REVIEW_ROOT / "review_comparison_summary.json",
                REVIEW_ROOT / "review_verdicts.jsonl",
            ),
            row_id_prefix="CLINCTX", frozen_lineage="clinical_trial_failure_v1",
        ),
        review_rows=reviewed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-review", "build"))
    args = parser.parse_args(argv)
    result = prepare_review_queue() if args.action == "prepare-review" else build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
