"""Profile and transparently group Bioavailability qualifying conditions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.benchmark_dataset import has_reported_text
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    DIRECT_CLAIMS_PATH,
    DIRECT_SOURCE_ROWS_PATH,
)


ANALYSIS_VERSION = "bioavailability_qualifying_conditions.v1"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/qualifying_conditions_analysis_v1"
)

# Multi-label rules.  Ordering also defines the mutually exclusive primary group.
# More specific experimental modifiers precede broad mechanism/study-context rules.
CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "co_treatment_or_ddi",
        r"\bco[- ]?admin(?:istered|istration)?\b|\bcoadministration\b|\bconcomitant\b|"
        r"\bpretreat(?:ed|ment)?\b|\bdrug[- ]drug\b|\bcombination (?:therapy|treatment)\b|"
        r"\binteraction\b|\binhibitor\b|\binducer\b|\bgrapefruit\b|\bst\.? john'?s wort\b|"
        r"\bketoconazole\b|\bcimetidine\b|\brifamp(?:in|icin)\b|\britonavir\b|"
        r"\bcyclosporin(?:e)?\b|\bantacid\b|\bomeprazole\b|\bwith charcoal\b",
    ),
    (
        "food_or_prandial_state",
        r"\bfood\b|\bmeals?\b|\bfed\b|\bfeeding\b|\bunfed\b|\bstarved\b|"
        r"\bfast(?:ed|ing)?\b|\bnon[- ]?fast(?:ed|ing)\b|"
        r"\bpost[- ]?prandial\b|\bpre[- ]?prandial\b|\bempty stomach\b|\bbreakfast\b|"
        r"\bdiet(?:ary)?\b|\bcalori(?:e|c)\b|\bhigh[- ]fat\b|\blow[- ]fat\b",
    ),
    (
        "formulation_delivery_or_solid_state",
        r"\bformulat(?:ion|ions|ed)\b|\btablets?\b|\bcapsules?\b|\bsolution\b|\bsuspension\b|"
        r"\bemulsion\b|\bnano(?:particles?|carriers?|crystals?|suspension|emulsion)\b|"
        r"\bliposom(?:e|es|al)\b|\bmicell(?:e|es|ar)\b|\bsolid dispersion\b|\bvehicle\b|"
        r"\bcyclodextr(?:in|an)\b|\b(?:immediate|extended|sustained|controlled|prolonged|delayed|slow)[- ]release\b|"
        r"\benteric\b|\bcoated\b|\bdepot\b|\bpellet\b|\bgranule\b|\bpowder\b|"
        r"\bcrystal(?:line|lized|lised)?\b|\bamorphous\b|\bpolymorph\b|\bmicroni[sz]ed\b|"
        r"\bparticle size\b|\blipid[- ]based\b|\bself[- ]emulsif|\bencapsulat\w*\b|"
        r"\bmicrospheres?\b|\bhpmc\b|\btween\b|\bpeg ?\d*\b|\bcorn oil\b",
    ),
    (
        "molecular_form_salt_or_prodrug",
        r"\bpro[- ]?drug\b|\bsalts?\b|\bhydrochloride\b|\bhcl\b|\bfree base\b|\bfree acid\b|"
        r"\bsodium form\b|\bpotassium form\b|\bcalcium form\b|\bester\b|\baglycone\b|"
        r"\benantiomer\b|\bstereoisomer\b|\bactive metabolite\b|\bparent (?:drug|compound)\b|"
        r"\bfree (?:form|drug|parent)\b|\bunconjugated\b|\bconjugated form\b|\bracemic\b|"
        r"\badministered as\b",
    ),
    (
        "disease_organ_function_or_surgery",
        r"\bcirrhosis\b|\bhepatic (?:failure|impairment|disease|dysfunction)\b|"
        r"\bliver (?:failure|impairment|disease|dysfunction|transplant)\b|"
        r"\brenal (?:failure|impairment|disease|dysfunction|function)\b|"
        r"\bkidney (?:failure|impairment|disease|dysfunction|transplant)\b|"
        r"\bheart failure\b|\bcongestive heart\b|\bchf\b|\bcystic fibrosis\b|\bcf\b|"
        r"\bcrf\b|\bdiabet(?:es|ic)\b|\bt2dm\b|"
        r"\bcancer\b|\btumou?r\b|\bhiv\b|\baids\b|\bhypertension\b|\bobes(?:e|ity)\b|"
        r"\bmalabsorb\w*\b|\bgastrectomy\b|\bbariatric\b|\bbile duct\b|\bcannulat\w*\b|"
        r"\binflammatory bowel\b|\bcrohn'?s\b|\bulcerative colitis\b|\bdisease\b|"
        r"\btransplant\w*\b|\binsufficiency\b|\brheumatoid arthritis\b|\bshort bowel\b|"
        r"\bjejunoileal bypass\b|\bcritically ill\b|\bhyperthyroid\w*\b|\bmucositis\b|"
        r"\bspinal cord injury\b|\bpatients?\b",
    ),
    (
        "genotype_or_metabolizer_phenotype",
        r"\bgenotyp(?:e|ic)\b|\bphenotyp(?:e|ic)\b|\bpolymorph(?:ism|ic)\b|\ballele\b|"
        r"\bhaplotype\b|\bpoor metabolizers?\b|\bextensive metabolizers?\b|"
        r"\bintermediate metaboli[sz]ers?\b|\bultra[- ]?rapid metaboli[sz]ers?\b|"
        r"\bpoor metaboli[sz]ers?\b|\bextensive metaboli[sz]ers?\b|\bslow acetylators?\b|"
        r"\bwild[- ]type\b|\bknockout\b|\bmutant\b",
    ),
    (
        "demographic_or_physiologic_state",
        r"\bmale\b|\bfemale\b|\bmen\b|\bwomen\b|\belderly\b|\bgeriatric\b|"
        r"\bpediatric\b|\bpaediatric\b|\bchildren\b|\bchild\b|\badolescen\w*\b|"
        r"\binfants?\b|\bneonat\w*\b|\bpregnan\w*\b|\blactat\w*\b|"
        r"\bpostmenopausal\b|\bhealthy (?:subjects?|volunteers?|adults?)\b|"
        r"\bvolunteers?\b|\bjapanese\b|\bcaucasian\b|\basian\b|\bethnic\w*\b|"
        r"\baged? \d+\b|\bage group\b|\bold age\b|\bhealthy\b|\banestheti[sz]ed\b",
    ),
    (
        "dose_regimen_or_sampling_time",
        r"\bsingle dose\b|\bmultiple doses?\b|\brepeat(?:ed)? doses?\b|\bsteady[- ]state\b|"
        r"\bchronic\b|\bacute\b|\bdos(?:e|ed|ing)\b|\bmg(?:/| per )kg\b|"
        r"\bonce daily\b|\btwice daily\b|\bloading dose\b|\bday \d+\b|\bweek \d+\b|"
        r"\bhours? (?:after|before|post)\b|\bpost[- ]dose\b|\bpre[- ]dose\b|"
        r"\btreatment (?:duration|period)\b|\brepeated administration\b|\bsingle administration\b|"
        r"\bdaily\b|\bfor \d+ (?:days?|weeks?|months?)\b",
    ),
    (
        "administration_route_or_gi_environment",
        r"\boral administration\b|\bintragastric\b|\bgavage\b|\bintravenous\b|"
        r"\bsubcutaneous\b|\bintramuscular\b|\broute of administration\b|"
        r"\bgastric ph\b|\bintestinal ph\b|\bachlorhydria\b|\bhypochlorhydria\b|"
        r"\bgastrointestinal tract\b|\bgut lumen\b",
    ),
    (
        "first_pass_metabolism_or_clearance",
        r"\bfirst[- ]pass\b|\bpresystemic\b|\bpre[- ]systemic\b|\bhepatic extraction\b|"
        r"\bgut[- ]wall metabolism\b|\bintestinal metabolism\b|\bhepatic metabolism\b|"
        r"\bmetabolic clearance\b|\bhigh clearance\b|\brapid clearance\b|"
        r"\bextensive metabolism\b|\bmetaboli[sz]ed extensively\b|\bhepatic clearance\b|"
        r"\bclearance\b|\belimination\b|\bmetabolism\b",
    ),
    (
        "absorption_solubility_or_permeability",
        r"\bsolubility\b|\bsoluble\b|\binsoluble\b|\babsorption\b|\babsorbed\b|"
        r"\bpermeability\b|\bpermeable\b|\bdissolution\b|\bprecipitat\w*\b|"
        r"\befflux\b|\bp[- ]?gp\b|\bintestinal transport\b|\bdegradation\b|"
        r"\bchemical stability\b|\bacid labile\b|\blipophil\w*\b",
    ),
    (
        "study_arm_comparator_or_model_context",
        r"\bcontrol\b|\bplacebo\b|\bbaseline\b|\badministered alone\b|\bgiven alone\b|\balone\b|"
        r"\bmonotherapy\b|\bcrossover\b|\brandomi[sz]ed\b|\bphase [i1-4v]+\b|"
        r"\bpredicted\b|\bsimulat(?:ed|ion)\b|\bmodel(?:led|ed|ing)?\b|"
        r"\bpopulation pharmacokinetic\b|\bnormal (?:group|condition|subjects?)\b|"
        r"\bcalculated\b|\bestimated\b|\btotal drug\b|\bunbound drug\b|\bgroup\b",
    ),
    (
        "processing_storage_or_environment",
        r"\bstorage\b|\bstored\b|\btemperature\b|\bcook(?:ed|ing)?\b|\bboil(?:ed|ing)?\b|"
        r"\bheating\b|\bprocessing\b|\blight exposure\b|\bhumidity\b|\bpreformed\b|"
        r"\bsoils?\b|\bwater temperature\b",
    ),
)

FIELD_INTERPRETATION = {
    "co_treatment_or_ddi": "true_contextual_modifier",
    "food_or_prandial_state": "true_contextual_modifier",
    "formulation_delivery_or_solid_state": "true_contextual_modifier",
    "molecular_form_salt_or_prodrug": "true_contextual_modifier",
    "disease_organ_function_or_surgery": "true_contextual_modifier",
    "genotype_or_metabolizer_phenotype": "true_contextual_modifier",
    "demographic_or_physiologic_state": "true_contextual_modifier",
    "dose_regimen_or_sampling_time": "true_contextual_modifier",
    "administration_route_or_gi_environment": "true_contextual_modifier",
    "processing_storage_or_environment": "true_contextual_modifier",
    "first_pass_metabolism_or_clearance": "mechanism_or_explanation_not_condition",
    "absorption_solubility_or_permeability": "mechanism_or_explanation_not_condition",
    "study_arm_comparator_or_model_context": "study_metadata_or_nonexperimental_context",
    "other_unresolved": "unresolved",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    claims_path = Path(args.claims)
    source_path = Path(args.direct_source_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = pd.read_parquet(claims_path, columns=["canonical_claim_id", "qualifying_conditions"])
    source_rows = pd.read_parquet(source_path, columns=["source_record_id", "qualifying_conditions"])
    claim_profile, claim_values = profile_frame(claims, id_field="canonical_claim_id")
    source_profile, source_values = profile_frame(source_rows, id_field="source_record_id")

    group_summary = summarize_primary_groups(claim_values, source_values)
    multilabel_summary = summarize_multilabel_groups(claim_values, source_values)
    cooccurrence = summarize_cooccurrence(claim_values)
    examples = representative_examples(claim_values)

    paths = {
        "condition_value_counts": output_dir / "condition_value_counts.csv",
        "primary_group_counts": output_dir / "primary_group_counts.csv",
        "multilabel_group_counts": output_dir / "multilabel_group_counts.csv",
        "group_cooccurrence": output_dir / "group_cooccurrence.csv",
        "representative_examples": output_dir / "representative_examples.csv",
    }
    claim_values.to_csv(paths["condition_value_counts"], index=False)
    group_summary.to_csv(paths["primary_group_counts"], index=False)
    multilabel_summary.to_csv(paths["multilabel_group_counts"], index=False)
    cooccurrence.to_csv(paths["group_cooccurrence"], index=False)
    examples.to_csv(paths["representative_examples"], index=False)

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_grain": "canonical claim after cross-source deduplication",
        "classification": {
            "method": "transparent case-insensitive multi-label regex taxonomy",
            "primary_group_rule": "first matching category in declared priority order",
            "other_group": "other_unresolved",
            "category_priority": [name for name, _ in CATEGORY_PATTERNS] + ["other_unresolved"],
            "field_interpretation": FIELD_INTERPRETATION,
            "not_a_causal_model": True,
        },
        "canonical_claims": claim_profile,
        "pre_dedup_direct_source_rows": source_profile,
        "quality_findings": {
            "nonempty_exact_values_occurring_once": int(
                (claim_values["canonical_claim_record_count"] == 1).sum()
            ),
            "singleton_record_share_of_nonempty": _ratio(
                int((claim_values["canonical_claim_record_count"] == 1).sum()),
                claim_profile["n_nonempty"],
            ),
            "top_10_exact_value_record_share_of_nonempty": _ratio(
                int(claim_values.head(10)["canonical_claim_record_count"].sum()),
                claim_profile["n_nonempty"],
            ),
            "top_100_exact_value_record_share_of_nonempty": _ratio(
                int(claim_values.head(100)["canonical_claim_record_count"].sum()),
                claim_profile["n_nonempty"],
            ),
            "multi_label_claim_records": int(
                claim_values.loc[
                    claim_values["n_matched_groups"] > 1, "canonical_claim_record_count"
                ].sum()
            ),
            "multi_label_claim_record_share_of_nonempty": _ratio(
                int(
                    claim_values.loc[
                        claim_values["n_matched_groups"] > 1,
                        "canonical_claim_record_count",
                    ].sum()
                ),
                claim_profile["n_nonempty"],
            ),
        },
        "inputs": {
            "canonical_claims": {
                "path": str(claims_path),
                "sha256": _sha256_file(claims_path),
            },
            "direct_source_rows": {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
            },
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def profile_frame(frame: pd.DataFrame, *, id_field: str) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = frame["qualifying_conditions"]
    nonempty_mask = raw.map(has_reported_text)
    values = raw.loc[nonempty_mask].astype(str).str.strip()
    counts = values.value_counts(dropna=False).rename("record_count").reset_index()
    counts.columns = ["qualifying_conditions", "record_count"]
    counts["normalized_condition"] = counts["qualifying_conditions"].map(normalize_condition)
    classifications = counts["normalized_condition"].map(classify_condition)
    counts["primary_group"] = classifications.map(lambda item: item[0])
    counts["matched_groups"] = classifications.map(lambda item: "|".join(item[1]))
    counts["n_matched_groups"] = classifications.map(lambda item: len(item[1]))
    counts["field_interpretation"] = counts["primary_group"].map(FIELD_INTERPRETATION)
    count_name = "canonical_claim_record_count" if id_field == "canonical_claim_id" else "source_record_count"
    counts = counts.rename(columns={"record_count": count_name})
    profile = {
        "n_rows": len(frame),
        "n_nonempty": int(nonempty_mask.sum()),
        "nonempty_rate": _ratio(int(nonempty_mask.sum()), len(frame)),
        "n_empty_or_null": int((~nonempty_mask).sum()),
        "n_distinct_exact_nonempty": int(values.nunique()),
        "n_distinct_normalized_nonempty": int(counts["normalized_condition"].nunique()),
        "id_field": id_field,
    }
    return profile, counts.sort_values(count_name, ascending=False, kind="stable").reset_index(drop=True)


def normalize_condition(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.strip(" .;,")


def classify_condition(text: str) -> tuple[str, list[str]]:
    matches = [name for name, pattern in CATEGORY_PATTERNS if re.search(pattern, text, re.IGNORECASE)]
    if not matches:
        return "other_unresolved", ["other_unresolved"]
    return matches[0], matches


def summarize_primary_groups(claims: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    claim_counts = _group_counts(claims, "primary_group", "canonical_claim_record_count")
    source_counts = _group_counts(sources, "primary_group", "source_record_count")
    output = claim_counts.merge(source_counts, on="group", how="outer").fillna(0)
    total_claims = int(claims["canonical_claim_record_count"].sum())
    total_sources = int(sources["source_record_count"].sum())
    output["canonical_claim_share_nonempty"] = output["canonical_claim_records"] / total_claims
    output["source_record_share_nonempty"] = output["source_records"] / total_sources
    output["field_interpretation"] = output["group"].map(FIELD_INTERPRETATION)
    return output.sort_values("canonical_claim_records", ascending=False, kind="stable").reset_index(drop=True)


def summarize_multilabel_groups(claims: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    groups = [name for name, _ in CATEGORY_PATTERNS] + ["other_unresolved"]
    rows = []
    for group in groups:
        claim_mask = claims["matched_groups"].str.split("|").map(lambda values: group in values)
        source_mask = sources["matched_groups"].str.split("|").map(lambda values: group in values)
        rows.append(
            {
                "group": group,
                "canonical_claim_records": int(
                    claims.loc[claim_mask, "canonical_claim_record_count"].sum()
                ),
                "canonical_claim_exact_values": int(claim_mask.sum()),
                "source_records": int(sources.loc[source_mask, "source_record_count"].sum()),
                "source_exact_values": int(source_mask.sum()),
                "field_interpretation": FIELD_INTERPRETATION[group],
            }
        )
    output = pd.DataFrame(rows)
    total_claims = int(claims["canonical_claim_record_count"].sum())
    output["canonical_claim_share_nonempty"] = output["canonical_claim_records"] / total_claims
    return output.sort_values("canonical_claim_records", ascending=False, kind="stable").reset_index(drop=True)


def summarize_cooccurrence(claims: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[tuple[str, str]] = Counter()
    for row in claims.to_dict(orient="records"):
        groups = str(row["matched_groups"]).split("|")
        weight = int(row["canonical_claim_record_count"])
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                counter[tuple(sorted((left, right)))] += weight
    return pd.DataFrame(
        [
            {"group_a": pair[0], "group_b": pair[1], "canonical_claim_records": count}
            for pair, count in counter.most_common()
        ]
    )


def representative_examples(claims: pd.DataFrame, *, per_group: int = 12) -> pd.DataFrame:
    rows = []
    for group, group_rows in claims.groupby("primary_group", sort=False):
        for rank, row in enumerate(group_rows.head(per_group).to_dict(orient="records"), start=1):
            rows.append(
                {
                    "group": group,
                    "rank_within_group": rank,
                    "qualifying_conditions": row["qualifying_conditions"],
                    "canonical_claim_records": row["canonical_claim_record_count"],
                    "matched_groups": row["matched_groups"],
                }
            )
    return pd.DataFrame(rows)


def _group_counts(frame: pd.DataFrame, field: str, count_field: str) -> pd.DataFrame:
    return (
        frame.groupby(field, as_index=False)
        .agg(**{
            "records": (count_field, "sum"),
            "exact_values": ("qualifying_conditions", "size"),
            "normalized_values": ("normalized_condition", "nunique"),
        })
        .rename(
            columns={
                field: "group",
                "records": "canonical_claim_records" if count_field.startswith("canonical") else "source_records",
                "exact_values": "canonical_claim_exact_values" if count_field.startswith("canonical") else "source_exact_values",
                "normalized_values": "canonical_claim_normalized_values" if count_field.startswith("canonical") else "source_normalized_values",
            }
        )
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", default=str(DIRECT_CLAIMS_PATH))
    parser.add_argument("--direct-source-rows", default=str(DIRECT_SOURCE_ROWS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
