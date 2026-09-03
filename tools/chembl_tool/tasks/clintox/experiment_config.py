"""Paper-facing ClinTox retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    evidence_family,
)


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        evidence_family(
            "direct_clinical_toxicity",
            family_label="Tier 1",
            source_group_prefixes=("Tier 1.",),
            legacy_output_group_id="Direct.clinical_toxicity",
        ),
    ),
    mechanism_groups=tuple(
        evidence_family(
            {
                1: "clinical_human_safety",
                2: "in_vivo_toxicology",
                3: "organ_specific_toxicity",
                4: "genotoxicity_carcinogenicity",
                5: "cellular_stress_pathways",
                6: "general_cytotoxicity",
                7: "off_target_ddi_exposure",
            }[tier],
            family_label=f"Tier {tier}",
            source_group_prefixes=(f"Tier {tier}.",),
            legacy_output_group_id=f"Mechanism.tier_{tier}",
        )
        for tier in range(1, 8)
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        evidence_family(
            "clinical_trial_failure",
            family_label="Direct",
            source_group_ids=("Direct.clinical_trial_failure",),
        ),
    ),
    mechanism_groups=(
        evidence_family(
            "clinical_trial_failure",
            family_label="Direct",
            source_group_ids=("Direct.clinical_trial_failure",),
        ),
        evidence_family(
            "clinical_human_safety",
            family_label="Clinical",
            source_group_ids=("Mechanism.clinical_human_safety",),
            legacy_output_group_id="Clinical.clinical_human_safety",
        ),
        *(
            evidence_family(
                family,
                family_label="Mechanism",
                source_group_ids=(f"Mechanism.{family}",),
            )
            for family in (
                "in_vivo_toxicology",
                "organ_specific_toxicity",
                "genotoxicity_carcinogenicity",
                "cellular_stress_pathways",
                "general_cytotoxicity",
                "off_target_ddi_exposure",
            )
        ),
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
