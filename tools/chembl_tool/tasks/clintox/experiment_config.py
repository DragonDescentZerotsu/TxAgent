"""Paper-facing ClinTox retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.clinical_toxicity",
            "Tier 1",
            "direct_clinical_toxicity",
            source_group_prefixes=("Tier 1.",),
        ),
    ),
    mechanism_groups=tuple(
        EvidenceGroupSpec(
            f"Mechanism.tier_{tier}",
            f"Tier {tier}",
            {
                1: "clinical_human_safety",
                2: "in_vivo_toxicology",
                3: "organ_specific_toxicity",
                4: "genotoxicity_carcinogenicity",
                5: "cellular_stress_pathways",
                6: "general_cytotoxicity",
                7: "off_target_ddi_exposure",
            }[tier],
            source_group_prefixes=(f"Tier {tier}.",),
        )
        for tier in range(1, 8)
    ),
)

STARLING_RAW = SourceExperimentConfig(
    source_name="starling_raw",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.human_organ_toxicity",
            "Direct",
            "clinical_human_safety",
            source_groups=("Direct.human_organ_toxicity",),
        ),
    ),
    mechanism_groups=(
        EvidenceGroupSpec(
            "Mechanism.clinical_human_safety",
            "Mechanism",
            "clinical_human_safety",
            source_groups=(
                "Direct.human_organ_toxicity",
                "Context.human_organ_toxicity",
            ),
        ),
        *(
            EvidenceGroupSpec(
                f"Mechanism.{family}",
                "Mechanism",
                family,
                source_groups=(f"Mechanism.{family}",),
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

SOURCES = {"chembl": CHEMBL, "starling_raw": STARLING_RAW}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
