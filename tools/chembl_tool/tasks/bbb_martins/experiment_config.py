"""Paper-facing BBB retrieval views."""

from tools.chembl_tool.common.experiment_retrieval import EvidenceGroupSpec, SourceExperimentConfig


CHEMBL = SourceExperimentConfig(
    source_name="chembl",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        EvidenceGroupSpec(
            f"Mechanism.tier_{tier}",
            f"Tier {tier}",
            {
                1: "direct_brain_exposure",
                2: "passive_permeability",
                3: "efflux_transport",
                4: "influx_transport",
            }[tier],
            source_group_prefixes=(f"Tier {tier}.",),
        )
        for tier in range(1, 5)
    ),
)

STARLING = SourceExperimentConfig(
    source_name="starling",
    direct_groups=(
        EvidenceGroupSpec(
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
    ),
    mechanism_groups=(),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
