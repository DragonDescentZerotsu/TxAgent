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
    mechanism_groups=(
        EvidenceGroupSpec(
            "Mechanism.tier_1",
            "Tier 1",
            "direct_brain_exposure",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_2",
            "Tier 2",
            "passive_permeability",
            source_groups=("Mechanism.passive_permeability",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_3",
            "Tier 3",
            "efflux_transport",
            source_groups=("Mechanism.efflux_transport",),
        ),
        EvidenceGroupSpec(
            "Mechanism.tier_4",
            "Tier 4",
            "influx_transport",
            source_groups=("Mechanism.influx_transport",),
        ),
    ),
)

# The v6 normalized index preserves the same four paper-facing evidence
# families. A distinct source name makes its provenance explicit without
# changing the historical ``starling`` runtime or any default.
STARLING_V6 = SourceExperimentConfig(
    source_name="starling_v6",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=STARLING.mechanism_groups,
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING, "starling_v6": STARLING_V6}


def get_source_config(source: str) -> SourceExperimentConfig:
    return SOURCES[source]
