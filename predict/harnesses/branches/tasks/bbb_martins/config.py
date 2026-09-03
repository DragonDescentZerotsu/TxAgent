"""Paper-facing BBB retrieval views."""

from predict.harnesses.branches.retrieval import BranchDefinition, BranchRetrievalConfig


CHEMBL = BranchRetrievalConfig(
    source_name="chembl",
    direct_groups=(
        BranchDefinition(
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        BranchDefinition(
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

STARLING = BranchRetrievalConfig(
    source_name="starling",
    direct_groups=(
        BranchDefinition(
            "Direct.bbb",
            "Tier 1",
            "direct_bbb",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
    ),
    mechanism_groups=(
        BranchDefinition(
            "Mechanism.tier_1",
            "Tier 1",
            "direct_brain_exposure",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
        BranchDefinition(
            "Mechanism.tier_2",
            "Tier 2",
            "passive_permeability",
            source_groups=("Mechanism.passive_permeability",),
        ),
        BranchDefinition(
            "Mechanism.tier_3",
            "Tier 3",
            "efflux_transport",
            source_groups=("Mechanism.efflux_transport",),
        ),
        BranchDefinition(
            "Mechanism.tier_4",
            "Tier 4",
            "influx_transport",
            source_groups=("Mechanism.influx_transport",),
        ),
    ),
)

# Parallel retrieval-only source-purity view.  Keep ``STARLING`` unchanged so
# historical four-level artifacts remain exactly reproducible.
STARLING_SOURCE_PURITY = BranchRetrievalConfig(
    source_name="starling_source_purity",
    direct_groups=STARLING.direct_groups,
    mechanism_groups=(
        BranchDefinition(
            "Direct.measured_cns_access",
            "Direct measured CNS access",
            "direct_brain_exposure",
            source_groups=("Tier 1.starling_direct_bbb_evidence",),
        ),
        BranchDefinition(
            "Proxy.central_functional_access",
            "Near-direct functional CNS proxy",
            "central_functional_access_proxy",
            source_groups=("Proxy.central_functional_access",),
        ),
        BranchDefinition(
            "Mechanism.passive_permeability",
            "Passive permeability",
            "passive_permeability",
            source_groups=("Mechanism.passive_permeability",),
        ),
        BranchDefinition(
            "Mechanism.efflux_transport",
            "Efflux transport",
            "efflux_transport",
            source_groups=("Mechanism.efflux_transport",),
        ),
        BranchDefinition(
            "Mechanism.influx_transport",
            "Influx transport",
            "influx_transport",
            source_groups=("Mechanism.influx_transport",),
        ),
    ),
)

SOURCES = {
    "chembl": CHEMBL,
    "starling": STARLING,
    "starling_source_purity": STARLING_SOURCE_PURITY,
}

def get_source_config(source: str) -> BranchRetrievalConfig:
    return SOURCES[source]
