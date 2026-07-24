"""BBB-specific ChEMBL assay mapping for the H1/H2 extension families."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from tools.chembl_tool.common.task_workflows.distance_assay_manifest import DistanceAssayDecision
from tools.chembl_tool.common.text import join_text_parts, normalize_text

from .endpoint_groups import EndpointAssignment


PXR_CAR_TARGETS = {
    "CHEMBL3401",  # human PXR / NR1I2
    "CHEMBL5503",  # human CAR / NR1I3
    "CHEMBL1743244",  # mouse PXR
    "CHEMBL2146315",  # rat PXR
    "CHEMBL3069",  # mouse CAR
    "CHEMBL3509594",  # rat CAR
}

HUMAN_MMP9_TARGETS = {
    "CHEMBL321",  # human MMP-9
    "CHEMBL3885505",  # human MMP-2/MMP-9
    "CHEMBL4523972",  # human MMP-2/MMP-9
}

HUMAN_MMP3_TARGETS = {
    "CHEMBL283",  # human stromelysin-1 / MMP-3
}

HUMAN_NRF2_TARGETS = {
    "CHEMBL1075094",  # human NRF2 / NFE2L2
}

HUMAN_KEAP1_NRF2_TARGETS = {
    "CHEMBL3038498",  # human KEAP1-NRF2 protein-protein interaction
}

HUMAN_HIF1_TARGETS = {
    "CHEMBL4261",  # human HIF-1alpha
    "CHEMBL2221345",  # human HIF-1alpha/HIF-2alpha family
    "CHEMBL5909",  # human HIF-1alpha inhibitor target
}

HUMAN_PHD2_TARGETS = {
    "CHEMBL5697",  # human EGLN1 / PHD2
}

BBB_BARRIER_MODELS = (
    "hcmec d3",
    "human brain microvascular endothelial",
    "human brain capillary endothelial",
    "brain microvascular endothelial",
    "brain capillary endothelial",
    "brain endothelial",
    "bbmec",
    "bmec",
    "rbec",
    "bend 3",
)

TRANSPORTABLE_BARRIER_MODELS = (
    "caco 2",
    "caco2",
    "mdck",
    "transwell",
)

TIGHT_JUNCTION_READOUTS = (
    "teer",
    "transepithelial electrical resistance",
    "transendothelial electrical resistance",
    "tight junction opening",
    "tight junction integrity",
    "barrier integrity",
    "zo 1 staining",
    "zonula occludens",
    "occludin integrity",
    "occludin expression",
    "claudin 1 mediated tight junction",
    "claudin 5",
    "paracellular marker",
    "fitc dextran",
)

PXR_CAR_FUNCTIONAL_TERMS = (
    "agonist activity",
    "agonism",
    "activation of",
    "activator of",
    "activators of",
    "transactivation",
    "reporter gene",
    "reporter assay",
    "luciferase",
)

PXR_CAR_EXCLUDED_TERMS = (
    "binding affinity",
    "binding to",
    "displacement",
    "antagonist",
    "inverse agonist",
    "inhibition of",
    "counterscreen",
)

MMP9_FUNCTIONAL_TERMS = (
    "inhibition of mmp9",
    "inhibition of mmp 9",
    "inhibition of matrix metalloproteinase 9",
    "matrix metalloprotease 9 enzyme inhibition",
    "matrix metalloproteinase 9 enzyme inhibition",
    "mmp9 activity",
    "mmp 9 activity",
    "gelatinase b activity",
)

MMP9_BINDING_ONLY_TERMS = (
    "binding affinity",
    "binding to",
    "displacement",
    "dissociation constant",
)

ASSAY_QUALITY_EXCLUSION_TERMS = (
    "insoluble",
    "not tested",
    "not determined",
)

MMP9_INDIRECT_READOUT_TERMS = (
    "cell invasion",
    "cell migration",
    "wound healing",
    "mrna expression",
    "gene expression",
    "cell viability",
    "tumor growth",
)

MMP9_DIRECT_METHOD_TERMS = (
    "enzyme activity",
    "enzyme inhibition",
    "fluorogenic substrate",
    "gelatin zymography",
    "gelatinolytic activity",
    "substrate cleavage",
)

MMP3_FUNCTIONAL_TERMS = (
    "inhibition of mmp3",
    "inhibition of mmp 3",
    "inhibition of mmp-3",
    "inhibition of matrix metalloproteinase 3",
    "inhibition of stromelysin 1",
    "inhibition of stromelysin-1",
    "mmp3 activity",
    "mmp 3 activity",
    "mmp-3 activity",
    "stromelysin 1 activity",
    "stromelysin-1 activity",
)

NRF2_FUNCTIONAL_TERMS = (
    "nrf2 activator",
    "nrf2 activators",
    "nrf2 activation",
    "activation of nrf2",
    "inhibition of nrf2",
    "nrf2 inhibitor",
    "nrf2 inhibitors",
    "nrf2 translocation",
    "nrf2 nuclear translocation",
    "nrf2 transcription",
    "nrf2 transcriptional",
    "nrf2 dependent transcriptional",
    "nrf2 dependant transcriptional",
    "antioxidant response element",
    "are driven luciferase",
    "are mediated luciferase",
    "are luciferase",
    "are fluc",
)

NRF2_EXCLUDED_PHENOTYPES = (
    "cell viability",
    "cell death",
    "cytoprotection",
    "tumor growth",
    "cell proliferation",
    "nqo1 enzymatic activity",
    "nqo1 activity",
    "nqo1 protein level",
    "nqo1 protein expression",
    "nqo1 gene expression",
    "nqo1 mrna expression",
    "gclm expression",
    "gclm gene expression",
)

NRF2_DIRECT_READOUT_TERMS = (
    "reporter gene",
    "luciferase",
    "are fluc",
    "nuclear translocation",
    "translocation to nucleus",
)

NRF2_DOWNSTREAM_MARKER_TERMS = (
    "nqo1",
    "gclm",
)

NRF2_REPORTER_TERMS = (
    "reporter gene",
    "luciferase",
    "are fluc",
)

KEAP1_NRF2_PPI_TERMS = (
    "keap1 nrf2 interaction",
    "keap1 and nrf2",
    "keap1 to nrf2",
    "keap1 kelch",
    "nrf2 keap1 fp assay",
    "nrf2 peptide",
    "protein protein interaction",
    "fluorescence polarization",
    "fluorescence polarisation",
    "fluorescence anisotropy",
    "tr fret",
)

KEAP1_BINDING_ONLY_TERMS = (
    "thermofluor",
    "thermal shift",
    "protein thermal stability",
    "surface plasmon resonance",
    "binding affinity",
)

HIF1_FUNCTIONAL_TERMS = (
    "hypoxia response element",
    "hre luciferase",
    "hre dependent",
    "hre driven",
    "hre reporter",
    "hif1 activation",
    "hif1alpha activation",
    "hif 1alpha activation",
    "hif1 transcriptional",
    "hif1alpha transcriptional",
    "hif 1alpha transcriptional",
    "hif1 transactivation",
    "hif 1alpha transactivation",
    "hif1alpha stabilization",
    "hif 1alpha stabilization",
    "hif1alpha accumulation",
    "hif 1alpha accumulation",
    "hif1alpha nuclear translocation",
    "hif 1alpha nuclear translocation",
)

HIF1_EXCLUDED_PHENOTYPES = (
    "cell viability",
    "cell proliferation",
    "cell migration",
    "wound healing",
    "tumor growth",
    "vegf production",
    "vegf expression",
    "erythropoietin production",
    "erythropoietin secretion",
    "epo production",
    "epo release",
)

HIF1_BINDING_ONLY_TERMS = (
    "binding affinity",
    "dissociation constant",
    "displacement of",
    "protein protein interaction",
    "surface plasmon resonance",
)

PHD2_FUNCTIONAL_TERMS = (
    "phd2 enzyme",
    "phd2 activity",
    "phd2 enzymatic",
    "inhibition of phd2",
    "inhibition of hif phd2",
    "inhibition of human egln1",
    "egln 1 activity",
    "hif ph assay",
    "hif ph2",
    "hydroxylation of pro564",
    "prolyl hydroxylation reaction",
)

PHD2_BINDING_ONLY_TERMS = (
    "binding affinity",
    "dissociation constant",
    "displacement of",
    "binding to phd2",
)

BASE_ABUNDANCE_TERMS = (
    "mrna expression",
    "protein expression",
    "expression level",
    "protein level",
    "mrna level",
    "protein localization",
    "membrane localization",
)


def classify_distance_assay(row: Mapping[str, Any]) -> DistanceAssayDecision:
    """Map a ChEMBL assay to at most one predeclared BBB distance family."""
    text = _full_text(row)
    target_id = str(row.get("target_chembl_id") or "").strip()

    if _has_any(text, TIGHT_JUNCTION_READOUTS):
        return _tight_junction_base_state_decision(row, text)
    if target_id in PXR_CAR_TARGETS:
        return _pxr_car_base_state_decision(row, text)
    if target_id in HUMAN_MMP9_TARGETS:
        return _mmp9_decision(row, text)
    if target_id in HUMAN_MMP3_TARGETS:
        return _mmp3_decision(row, text)
    if target_id in HUMAN_NRF2_TARGETS:
        return _nrf2_decision(row, text)
    if target_id in HUMAN_KEAP1_NRF2_TARGETS:
        # ChEMBL assigns both cellular NRF2-state assays and biochemical PPI
        # assays to this target.  The measured endpoint, not only the target
        # identifier, determines whether the record is H1 or H2.
        if _has_any(text, NRF2_FUNCTIONAL_TERMS):
            return _nrf2_decision(row, text)
        return _keap1_nrf2_decision(row, text)
    if target_id in HUMAN_HIF1_TARGETS:
        return _hif1_decision(row, text)
    if target_id in HUMAN_PHD2_TARGETS:
        # A cellular HRE/stabilization assay measures the HIF-1 state (H1),
        # whereas a direct hydroxylase assay measures PHD2 itself (H2).
        if _has_any(text, HIF1_FUNCTIONAL_TERMS):
            return _hif1_decision(row, text)
        return _phd2_decision(row, text)
    return DistanceAssayDecision(status="unmatched")


def classify_base_measured_states(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Assign every frozen D/C assay to coarse and observed measured-state nodes."""
    tier = str(row.get("tier") or "").strip()
    text = _full_text(row)
    target_id = str(row.get("target_chembl_id") or "").strip()
    nodes: set[str] = set()
    if tier == "Tier 1":
        nodes.add("direct_brain_exposure")
    elif tier == "Tier 2":
        nodes.add("passive_permeability")
        if _has_any(text, TIGHT_JUNCTION_READOUTS):
            nodes.add("tight_junction_integrity")
    elif tier == "Tier 3":
        nodes.add("efflux_transport")
        if _has_any(text, BASE_ABUNDANCE_TERMS):
            nodes.add("efflux_transporter_abundance")
        if target_id in PXR_CAR_TARGETS and _has_any(text, PXR_CAR_FUNCTIONAL_TERMS):
            nodes.add("pxr_car_activation")
    elif tier == "Tier 4":
        nodes.add("influx_transport")
        if _has_any(text, BASE_ABUNDANCE_TERMS):
            nodes.add("influx_transporter_abundance")
    return tuple(sorted(nodes))


def assign_distance_endpoint_group(row: Mapping[str, Any]) -> EndpointAssignment:
    """Use the frozen assay manifest fields when building extension evidence rows."""
    family_id = str(row.get("distance_family_id") or "").strip()
    level = str(row.get("distance_level") or "").strip()
    source_group_id = str(row.get("source_group_id") or "").strip()
    if not family_id or level not in {"H1", "H2"} or not source_group_id:
        raise ValueError(
            f"Distance evidence row lacks a frozen family mapping: assay={row.get('assay_chembl_id')}"
        )
    return EndpointAssignment(
        tier=f"Distance {level}",
        endpoint_group=family_id,
        group_id=source_group_id,
        evidence_direction=str(row.get("effect_direction") or "context_dependent"),
        evidence_strength="moderate" if level == "H1" else "distant",
        reason=str(row.get("mapping_reason") or f"Frozen {level} family `{family_id}`."),
    )


def _tight_junction_base_state_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    matched_readouts = _matched_terms(text, TIGHT_JUNCTION_READOUTS)
    matched_bbb_models = _matched_terms(text, BBB_BARRIER_MODELS)
    matched_proxy_models = _matched_terms(text, TRANSPORTABLE_BARRIER_MODELS)
    base = dict(
        family_id="tight_junction_integrity",
        distance_level="B",
        measured_node="tight_junction_integrity",
        matched_rules=(*matched_readouts, *matched_bbb_models, *matched_proxy_models),
    )
    if not matched_readouts:
        return DistanceAssayDecision(status="unmatched")
    if _has_any(text, ASSAY_QUALITY_EXCLUSION_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="matched" if matched_bbb_models else "transportable",
            quality_status="explicit_assay_quality_failure",
            reason="Assay description explicitly reports insoluble, not tested, or not determined evidence.",
            **base,
        )
    if not matched_bbb_models and not matched_proxy_models:
        return DistanceAssayDecision(
            status="out_of_scope",
            scope_match="out_of_scope",
            quality_status="not_evaluated",
            reason="Tight-junction term occurs outside a BBB-relevant or transportable barrier model.",
            **base,
        )
    if _n_unique_molecules(row) < 1 or not tuple(row.get("standard_types") or ()):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="matched" if matched_bbb_models else "transportable",
            quality_status="missing_compound_activity_or_endpoint",
            reason="No compound-level activity or normalized endpoint is available.",
            **base,
        )
    return DistanceAssayDecision(
        status="base_state_overlap",
        scope_match="matched" if matched_bbb_models else "transportable",
        quality_status="pass",
        effect_direction="assay-specific increase or decrease in junction/barrier integrity",
        reason=(
            "Tight-junction/barrier-integrity is already represented in frozen D/C and cannot be introduced "
            "as an H1 extension."
        ),
        **base,
    )


def _pxr_car_base_state_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, PXR_CAR_FUNCTIONAL_TERMS)
    excluded = _matched_terms(text, PXR_CAR_EXCLUDED_TERMS)
    base = dict(
        family_id="pxr_car_activation",
        distance_level="B",
        measured_node="pxr_car_activation",
        matched_rules=(*functional, *excluded),
    )
    if _confidence(row) < 8 or str(row.get("relationship_type") or "") not in {"D", ""}:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="non_direct_or_low_confidence_target",
            reason="PXR/CAR target assignment is not direct with confidence >=8.",
            **base,
        )
    if _has_any(text, ASSAY_QUALITY_EXCLUSION_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="explicit_assay_quality_failure",
            reason="Assay description explicitly reports insoluble, not tested, or not determined evidence.",
            **base,
        )
    if excluded or not functional:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_activation",
            reason="Only functional PXR/CAR activation is admissible; binding, antagonism, and counterscreens are excluded.",
            **base,
        )
    if _n_unique_molecules(row) < 1 or not tuple(row.get("standard_types") or ()):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="missing_compound_activity_or_endpoint",
            reason="No compound-level activity or normalized endpoint is available.",
            **base,
        )
    return DistanceAssayDecision(
        status="base_state_overlap",
        scope_match="matched" if _has_any(text, BBB_BARRIER_MODELS) else "transportable",
        quality_status="pass",
        effect_direction="activates PXR/CAR; expected transporter induction remains context-dependent",
        reason=(
            "Functional PXR/CAR activation is already represented in frozen D/C assays that measure MDR1/P-gp "
            "induction and cannot be introduced as H1/H2."
        ),
        **base,
    )


def _mmp9_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, MMP9_FUNCTIONAL_TERMS)
    endpoint_types = {normalize_text(value) for value in row.get("standard_types") or ()}
    functional_endpoint = bool(endpoint_types & {"ic50", "ki", "inhibition", "activity", "ec50"})
    binding_only = _matched_terms(text, MMP9_BINDING_ONLY_TERMS)
    base = dict(
        family_id="mmp9_activity",
        tree_node_id="Distance.h1.passive_permeability",
        parent_c_family_id="Mechanism.tier_2",
        distance_level="H1",
        source_group_id="Distance H1.passive_permeability",
        measured_node="mmp9_activity",
        matched_rules=(*functional, *binding_only),
    )
    if _confidence(row) < 8 or str(row.get("relationship_type") or "") not in {"D", ""}:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="non_direct_or_low_confidence_target",
            reason="MMP-9 target assignment is not direct with confidence >=8.",
            **base,
        )
    if _has_any(text, ASSAY_QUALITY_EXCLUSION_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="explicit_assay_quality_failure",
            reason="Assay description explicitly reports insoluble, not tested, or not determined evidence.",
            **base,
        )
    if _has_any(text, MMP9_INDIRECT_READOUT_TERMS) and not _has_any(text, MMP9_DIRECT_METHOD_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="indirect_downstream_readout",
            reason="MMP-9 target is present, but the measured endpoint is an indirect cellular phenotype.",
            **base,
        )
    if binding_only or (not functional and not functional_endpoint):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_mmp9_activity",
            reason="MMP-9 binding-only or semantically ambiguous assays are excluded.",
            **base,
        )
    if _n_unique_molecules(row) < 1 or not tuple(row.get("standard_types") or ()):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="missing_compound_activity_or_endpoint",
            reason="No compound-level activity or normalized endpoint is available.",
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="matched" if _has_any(text, BBB_BARRIER_MODELS) else "transportable",
        quality_status="pass",
        effect_direction="assay-specific MMP-9 activity or inhibition",
        reason="Direct human MMP-9 functional activity/inhibition assay with a usable compound endpoint.",
        **base,
    )


def _mmp3_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, MMP3_FUNCTIONAL_TERMS)
    endpoint_types = {normalize_text(value) for value in row.get("standard_types") or ()}
    functional_endpoint = bool(endpoint_types & {"ic50", "ki", "inhibition", "activity", "ec50"})
    binding_only = _matched_terms(text, MMP9_BINDING_ONLY_TERMS)
    base = dict(
        family_id="mmp3_activity",
        tree_node_id="Distance.h1.passive_permeability",
        parent_c_family_id="Mechanism.tier_2",
        distance_level="H1",
        source_group_id="Distance H1.passive_permeability",
        measured_node="mmp3_activity",
        matched_rules=(*functional, *binding_only),
    )
    if _confidence(row) < 8 or str(row.get("relationship_type") or "") not in {"D", ""}:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="non_direct_or_low_confidence_target",
            reason="MMP-3 target assignment is not direct with confidence >=8.",
            **base,
        )
    if _has_any(text, ASSAY_QUALITY_EXCLUSION_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="explicit_assay_quality_failure",
            reason="Assay description explicitly reports insoluble, not tested, or not determined evidence.",
            **base,
        )
    if _has_any(text, MMP9_INDIRECT_READOUT_TERMS) and not _has_any(text, MMP9_DIRECT_METHOD_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="indirect_downstream_readout",
            reason="MMP-3 target is present, but the measured endpoint is an indirect cellular phenotype.",
            **base,
        )
    if binding_only or (not functional and not functional_endpoint):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_mmp3_activity",
            reason="MMP-3 binding-only or semantically ambiguous assays are excluded.",
            **base,
        )
    if _n_unique_molecules(row) < 1 or not tuple(row.get("standard_types") or ()):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="missing_compound_activity_or_endpoint",
            reason="No compound-level activity or normalized endpoint is available.",
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="transportable",
        quality_status="pass",
        effect_direction="assay-specific MMP-3 activity or inhibition",
        reason="Direct human MMP-3 functional activity/inhibition assay with a usable compound endpoint.",
        **base,
    )


def _nrf2_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, NRF2_FUNCTIONAL_TERMS)
    excluded = _matched_terms(text, NRF2_EXCLUDED_PHENOTYPES)
    direct_readout = _matched_terms(text, NRF2_DIRECT_READOUT_TERMS)
    downstream_markers = _matched_terms(text, NRF2_DOWNSTREAM_MARKER_TERMS)
    reporter = _matched_terms(text, NRF2_REPORTER_TERMS)
    endpoint_types = _endpoint_types(row)
    base = dict(
        family_id="nrf2_activation",
        tree_node_id="Distance.h1.efflux_transport",
        parent_c_family_id="Mechanism.tier_3",
        distance_level="H1",
        source_group_id="Distance H1.efflux_transport",
        measured_node="nrf2_activation",
        matched_rules=(*functional, *direct_readout, *downstream_markers, *excluded),
    )
    minimum_confidence = 5 if str(row.get("target_chembl_id") or "") in HUMAN_KEAP1_NRF2_TARGETS else 8
    failure = _common_extension_quality_failure(
        row,
        base,
        minimum_confidence=minimum_confidence,
    )
    if failure is not None:
        return failure
    indirect_marker = bool(downstream_markers and not reporter)
    if (excluded and not reporter) or indirect_marker or not functional or not endpoint_types & {
        "activity", "ec50", "ic50", "potency", "fc", "fold change", "cd"
    }:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_nrf2_state",
            reason=(
                "Only cellular NRF2/ARE transcriptional or nuclear-translocation endpoints are admissible; "
                "generic downstream stress phenotypes are excluded."
            ),
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="transportable",
        quality_status="pass",
        effect_direction="assay-specific activation or inhibition of cellular NRF2 signaling",
        reason="Functional cellular NRF2/ARE or nuclear-translocation assay with a usable compound endpoint.",
        **base,
    )


def _keap1_nrf2_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, KEAP1_NRF2_PPI_TERMS)
    binding_only = _matched_terms(text, KEAP1_BINDING_ONLY_TERMS)
    endpoint_types = _endpoint_types(row)
    base = dict(
        family_id="keap1_nrf2_interaction",
        tree_node_id="Distance.h2.efflux_transport",
        parent_c_family_id="Mechanism.tier_3",
        distance_level="H2",
        source_group_id="Distance H2.efflux_transport",
        measured_node="keap1_nrf2_interaction",
        matched_rules=(*functional, *binding_only),
    )
    failure = _common_extension_quality_failure(row, base, minimum_confidence=4)
    if failure is not None:
        return failure
    if binding_only or not functional or not endpoint_types & {
        "ac50", "ic50", "ki", "inhibition", "activity", "potency"
    }:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_keap1_nrf2_ppi",
            reason=(
                "Require compound inhibition of the KEAP1-NRF2 interaction; thermal-shift, SPR, and "
                "binding-affinity-only records are excluded."
            ),
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="transportable",
        quality_status="pass",
        effect_direction="inhibits the KEAP1-NRF2 protein-protein interaction",
        reason="Direct biochemical KEAP1-NRF2 interaction-inhibition assay with a usable compound endpoint.",
        **base,
    )


def _hif1_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, HIF1_FUNCTIONAL_TERMS)
    excluded = _matched_terms(text, HIF1_EXCLUDED_PHENOTYPES)
    binding_only = _matched_terms(text, HIF1_BINDING_ONLY_TERMS)
    endpoint_types = _endpoint_types(row)
    base = dict(
        family_id="hif1_activation",
        tree_node_id="Distance.h1.influx_transport",
        parent_c_family_id="Mechanism.tier_4",
        distance_level="H1",
        source_group_id="Distance H1.influx_transport",
        measured_node="hif1_activation",
        matched_rules=(*functional, *excluded, *binding_only),
    )
    failure = _common_extension_quality_failure(row, base, minimum_confidence=5)
    if failure is not None:
        return failure
    if excluded or binding_only or not functional or not endpoint_types & {
        "activity", "ec50", "ic50", "potency", "inhibition", "inh"
    }:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_hif1_state",
            reason=(
                "Require HRE transcription, HIF-1alpha stabilization/accumulation, or nuclear translocation; "
                "binding-only and downstream VEGF/EPO or viability phenotypes are excluded."
            ),
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="transportable",
        quality_status="pass",
        effect_direction="assay-specific activation, stabilization, or inhibition of cellular HIF-1 signaling",
        reason="Functional cellular HIF-1/HRE state assay with a usable compound endpoint.",
        **base,
    )


def _phd2_decision(row: Mapping[str, Any], text: str) -> DistanceAssayDecision:
    functional = _matched_terms(text, PHD2_FUNCTIONAL_TERMS)
    binding_only = _matched_terms(text, PHD2_BINDING_ONLY_TERMS)
    endpoint_types = _endpoint_types(row)
    base = dict(
        family_id="phd2_activity",
        tree_node_id="Distance.h2.influx_transport",
        parent_c_family_id="Mechanism.tier_4",
        distance_level="H2",
        source_group_id="Distance H2.influx_transport",
        measured_node="phd2_activity",
        matched_rules=(*functional, *binding_only),
    )
    failure = _common_extension_quality_failure(row, base, minimum_confidence=8)
    if failure is not None:
        return failure
    if binding_only or not functional or not endpoint_types & {
        "ic50", "inhibition", "activity", "potency"
    }:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="not_functional_phd2_activity",
            reason="Require direct PHD2/EGLN1 hydroxylase activity or inhibition; binding-only records are excluded.",
            **base,
        )
    return DistanceAssayDecision(
        status="include",
        scope_match="transportable",
        quality_status="pass",
        effect_direction="assay-specific PHD2 hydroxylase activity or inhibition",
        reason="Direct human PHD2/EGLN1 functional enzyme assay with a usable compound endpoint.",
        **base,
    )


def _full_text(row: Mapping[str, Any]) -> str:
    return normalize_text(
        join_text_parts(
            row.get("description"),
            row.get("assay_cell_type"),
            row.get("assay_tissue"),
            row.get("assay_organism"),
            row.get("target_pref_name"),
            _join_values(row.get("target_genes")),
            _join_values(row.get("target_synonyms")),
            _join_values(row.get("standard_types")),
        )
    )


def _endpoint_types(row: Mapping[str, Any]) -> set[str]:
    return {normalize_text(value) for value in row.get("standard_types") or ()}


def _common_extension_quality_failure(
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    minimum_confidence: int = 8,
) -> DistanceAssayDecision | None:
    if _confidence(row) < minimum_confidence or str(row.get("relationship_type") or "") not in {
        "D",
        "H",
        "",
    }:
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="unsupported_or_low_confidence_target",
            reason=(
                f"Target assignment is neither direct/homologous nor confidence >= {minimum_confidence}."
            ),
            **base,
        )
    if _has_any(_full_text(row), ASSAY_QUALITY_EXCLUSION_TERMS):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="explicit_assay_quality_failure",
            reason="Assay description explicitly reports insoluble, not tested, or not determined evidence.",
            **base,
        )
    if _n_unique_molecules(row) < 1 or not tuple(row.get("standard_types") or ()):
        return DistanceAssayDecision(
            status="quality_fail",
            scope_match="transportable",
            quality_status="missing_compound_activity_or_endpoint",
            reason="No compound-level activity or normalized endpoint is available.",
            **base,
        )
    return None


def _join_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(str(item) for item in value)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return bool(_matched_terms(text, terms))


def _matched_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """Match normalized phrases without allowing substrings inside words.

    The boundary is important for short assay abbreviations: plain substring
    matching makes ``teer`` match ``volunteer`` and pollutes the exclusion
    audit with unrelated clinical assays.
    """
    return tuple(
        term
        for term in terms
        if re.search(
            rf"(?<![a-z0-9]){re.escape(normalize_text(term))}(?![a-z0-9])",
            text,
        )
    )


def _confidence(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("confidence_score") or 0)
    except (TypeError, ValueError):
        return 0


def _n_unique_molecules(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("n_unique_molecules") or 0)
    except (TypeError, ValueError):
        return 0
