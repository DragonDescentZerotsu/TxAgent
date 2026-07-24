"""Frozen BBB D/C/H1/H2 graph for the ChEMBL-only distance experiment."""

from __future__ import annotations

from tools.chembl_tool.common.evidence_distance import (
    FORWARD,
    BiologicalNode,
    DistanceExpansionConfig,
    DistanceFamilySpec,
    DistanceTreeNodeSpec,
    MechanismEdge,
)

from .experiment_config import CHEMBL


CHEMBL_RELEASE = "ChEMBL 36"


NODES = (
    BiologicalNode(
        "direct_brain_exposure",
        "Direct brain exposure",
        "Measured brain, CSF, or unbound brain exposure relative to systemic exposure.",
        "brain/plasma, CSF/plasma, logBB, Kp,uu,brain, or brain concentration",
        core_anchor=True,
    ),
    BiologicalNode(
        "passive_permeability",
        "Passive or paracellular permeability",
        "Compound passage across an artificial membrane or epithelial/endothelial barrier model.",
        "Papp, permeability coefficient, or directional passive transport",
        core_anchor=True,
    ),
    BiologicalNode(
        "efflux_transport",
        "Efflux transport",
        "Functional export of a compound by BBB-relevant ABC transporters.",
        "efflux ratio, bidirectional Papp, ATPase, substrate transport, or probe accumulation",
        core_anchor=True,
    ),
    BiologicalNode(
        "influx_transport",
        "Influx transport",
        "Functional uptake of a compound by BBB-relevant solute carriers or uptake systems.",
        "uptake, transport, substrate kinetics, or functional inhibition",
        core_anchor=True,
    ),
    BiologicalNode(
        "tight_junction_integrity",
        "Barrier tight-junction integrity",
        "Integrity and organization of endothelial or permeability-model tight junctions.",
        "TEER or ZO-1, occludin, claudin, and barrier-integrity readouts",
        core_anchor=True,
    ),
    BiologicalNode(
        "efflux_transporter_abundance",
        "Efflux-transporter abundance",
        "Expression or membrane abundance of ABCB1/P-gp or ABCG2/BCRP.",
        "transporter mRNA, protein abundance, or membrane localization",
        core_anchor=True,
    ),
    BiologicalNode(
        "influx_transporter_abundance",
        "Influx-transporter abundance",
        "Expression or membrane abundance of BBB-relevant influx transporters.",
        "SLC transporter mRNA, protein abundance, or membrane localization",
        core_anchor=True,
    ),
    BiologicalNode(
        "pxr_car_activation",
        "PXR/CAR functional activation",
        "Functional activation of the xenobiotic-sensing nuclear receptors PXR or CAR.",
        "agonist or transactivation reporter response",
        core_anchor=True,
    ),
    BiologicalNode(
        "mmp9_activity",
        "MMP-9 proteolytic activity",
        "Functional matrix metalloproteinase-9 activity capable of remodeling the neurovascular barrier.",
        "enzyme activity or inhibition potency against MMP-9",
    ),
    BiologicalNode(
        "mmp3_activity",
        "MMP-3 proteolytic activity",
        "Functional stromelysin-1/MMP-3 activity capable of remodeling the neurovascular barrier.",
        "enzyme activity or inhibition potency against MMP-3",
    ),
    BiologicalNode(
        "nrf2_activation",
        "NRF2 transcriptional activation",
        "Functional NRF2 nuclear translocation or antioxidant-response-element transcription.",
        "NRF2 translocation, ARE reporter activity, or functional transcriptional activation",
    ),
    BiologicalNode(
        "keap1_nrf2_interaction",
        "KEAP1-NRF2 interaction",
        "Functional or biochemical interaction through which KEAP1 retains and targets NRF2 for degradation.",
        "compound inhibition potency against the KEAP1-NRF2 protein-protein interaction",
    ),
    BiologicalNode(
        "hif1_activation",
        "HIF-1 transcriptional activation",
        "Functional stabilization or transcriptional activation of hypoxia-inducible factor 1.",
        "HIF-1alpha stabilization, nuclear translocation, HRE reporter activity, or functional modulation",
    ),
    BiologicalNode(
        "phd2_activity",
        "PHD2/EGLN1 hydroxylase activity",
        "HIF prolyl-hydroxylase activity that promotes HIF-1alpha turnover under oxygenated conditions.",
        "enzyme activity or inhibition potency against human PHD2/EGLN1",
    ),
)


EDGES = (
    MechanismEdge(
        edge_id="tight_junction_to_passive_permeability",
        upstream_node="tight_junction_integrity",
        downstream_node="passive_permeability",
        relation_type="functional_component",
        effect_direction="greater junction integrity generally lowers paracellular permeability",
        admissible_inference_directions=(FORWARD,),
        applicability="Intact epithelial/endothelial monolayers used as BBB or transportable permeability models.",
        rationale=(
            "Tight-junction organization is a direct functional determinant of paracellular barrier permeability; "
            "TEER and junction-protein integrity are therefore adjacent mechanistic readouts, not permeability itself."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/24262108/",
            "https://pubmed.ncbi.nlm.nih.gov/28787238/",
        ),
    ),
    MechanismEdge(
        edge_id="mmp3_to_mmp9_activity",
        upstream_node="mmp3_activity",
        downstream_node="mmp9_activity",
        relation_type="causal_process",
        effect_direction="MMP-3 can proteolytically activate proMMP-9",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Proteolytic extracellular contexts with available proMMP-9; the magnitude is conditional on TIMPs "
            "and competing activation pathways."
        ),
        rationale=(
            "Biochemical processing experiments and ex-vivo inhibition studies show stromelysin-1/MMP-3 can "
            "convert progelatinase B/proMMP-9 into active MMP-9."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/7890773/",
            "https://pubmed.ncbi.nlm.nih.gov/11485578/",
        ),
    ),
    MechanismEdge(
        edge_id="mmp3_to_tight_junction_integrity",
        upstream_node="mmp3_activity",
        downstream_node="tight_junction_integrity",
        relation_type="causal_process",
        effect_direction="greater MMP-3 activity can reduce endothelial junction integrity",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Inflammatory or barrier-injury contexts in which extracellular MMP-3 can access endothelial or "
            "perivascular substrates."
        ),
        rationale=(
            "MMP-3 perturbation studies directly connect stromelysin-1 activity to reduced tight-junction and "
            "VE-cadherin proteins and increased BBB permeability; this is a one-edge shortcut."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/33859779/",
            "https://pubmed.ncbi.nlm.nih.gov/16624562/",
        ),
    ),
    MechanismEdge(
        edge_id="pxr_car_to_efflux_transporter_abundance",
        upstream_node="pxr_car_activation",
        downstream_node="efflux_transporter_abundance",
        relation_type="causal_process",
        effect_direction="PXR/CAR activation can increase ABCB1 and/or ABCG2 expression",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Functional PXR/CAR activation with species-aware interpretation; induction is conditional on cell "
            "context and exposure duration."
        ),
        rationale=(
            "PXR and CAR are xenobiotic sensors with experimentally demonstrated regulation of ABC efflux "
            "transporters in brain microvascular endothelial and BBB models."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/21517853/",
            "https://pubmed.ncbi.nlm.nih.gov/23340159/",
        ),
    ),
    MechanismEdge(
        edge_id="efflux_transporter_abundance_to_transport",
        upstream_node="efflux_transporter_abundance",
        downstream_node="efflux_transport",
        relation_type="functional_component",
        effect_direction="greater functional membrane abundance can increase efflux capacity",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "ABCB1/P-gp or ABCG2/BCRP abundance at a polarized barrier membrane; abundance alone remains a "
            "conditional proxy for functional transport."
        ),
        rationale="Transporter abundance is an upstream determinant of measured efflux capacity at the BBB.",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/23836171/",
            "https://pubmed.ncbi.nlm.nih.gov/23777437/",
        ),
    ),
    MechanismEdge(
        edge_id="mmp9_to_tight_junction_integrity",
        upstream_node="mmp9_activity",
        downstream_node="tight_junction_integrity",
        relation_type="causal_process",
        effect_direction="greater MMP-9 activity can reduce junction-protein integrity",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Barrier-remodeling or inflammatory contexts in which extracellular MMP-9 can access neurovascular "
            "matrix and junction proteins."
        ),
        rationale=(
            "Experimental BBB injury studies connect MMP-9 activation to loss of tight-junction proteins and "
            "increased barrier permeability; because tight-junction integrity is already in the frozen D/C "
            "measured-state envelope, MMP-9 activity is one admissible edge outside that base."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/28523565/",
            "https://pubmed.ncbi.nlm.nih.gov/21803344/",
            "https://pubmed.ncbi.nlm.nih.gov/23532086/",
        ),
    ),
    MechanismEdge(
        edge_id="nrf2_to_efflux_transporter_abundance",
        upstream_node="nrf2_activation",
        downstream_node="efflux_transporter_abundance",
        relation_type="causal_process",
        effect_direction="NRF2 activation can increase BBB ABC-transporter expression and activity",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Brain or spinal barrier endothelium under basal or oxidative/hypoxia-reoxygenation contexts; the "
            "specific ABC transporter and magnitude remain context dependent."
        ),
        rationale=(
            "Genetic and pharmacological NRF2 activation studies at barrier endothelium report increased ABC "
            "transporter expression and functional efflux."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/24948812/",
            "https://pubmed.ncbi.nlm.nih.gov/28298215/",
        ),
    ),
    MechanismEdge(
        edge_id="keap1_nrf2_interaction_to_nrf2_activation",
        upstream_node="keap1_nrf2_interaction",
        downstream_node="nrf2_activation",
        relation_type="causal_process",
        effect_direction="disrupting KEAP1-NRF2 interaction stabilizes and activates NRF2",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Assays that specifically measure KEAP1-NRF2 interaction inhibition; nonspecific electrophilic "
            "stress responses without target engagement are excluded."
        ),
        rationale=(
            "KEAP1 is the substrate adaptor for CUL3-dependent NRF2 ubiquitination; disrupting the interaction "
            "prevents NRF2 turnover and enables transcriptional activation."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/15572695/",
            "https://pubmed.ncbi.nlm.nih.gov/12193649/",
        ),
    ),
    MechanismEdge(
        edge_id="hif1_to_influx_transporter_abundance",
        upstream_node="hif1_activation",
        downstream_node="influx_transporter_abundance",
        relation_type="causal_process",
        effect_direction="HIF-1 activation can increase GLUT1 expression and glucose uptake",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Hypoxic or HIF-stabilized neural/endothelial contexts; this edge is restricted to the GLUT1 part "
            "of the heterogeneous BBB influx family."
        ),
        rationale=(
            "HIF-1 is a transcriptional regulator of GLUT1, and brain studies connect HIF-1alpha with GLUT1 "
            "expression and glucose uptake."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/32079227/",
            "https://pubmed.ncbi.nlm.nih.gov/25879623/",
        ),
    ),
    MechanismEdge(
        edge_id="phd2_to_hif1_activation",
        upstream_node="phd2_activity",
        downstream_node="hif1_activation",
        relation_type="causal_process",
        effect_direction="inhibiting PHD2 hydroxylase activity stabilizes HIF-1alpha",
        admissible_inference_directions=(FORWARD,),
        applicability=(
            "Direct functional inhibition of human PHD2/EGLN1 with a HIF peptide or validated enzyme substrate; "
            "binding-only measurements are excluded."
        ),
        rationale=(
            "PHD2 hydroxylates HIF-alpha for oxygen-dependent turnover; selective PHD inhibition produces "
            "HIF-1alpha stabilization."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/38637657/",
            "https://pubmed.ncbi.nlm.nih.gov/34339019/",
        ),
    ),
)


EXTENSION_TREE_NODES = (
    DistanceTreeNodeSpec(
        tree_node_id="Distance.h1.passive_permeability",
        display_name="One-hop upstream evidence for passive/barrier permeability",
        parent_id="Mechanism.tier_2",
        parent_c_family_id="Mechanism.tier_2",
        source_group_id="Distance H1.passive_permeability",
        declared_level="H1",
        anchor_nodes=("tight_junction_integrity",),
        measurement_family_ids=("mmp9_activity", "mmp3_activity"),
    ),
    DistanceTreeNodeSpec(
        tree_node_id="Distance.h1.efflux_transport",
        display_name="One-hop upstream evidence for efflux transport",
        parent_id="Mechanism.tier_3",
        parent_c_family_id="Mechanism.tier_3",
        source_group_id="Distance H1.efflux_transport",
        declared_level="H1",
        anchor_nodes=("efflux_transporter_abundance",),
        measurement_family_ids=("nrf2_activation",),
    ),
    DistanceTreeNodeSpec(
        tree_node_id="Distance.h2.efflux_transport",
        display_name="Two-hop upstream evidence for efflux transport",
        parent_id="Distance.h1.efflux_transport",
        parent_c_family_id="Mechanism.tier_3",
        source_group_id="Distance H2.efflux_transport",
        declared_level="H2",
        anchor_nodes=("nrf2_activation",),
        measurement_family_ids=("keap1_nrf2_interaction",),
    ),
    DistanceTreeNodeSpec(
        tree_node_id="Distance.h1.influx_transport",
        display_name="One-hop upstream evidence for influx transport",
        parent_id="Mechanism.tier_4",
        parent_c_family_id="Mechanism.tier_4",
        source_group_id="Distance H1.influx_transport",
        declared_level="H1",
        anchor_nodes=("influx_transporter_abundance",),
        measurement_family_ids=("hif1_activation",),
    ),
    DistanceTreeNodeSpec(
        tree_node_id="Distance.h2.influx_transport",
        display_name="Two-hop upstream evidence for influx transport",
        parent_id="Distance.h1.influx_transport",
        parent_c_family_id="Mechanism.tier_4",
        source_group_id="Distance H2.influx_transport",
        declared_level="H2",
        anchor_nodes=("hif1_activation",),
        measurement_family_ids=("phd2_activity",),
    ),
)


EXTENSION_FAMILIES = (
    DistanceFamilySpec(
        family_id="mmp9_activity",
        display_name="MMP-9 activity upstream of junction disruption",
        tree_node_id="Distance.h1.passive_permeability",
        parent_c_family_id="Mechanism.tier_2",
        source_group_id="Distance H1.passive_permeability",
        measured_node="mmp9_activity",
        declared_level="H1",
        admissible_path=("mmp9_activity", "tight_junction_integrity"),
        scope_rule=(
            "Human MMP-9 biochemical or cellular functional assays are transportable but condition-dependent; "
            "generic collagenase and non-MMP-9 matrix assays are out of scope."
        ),
        quality_rule=(
            "Require a direct human MMP-9 target assignment with confidence >=8, compound-level activities, "
            "and functional activity or inhibition potency; binding-only records are excluded."
        ),
    ),
    DistanceFamilySpec(
        family_id="mmp3_activity",
        display_name="MMP-3 activity upstream of junction disruption",
        tree_node_id="Distance.h1.passive_permeability",
        parent_c_family_id="Mechanism.tier_2",
        source_group_id="Distance H1.passive_permeability",
        measured_node="mmp3_activity",
        declared_level="H1",
        admissible_path=("mmp3_activity", "tight_junction_integrity"),
        scope_rule=(
            "Human MMP-3 biochemical functional assays are transportable but context-dependent; generic "
            "stromelysin-family or non-MMP-3 matrix assays are out of scope."
        ),
        quality_rule=(
            "Require a direct human MMP-3 target assignment with confidence >=8, compound-level activities, "
            "and functional activity or inhibition potency; binding-only records are excluded."
        ),
    ),
    DistanceFamilySpec(
        family_id="nrf2_activation",
        display_name="Functional NRF2 activation upstream of BBB efflux-transporter induction",
        tree_node_id="Distance.h1.efflux_transport",
        parent_c_family_id="Mechanism.tier_3",
        source_group_id="Distance H1.efflux_transport",
        measured_node="nrf2_activation",
        declared_level="H1",
        admissible_path=("nrf2_activation", "efflux_transporter_abundance"),
        scope_rule=(
            "Human cellular NRF2 nuclear-translocation or ARE-reporter assays are transportable to BBB "
            "transcriptional regulation; generic antioxidant phenotypes are out of scope."
        ),
        quality_rule=(
            "Require a functional NRF2 activation/translocation/readout with compound-level activity; "
            "biochemical KEAP1-NRF2 binding or PPI inhibition belongs to the H2 family instead."
        ),
    ),
    DistanceFamilySpec(
        family_id="keap1_nrf2_interaction",
        display_name="KEAP1-NRF2 interaction inhibition upstream of NRF2 activation",
        tree_node_id="Distance.h2.efflux_transport",
        parent_c_family_id="Mechanism.tier_3",
        source_group_id="Distance H2.efflux_transport",
        measured_node="keap1_nrf2_interaction",
        declared_level="H2",
        admissible_path=(
            "keap1_nrf2_interaction",
            "nrf2_activation",
            "efflux_transporter_abundance",
        ),
        scope_rule=(
            "Direct human KEAP1-NRF2 biochemical PPI assays are transportable but upstream of cellular NRF2 "
            "activation; unrelated KEAP1 complexes are out of scope."
        ),
        quality_rule=(
            "Require direct KEAP1-NRF2 PPI inhibition with a normalized compound endpoint; thermal-shift-only "
            "binding and cellular NRF2 reporter assays are excluded from this H2 family."
        ),
    ),
    DistanceFamilySpec(
        family_id="hif1_activation",
        display_name="Functional HIF-1 activation upstream of GLUT1 abundance",
        tree_node_id="Distance.h1.influx_transport",
        parent_c_family_id="Mechanism.tier_4",
        source_group_id="Distance H1.influx_transport",
        measured_node="hif1_activation",
        declared_level="H1",
        admissible_path=("hif1_activation", "influx_transporter_abundance"),
        scope_rule=(
            "Human HIF-1alpha stabilization, nuclear-translocation, or HRE reporter assays are transportable "
            "specifically to the GLUT1 part of the heterogeneous BBB influx family."
        ),
        quality_rule=(
            "Require direct functional modulation of HIF-1/HRE signaling with compound-level activity; generic "
            "hypoxia toxicity and downstream VEGF-only phenotypes are excluded."
        ),
    ),
    DistanceFamilySpec(
        family_id="phd2_activity",
        display_name="PHD2/EGLN1 activity upstream of HIF-1 activation",
        tree_node_id="Distance.h2.influx_transport",
        parent_c_family_id="Mechanism.tier_4",
        source_group_id="Distance H2.influx_transport",
        measured_node="phd2_activity",
        declared_level="H2",
        admissible_path=("phd2_activity", "hif1_activation", "influx_transporter_abundance"),
        scope_rule=(
            "Direct human PHD2/EGLN1 enzyme assays are transportable upstream evidence for HIF-1 signaling; "
            "other prolyl hydroxylases or nonspecific oxygen-sensing phenotypes are out of scope."
        ),
        quality_rule=(
            "Require direct functional PHD2/EGLN1 activity or inhibition with a normalized compound endpoint; "
            "binding-only measurements are excluded."
        ),
    ),
)


DISTANCE_CONFIG = DistanceExpansionConfig(
    task_name="bbb_martins",
    source_name="chembl",
    source_release=CHEMBL_RELEASE,
    base_source_config=CHEMBL,
    c_family_ids=("Mechanism.tier_2", "Mechanism.tier_3", "Mechanism.tier_4"),
    nodes=NODES,
    edges=EDGES,
    extension_tree_nodes=EXTENSION_TREE_NODES,
    extension_families=EXTENSION_FAMILIES,
)
