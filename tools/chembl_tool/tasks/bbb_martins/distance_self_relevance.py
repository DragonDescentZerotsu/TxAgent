"""Causal-subject audit for the frozen BBB v3 distance families.

This audit is separate from graph distance.  A path may have the correct hop
count yet fail to predict the retrieved molecule's own BBB behavior because a
downstream transporter acts on a different, unobserved substrate.
"""

from __future__ import annotations

from tools.chembl_tool.common.evidence_distance import FamilySelfRelevanceAudit


SELF_RELEVANCE_AUDIT = (
    FamilySelfRelevanceAudit(
        family_id="mmp9_activity",
        status="pass_same_molecule",
        causal_subject=(
            "The assay molecule remains the barrier perturbagen; the downstream tight-junction state is a "
            "shared physical barrier that also applies to that molecule."
        ),
        required_query_roles=(),
        rationale=(
            "MMP-9 perturbation can change BBB junction integrity without replacing the assay molecule with an "
            "unobserved transporter substrate.  Injury/inflammation, target exposure, timing, and the molecule's "
            "dependence on the paracellular barrier remain scope conditions rather than a hidden identity switch."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/28523565/",
            "https://pubmed.ncbi.nlm.nih.gov/21803344/",
        ),
    ),
    FamilySelfRelevanceAudit(
        family_id="mmp3_activity",
        status="pass_same_molecule",
        causal_subject=(
            "The assay molecule remains the barrier perturbagen; MMP-3-mediated junction change affects the "
            "physical barrier encountered by that molecule as well as by other molecules."
        ),
        required_query_roles=(),
        rationale=(
            "Direct MMP-3 perturbation studies connect the enzyme state to BBB junction integrity.  The inference "
            "is conditional on an injured/inflammatory barrier and relevant exposure, but it does not require the "
            "query to be an unmeasured transporter substrate."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/33859779/",
            "https://pubmed.ncbi.nlm.nih.gov/16624562/",
        ),
    ),
    FamilySelfRelevanceAudit(
        family_id="nrf2_activation",
        status="requires_query_role",
        causal_subject=(
            "The assay molecule is an NRF2 perturbagen, but the downstream transport experiment concerns a "
            "separate ABC-transporter probe substrate."
        ),
        required_query_roles=(
            "the query molecule is a substrate of the induced ABCB1, ABCG2, or ABCC2 transporter",
            "transporter induction occurs before or during measurement of the query molecule's BBB disposition",
        ),
        rationale=(
            "NRF2 activation can increase BBB efflux-transporter abundance and reduce brain accumulation of a "
            "known probe substrate such as verapamil.  An NRF2 assay alone does not establish that the activating "
            "molecule is itself transported, so the causal chain changes molecular subject."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/24948812/",
        ),
    ),
    FamilySelfRelevanceAudit(
        family_id="keap1_nrf2_interaction",
        status="requires_query_role",
        causal_subject=(
            "The assay molecule perturbs KEAP1-NRF2 and thereby the barrier environment, while the final efflux "
            "event applies only to a separate molecule with the required ABC-transporter substrate role."
        ),
        required_query_roles=(
            "KEAP1-NRF2 inhibition functionally activates NRF2 in the relevant barrier system",
            "the query molecule is a substrate of the induced ABCB1, ABCG2, or ABCC2 transporter",
            "transporter induction occurs before or during measurement of the query molecule's BBB disposition",
        ),
        rationale=(
            "The KEAP1-to-NRF2 edge is valid, but it inherits the unresolved molecule-identity switch in the "
            "NRF2-to-efflux path.  A biochemical PPI assay cannot supply the missing substrate evidence."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/15572695/",
            "https://pubmed.ncbi.nlm.nih.gov/24948812/",
        ),
    ),
    FamilySelfRelevanceAudit(
        family_id="hif1_activation",
        status="requires_query_role",
        causal_subject=(
            "The assay molecule is a HIF-1 perturbagen, but the demonstrated downstream influx event is transport "
            "of glucose through GLUT1 rather than transport of that perturbagen."
        ),
        required_query_roles=(
            "the query molecule is a GLUT1/SLC2A1 substrate",
            "HIF-1-dependent GLUT1 induction occurs before or during measurement of the query molecule's BBB disposition",
        ),
        rationale=(
            "HIF-1 regulates endothelial GLUT1 and glucose uptake.  That does not imply that an arbitrary HIF-1 "
            "activator or inhibitor is transported by GLUT1, so HIF-1 functional assays cannot by themselves "
            "predict the same molecule's BBB influx."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/23047702/",
        ),
    ),
    FamilySelfRelevanceAudit(
        family_id="phd2_activity",
        status="requires_query_role",
        causal_subject=(
            "The assay molecule perturbs PHD2 and HIF-1, but the downstream influx event still concerns glucose or "
            "another independently established GLUT1 substrate."
        ),
        required_query_roles=(
            "PHD2 modulation functionally changes HIF-1 in the relevant barrier system",
            "the query molecule is a GLUT1/SLC2A1 substrate",
            "GLUT1 induction occurs before or during measurement of the query molecule's BBB disposition",
        ),
        rationale=(
            "The PHD2-to-HIF-1 edge is valid but does not repair the subject switch at HIF-1-to-GLUT1-mediated "
            "transport.  A direct PHD2 enzyme assay contains no evidence that the same compound is a GLUT1 substrate."
        ),
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/34339019/",
            "https://pubmed.ncbi.nlm.nih.gov/23047702/",
        ),
    ),
)
