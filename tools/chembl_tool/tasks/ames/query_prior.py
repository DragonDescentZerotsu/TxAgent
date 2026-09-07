"""Ames structure-only and condition-specific None prompt contract."""

import json

from tools.chembl_tool.tasks.ames.experiment_config import get_progressive_task_contract


def build_query_prior_messages(branch: str, query: dict, single: dict | None = None) -> list[dict]:
    """Ames scientific prompt contract for fresh structure-only and None branches."""
    if branch == "single":
        task = "Assess the query molecule's structural and physicochemical bacterial mutagenicity prior."
        instructions = [
            "Use the supplied structure and property tool text, without analog evidence or memorized assay labels.",
            "Discuss electrophilicity, plausible metabolic activation or detoxification, bacterial accessibility, and uncertainty.",
            "A structural alert is a hypothesis, not a measured Ames result. Absence of an alert is not proof of a negative result.",
            "Do not invent a strain panel or activation condition; the final branch applies the reported condition.",
        ]
        schema = {
            "ames_prior": "positive | negative | mixed_or_unclear",
            "activation_dependence": "string", "structural_alerts": ["string"],
            "property_drivers": ["string"], "confidence": "high | moderate | low",
            "reasoning_summary": "string", "caveats": ["string"],
        }
    elif branch == "final":
        task = "Predict the condition-specific bacterial reverse-mutation label from the query prior alone."
        instructions = [*get_progressive_task_contract().task_instructions,
            "There is no retrieved analog evidence in this None control. Use only the supplied query properties and single-molecule analysis.",
            "Ignore recognized identities and memorized experimental labels; make a structural and mechanistic prediction.",
        ]
        schema = {
            "ames_prediction": "positive | negative", "confidence": "high | moderate | low",
            "main_reasons": ["string"], "single_molecule_assessment": "string",
            "conflicting_evidence": ["string"], "evidence_gaps": ["string"], "final_summary": "string",
        }
    else:
        raise ValueError(f"Unknown query prior branch: {branch}")
    payload = {"task": task, "query": query, "instructions": instructions, "required_json_schema": schema}
    if single is not None:
        payload["single_molecule_analysis"] = single
    return [
        {"role": "system", "content": "You are a medicinal chemistry bacterial mutagenicity analyst. Return one complete JSON object."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


