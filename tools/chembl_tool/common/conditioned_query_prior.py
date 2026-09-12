"""Fresh structure-only and None messages from a task's scientific contract."""

import json

from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract


def build_query_prior_messages(
    contract: ProgressiveTaskContract, branch: str, query: dict, single: dict | None = None,
) -> list[dict]:
    if branch == "single":
        task = f"Assess the query molecule's structural and physicochemical prior for {contract.endpoint_name}."
        instructions = [
            "Use only the supplied structure and property tool text, without analog evidence or memorized assay labels.",
            "Explain plausible structural and mechanistic liabilities, exposure limitations and uncertainty.",
            "A structural alert is a hypothesis, not an observed endpoint. Absence of an alert does not establish a negative result.",
            "Do not invent exposure, species or population conditions; the final branch applies the reported condition.",
        ]
        schema = {
            "endpoint_prior": f"{contract.positive_prediction} | {contract.negative_prediction} | mixed_or_unclear",
            "structural_alerts": ["string"], "property_drivers": ["string"],
            "confidence": "high | moderate | low", "reasoning_summary": "string",
            "caveats": ["string"],
        }
    elif branch == "final":
        task = f"Predict {contract.endpoint_name} from the query prior alone."
        instructions = [
            *contract.task_instructions,
            "There is no retrieved analog evidence in this None control. Use only the supplied query properties and single-molecule analysis.",
            "Ignore recognized identities and memorized experimental labels; make a structural and mechanistic prediction.",
        ]
        schema = {
            contract.prediction_field: f"{contract.positive_prediction} | {contract.negative_prediction}",
            "confidence": "high | moderate | low", "main_reasons": ["string"],
            "single_molecule_assessment": "string", "conflicting_evidence": ["string"],
            "evidence_gaps": ["string"], "final_summary": "string",
        }
    else:
        raise ValueError(f"Unknown query prior branch: {branch}")
    payload = {"task": task, "query": query, "instructions": instructions, "required_json_schema": schema}
    if single is not None:
        payload["single_molecule_analysis"] = single
    return [
        {"role": "system", "content": f"{contract.system_role} Return one complete JSON object."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
