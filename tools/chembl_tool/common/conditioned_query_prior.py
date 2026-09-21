"""Task-contract-driven structure and property query-prior messages."""

from __future__ import annotations

import json

from tools.chembl_tool.common.progressive_assay_reasoning import ProgressiveTaskContract


def build_query_prior_messages(
    contract: ProgressiveTaskContract, query: dict,
) -> list[dict]:
    payload = {
        "task": f"Assess the query molecule's structural and physicochemical prior for {contract.endpoint_name}.",
        "query": query,
        "instructions": [
            "Use only the supplied structure and property-tool text, without analog evidence or memorized assay labels.",
            "Explain plausible structural and mechanistic liabilities, exposure limitations, and uncertainty.",
            "A structural alert is a hypothesis, not an observed endpoint. Absence of an alert does not establish a negative result.",
            "Do not invent species, population, strain, activation, exposure, or organism conditions.",
        ],
        "required_json_schema": {
            "endpoint_prior": (
                f"{contract.positive_prediction} | {contract.negative_prediction} | mixed_or_unclear"
            ),
            "property_drivers": ["string"],
            "confidence": "high | moderate | low",
            "reasoning_summary": "string",
            "caveats": ["string"],
        },
    }
    return [
        {
            "role": "system",
            "content": f"{contract.system_role} Return one complete JSON object.",
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
