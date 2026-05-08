from __future__ import annotations

import math
from typing import Any

from tools.service.config import ServiceSettings
from tools.service.tools.base import BaseTool
from tools.service.tools.rdkit_properties import MoleculePropertiesTool


def _is_number(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _round_number(value: object) -> float | None:
    if not _is_number(value):
        return None
    return round(float(value), 2)


def _format_number(value: object, *, signed: bool = False) -> str:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        integer_value = int(round(numeric))
        return f"{integer_value:+d}" if signed else str(integer_value)
    rendered = f"{numeric:+.2f}" if signed else f"{numeric:.2f}"
    rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"-0", "+0"}:
        return "+0" if signed else "0"
    return rendered


def _format_delta(value: object) -> str:
    if value is None:
        return "not applicable"
    return _format_number(value, signed=True)


class PropertiesCompareTool(BaseTool):
    name = "properties_compare"
    version = "v1"
    description = "Compare all molecule_properties numeric and missing-value properties for two molecules, excluding functional groups."
    input_schema = {
        "type": "object",
        "properties": {
            "query_smiles": {"type": "string"},
            "reference_smiles": {"type": "string"},
            "logd_ph": {"type": "number", "default": 7.4},
        },
        "required": ["query_smiles", "reference_smiles"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "object"},
            "reference": {"type": "object"},
            "feature_comparisons": {"type": "array"},
            "text": {"type": "string"},
        },
    }

    def __init__(self, properties_tool: MoleculePropertiesTool | None = None) -> None:
        super().__init__()
        self.properties_tool = properties_tool or MoleculePropertiesTool()

    def initialize(self, settings: ServiceSettings) -> None:
        if not self.properties_tool.initialized:
            self.properties_tool.initialize(settings)
        self.initialized = True
        self.initialization_error = self.properties_tool.initialization_error

    def invoke(self, payload: dict[str, Any], *, return_debug: bool = False) -> dict[str, Any]:
        query_smiles = str(payload.get("query_smiles") or "").strip()
        reference_smiles = str(payload.get("reference_smiles") or payload.get("neighbor_smiles") or "").strip()
        logd_ph = payload.get("logd_ph")
        property_payload = {}
        if logd_ph is not None:
            property_payload["logd_ph"] = logd_ph

        query_output = self.properties_tool.invoke(
            {"query_smiles": query_smiles, **property_payload},
            return_debug=return_debug,
        )
        reference_output = self.properties_tool.invoke(
            {"query_smiles": reference_smiles, **property_payload},
            return_debug=return_debug,
        )

        query_features = {feature["feature_name"]: feature for feature in query_output["features"]}
        reference_features = {feature["feature_name"]: feature for feature in reference_output["features"]}
        comparisons = []
        for query_feature in query_output["features"]:
            feature_name = query_feature["feature_name"]
            reference_feature = reference_features[feature_name]
            comparisons.append(self._comparison_payload(query_feature, reference_feature))

        warnings = []
        warnings.extend(query_output.pop("_warnings", []))
        warnings.extend(reference_output.pop("_warnings", []))

        output = {
            "query": query_output["query"],
            "reference": reference_output["query"],
            "feature_comparisons": comparisons,
            "text": self._render_text(comparisons),
            "_warnings": sorted(set(str(warning) for warning in warnings)),
        }
        if return_debug:
            output["debug"] = {
                "query_debug": query_output.get("debug"),
                "reference_debug": reference_output.get("debug"),
            }
        return output

    def _comparison_payload(
        self,
        query_feature: dict[str, Any],
        reference_feature: dict[str, Any],
    ) -> dict[str, Any]:
        query_value = query_feature.get("feature_value")
        reference_value = reference_feature.get("feature_value")
        delta_value = None
        if _is_number(query_value) and _is_number(reference_value):
            delta_value = _round_number(float(query_value) - float(reference_value))

        return {
            "feature_name": query_feature["feature_name"],
            "display_name": query_feature["display_name"],
            "description": query_feature["description"],
            "source_family": query_feature["source_family"],
            "raw_name": query_feature["raw_name"],
            "query_value": query_value,
            "query_value_text": query_feature.get("feature_value_text"),
            "query_value_missing_reason": query_feature.get("feature_value_missing_reason"),
            "reference_value": reference_value,
            "reference_value_text": reference_feature.get("feature_value_text"),
            "reference_value_missing_reason": reference_feature.get("feature_value_missing_reason"),
            "delta_value": delta_value,
            "delta_text": _format_delta(delta_value),
        }

    def _render_text(self, comparisons: list[dict[str, Any]]) -> str:
        lines = ["Definitions: delta means query value minus reference value."]
        for comparison in comparisons:
            lines.append(
                f"{comparison['display_name']}: "
                f"query={comparison['query_value_text']} | "
                f"reference={comparison['reference_value_text']} | "
                f"delta={comparison['delta_text']}"
            )
        return "\n".join(lines)

