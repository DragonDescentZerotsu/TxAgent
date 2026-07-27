"""Text (non-JSON) group-prompt renderers for the new sub-branch formats.

Two formats are produced here, selected by `prompt_format`:

* ``morganfingerprint``  -- molecules + Morgan similarity/label + their records.
* ``assay_transfer_tool`` -- top-k assay records + transfer score,
  rendered through the same minimal-evidence record presentation as Morgan retrieval.

Both put the instruction block at the top and end with a selected required JSON
output schema. The default ``legacy`` schema remains unchanged; the
``assay-transfer`` profile is evidence-centric and omits molecule identity and rigid
direction/transferability enums. The legacy JSON prompt format is not handled here;
the pipeline keeps it inline.

Which metadata fields appear is decided entirely by ``group_prompt_field_policy`` --
this module only formats what that policy selects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.reasoning_validation import validated_branch_content
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_prompt_policy import (
    public_assay_transfer_score,
)
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_field_policy import included_fields

TEMPLATE_DIR = Path(__file__).with_name("group_prompt_templates")
INSTRUCTIONS_DIR = Path(__file__).with_name("prompt_instructions")
GROUP_DESCRIPTIONS_PATH = INSTRUCTIONS_DIR / "group_descriptions.md"

SUPPORTED_FORMATS = ("morganfingerprint", "assay_transfer_tool")
GROUP_OUTPUT_SCHEMA_PROFILES = ("legacy", "assay-transfer")


def instruction_file_provenance(
    prompt_format: str,
    instructions_file: str | Path | None = None,
    *,
    output_schema_profile: str = "legacy",
) -> dict[str, Any]:
    """Resolve, validate, and fingerprint one editable prompt-instruction file."""
    if instructions_file:
        path = Path(instructions_file)
    elif (
        prompt_format == "assay_transfer_tool"
        and output_schema_profile == "assay-transfer"
    ):
        path = INSTRUCTIONS_DIR / "assay_transfer_tool_assay_transfer_schema.txt"
    else:
        path = INSTRUCTIONS_DIR / f"{prompt_format}.txt"
    try:
        resolved = path.expanduser().resolve(strict=True)
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read group prompt instructions file {path}: {exc}") from exc
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Group prompt instructions file has no instruction lines: {resolved}")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "instruction_count": len(lines),
        "instructions": lines,
    }


def load_instructions(
    prompt_format: str,
    instructions_file: str | Path | None = None,
    *,
    output_schema_profile: str = "legacy",
) -> list[str]:
    """Load the editable, numbered instruction lines for a group prompt format.

    Instructions live in `prompt_instructions/<format>.txt` (one per line; blank lines
    and `#` comments ignored) so they can be edited without touching this module.
    """
    return list(
        instruction_file_provenance(
            prompt_format,
            instructions_file,
            output_schema_profile=output_schema_profile,
        )["instructions"]
    )


# The historical group-output contract. Keep this object unchanged so the default
# profile, prompt goldens, and completed runs remain comparable.
LEGACY_GROUP_OUTPUT_SCHEMA: dict[str, Any] = {
    "useful_for_bioavailability_reasoning": "boolean",
    "transferability": "high | moderate | low | not_applicable",
    "evidence_direction": (
        "supports_high_bioavailability | argues_against_high_bioavailability | absorption_support | "
        "permeability_support | solubility_support | solubility_risk | metabolic_stability_support | "
        "first_pass_or_clearance_risk | transporter_efflux_risk | neutral_or_unclear"
    ),
    "confidence": "high | moderate | low",
    "reasoning_summary": "string",
    "key_evidence": [
        {
            "molecule_chembl_id": "string",
            "similarity": "number or null",
            "similarity_bucket": "string",
            "assay_signal": "string",
            "activity_values": ["string"],
            "tool_summary": "string",
            "transferability": "high | moderate | low | not_applicable",
            "effect_on_bioavailability_reasoning": "string",
        }
    ],
    "caveats": ["string"],
}
GROUP_OUTPUT_SCHEMA = LEGACY_GROUP_OUTPUT_SCHEMA

ASSAY_TRANSFER_GROUP_OUTPUT_SCHEMA: dict[str, Any] = {
    "useful_for_bioavailability_reasoning": "boolean",
    "confidence": "high | moderate | low",
    "assay_transfer_assessment": "string",
    "bioavailability_implications": ["string"],
    "reasoning_summary": "string",
    "key_evidence": [
        {
            "record_rank": "integer or null",
            "assay_endpoint": "string",
            "transfer_likelihood": "number or null",
            "assay_observation": "string",
            "bioavailability_implication": "string",
            "limitations": ["string"],
        }
    ],
    "caveats": ["string"],
}


def group_output_schema(profile: str) -> dict[str, Any]:
    if profile == "legacy":
        return LEGACY_GROUP_OUTPUT_SCHEMA
    if profile == "assay-transfer":
        return ASSAY_TRANSFER_GROUP_OUTPUT_SCHEMA
    raise ValueError(f"Unknown group output schema profile: {profile!r}")


def group_output_validation(profile: str) -> dict[str, Any]:
    """Return the response validator contract paired with one rendered schema."""
    if profile == "legacy":
        return {
            "required_fields": (
                "transferability",
                "confidence",
                "reasoning_summary",
            ),
            "allowed_values": None,
            "forbidden_field_names": (),
        }
    if profile == "assay-transfer":
        return {
            "required_fields": (
                "useful_for_bioavailability_reasoning",
                "confidence",
                "assay_transfer_assessment",
                "bioavailability_implications",
                "reasoning_summary",
            ),
            "allowed_values": {
                "confidence": {"high", "moderate", "low"},
            },
            "forbidden_field_names": ("molecule_chembl_id",),
        }
    raise ValueError(f"Unknown group output schema profile: {profile!r}")


def group_output_schema_provenance(profile: str) -> dict[str, Any]:
    schema = group_output_schema(profile)
    serialized = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "profile": profile,
        "contract_version": (
            "bioavailability_group_output.legacy.v1"
            if profile == "legacy"
            else "bioavailability_group_output.assay_transfer.v1"
        ),
        "schema_sha256": hashlib.sha256(serialized).hexdigest(),
    }

def group_system_message(
    group: dict[str, Any],
    *,
    group_tools_enabled: bool = True,
    use_assay_transfer_likelihoods: bool = False,
) -> str:
    """System message shared with the legacy branch (kept byte-identical there)."""
    prefetched = group.get("tools_prefetched") or group.get("identity_blind")
    if prefetched:
        middle = "Use the harness-prefetched comparison results; do not call tools. " + (
            "Do not infer query identity. " if group.get("identity_blind") else ""
        )
    elif not group_tools_enabled:
        middle = "No tools are available for this branch. "
        if use_assay_transfer_likelihoods:
            middle += (
                "Use the supplied assay-transfer likelihoods as the best available "
                "transfer estimates. "
            )
    else:
        middle = "You may call the provided molecule comparison tools when structural or property differences matter. "
    return (
        "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
        "Reason about whether analog evidence for one aspect of oral bioavailability is transferable to the query molecule. "
        + middle
        + "Return only valid JSON."
    )


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_fields(source: dict[str, Any], pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return ordered (label, value) rows for included, non-empty fields."""
    rows: list[tuple[str, str]] = []
    for key, label in pairs:
        value = source.get(key)
        if isinstance(value, dict):
            parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, "", [], {})]
            if not parts:
                continue
            value = "; ".join(parts)
        if value in (None, "", [], {}):
            continue
        rows.append((label, str(value)))
    return rows


def _query_smiles(query: dict[str, Any]) -> str:
    if query.get("identity_hidden"):
        return "[identity hidden]"
    return str(query.get("canonical_smiles") or query.get("input_smiles") or "")


def _neighbor_header_source(neighbor: dict[str, Any], *, with_transfer_score: bool) -> dict[str, Any]:
    source = {
        "molecule_chembl_id": neighbor.get("molecule_chembl_id", ""),
        "canonical_smiles": neighbor.get("canonical_smiles", ""),
        "similarity": neighbor.get("similarity"),
        "similarity_bucket": neighbor.get("similarity_bucket", ""),
    }
    if with_transfer_score and "transfer_selection_score" in neighbor:
        source["assay_transfer_score"] = public_assay_transfer_score(neighbor)
    return source


def _evidence_records(neighbor: dict[str, Any], dataset: str) -> list[list[tuple[str, str]]]:
    pairs = included_fields("morganfingerprint.record", dataset)
    records: list[list[tuple[str, str]]] = []
    for row in neighbor.get("evidence_rows") or []:
        for example in evidence_for_llm(row).get("examples") or []:
            rendered = _render_fields(example, pairs)
            if rendered:
                records.append(rendered)
    return records


def _assay_transfer_evidence_record(
    selected_record: dict[str, Any], dataset: str, group: dict[str, Any]
) -> list[tuple[str, str]]:
    """Normalize one selected catalog record through minimal_evidence.v1 for display."""
    example = {
        "endpoint_type": selected_record.get("canonical_endpoint_key")
        or selected_record.get("endpoint_subtype")
        or selected_record.get("measurement_label"),
        "reported_value": selected_record.get("value_display", selected_record.get("value")),
        "reported_units": selected_record.get("unit_basis"),
        "context": selected_record.get("context") or {},
        "support_text": selected_record.get("support_text"),
    }
    normalized = evidence_for_llm(
        {
            "evidence_source": dataset or selected_record.get("source_id") or "unknown",
            "canonical_smiles": selected_record.get("canonical_smiles")
            or selected_record.get("original_smiles"),
            "group_id": group.get("group_id"),
            "tier": group.get("tier"),
            "endpoint_group": group.get("endpoint_group"),
            "standard_type": example["endpoint_type"],
            "standard_value": example["reported_value"],
            "standard_units": example["reported_units"],
            "evidence_text": example["support_text"],
            "source_record_examples": [example],
        }
    )
    normalized_example = (normalized.get("examples") or [{}])[0]
    return _render_fields(
        normalized_example,
        included_fields("morganfingerprint.record", dataset),
    )


def load_group_description(group_id: str) -> str:
    """Return the editable natural-language description of a branch, or the group id.

    Descriptions live in `prompt_instructions/group_descriptions.md` as
    `<group_id> = <description>` lines (blank lines and `#` comments ignored).
    """
    for line in GROUP_DESCRIPTIONS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or " = " not in line:
            continue
        key, _, description = line.partition(" = ")
        if key.strip() == group_id:
            return description.strip()
    return group_id


def _group_meta(group: dict[str, Any]) -> dict[str, str]:
    group_id = group.get("group_id", "")
    return {
        "group_id": group_id,
        "tier": group.get("tier", ""),
        "endpoint_group": group.get("endpoint_group", ""),
        "evidence_source": group.get("evidence_source")
        or (group.get("neighbors") or [{}])[0].get("evidence_source", ""),
        "description": load_group_description(group_id),
    }


def _dataset_key(group: dict[str, Any]) -> str:
    meta = _group_meta(group)
    return meta.get("evidence_source", "")


def _build_morgan_context(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    min_similarity: float,
    instructions_file: str | Path | None = None,
    output_schema_profile: str = "legacy",
) -> dict[str, Any]:
    dataset = _dataset_key(group)
    header_pairs = included_fields("morganfingerprint.neighbor", dataset)
    neighbors_ctx = []
    for neighbor in group.get("neighbors") or []:
        similarity = neighbor.get("similarity")
        if similarity is not None and float(similarity) < float(min_similarity):
            continue
        neighbors_ctx.append(
            {
                "rank": neighbor.get("rank"),
                "header": _render_fields(
                    _neighbor_header_source(neighbor, with_transfer_score=False), header_pairs
                ),
                "records": _evidence_records(neighbor, dataset),
            }
        )
    return {
        "instructions": load_instructions(
            "morganfingerprint",
            instructions_file,
            output_schema_profile=output_schema_profile,
        ),
        "group": _group_meta(group),
        "query_smiles": _query_smiles(query),
        "neighbors": neighbors_ctx,
        "output_schema": json.dumps(
            group_output_schema(output_schema_profile),
            indent=2,
            ensure_ascii=False,
        ),
    }


def _build_assay_transfer_context(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    instructions_file: str | Path | None = None,
    output_schema_profile: str = "legacy",
) -> dict[str, Any]:
    dataset = _dataset_key(group)
    header_pairs = included_fields("assay_transfer_tool.neighbor", dataset)
    neighbors_ctx = []
    for neighbor in group.get("neighbors") or []:
        selected_record = neighbor.get("transfer_winning_record")
        neighbors_ctx.append(
            {
                "rank": neighbor.get("rank"),
                "header": _render_fields(
                    _neighbor_header_source(neighbor, with_transfer_score=True), header_pairs
                ),
                "selected_record": (
                    _assay_transfer_evidence_record(selected_record, dataset, group)
                    if selected_record
                    else []
                ),
            }
        )
    return {
        "instructions": load_instructions(
            "assay_transfer_tool",
            instructions_file,
            output_schema_profile=output_schema_profile,
        ),
        "group": _group_meta(group),
        "query_smiles": _query_smiles(query),
        "neighbors": neighbors_ctx,
        "output_schema": json.dumps(
            group_output_schema(output_schema_profile),
            indent=2,
            ensure_ascii=False,
        ),
    }


def build_group_messages(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    prompt_format: str,
    options: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return [system, user] messages for a new text group-prompt format."""
    options = options or {}
    output_schema_profile = str(options.get("output_schema_profile", "legacy"))
    if prompt_format == "morganfingerprint":
        context = _build_morgan_context(
            query,
            group,
            min_similarity=float(options.get("prompt_min_similarity", 0.0)),
            instructions_file=options.get("instructions_file"),
            output_schema_profile=output_schema_profile,
        )
        template = "morganfingerprint.jinja"
    elif prompt_format == "assay_transfer_tool":
        context = _build_assay_transfer_context(
            query,
            group,
            instructions_file=options.get("instructions_file"),
            output_schema_profile=output_schema_profile,
        )
        # The layout is assay-transfer-specific, but record fields still come from
        # the shared minimal_evidence.v1 / morganfingerprint.record policy.
        template = "assay_transfer_tool.jinja"
    else:
        raise ValueError(f"Unknown text group-prompt format: {prompt_format!r}")
    user_content = _env().get_template(template).render(**context)
    return [
        {
            "role": "system",
            "content": group_system_message(
                group,
                group_tools_enabled=bool(options.get("group_tools_enabled", True)),
                use_assay_transfer_likelihoods=prompt_format == "assay_transfer_tool",
            ),
        },
        {"role": "user", "content": user_content},
    ]


# --- Final synthesis stage -------------------------------------------------------

FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "bioavailability_prediction": "high | low",
    "confidence": "high | moderate | low",
    "main_reasons": ["string"],
    "single_molecule_assessment": "string",
    "absorption_and_permeability_assessment": "string",
    "solubility_and_dissolution_assessment": "string",
    "metabolism_first_pass_and_clearance_assessment": "string",
    "transporter_efflux_assessment": "string",
    "direct_oral_bioavailability_analog_assessment": "string",
    "conflicting_evidence": ["string"],
    "evidence_gaps": ["string"],
    "final_summary": "string",
}

FINAL_SYSTEM_MESSAGE = (
    "You are a senior oral bioavailability reasoning model. Integrate group-level analog "
    "evidence into one final oral bioavailability assessment. Return only valid JSON."
)


def _content_to_text(obj: Any, indent: int = 0) -> str:
    """Render a nested analysis content dict as clean indented key: value text."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.append(_content_to_text(value, indent + 1))
            elif isinstance(value, list):
                if all(not isinstance(item, (dict, list)) for item in value):
                    lines.append(f"{pad}{key}: " + "; ".join(str(item) for item in value))
                else:
                    lines.append(f"{pad}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"{pad}  -")
                            lines.append(_content_to_text(item, indent + 2))
                        else:
                            lines.append(f"{pad}  - {item}")
            else:
                lines.append(f"{pad}{key}: {value}")
    else:
        lines.append(f"{pad}{obj}")
    return "\n".join(line for line in lines if line)


def _branch_text(branch_output: dict[str, Any]) -> str:
    content = validated_branch_content(branch_output)
    return _content_to_text(content) if content else "(no valid analysis returned)"


def build_final_messages(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    *,
    high_f_cutoff: float,
) -> list[dict[str, str]]:
    """Compile the final-synthesis [system, user] messages as clean text (no LLM needed)."""
    instructions = [
        line.replace("{high_f_cutoff}", f"{high_f_cutoff:g}")
        for line in load_instructions("final")
    ]
    context = {
        "instructions": instructions,
        "query_smiles": _query_smiles(retrieval.get("query") or {}),
        "coverage_text": _content_to_text(retrieval.get("coverage") or {}),
        "single_text": _branch_text(single_output),
        "groups": [
            {
                "group_id": item.get("group_id"),
                "status": item.get("status"),
                "text": _branch_text(item),
            }
            for item in group_outputs
        ],
        "output_schema": json.dumps(FINAL_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
    }
    user_content = _env().get_template("final.jinja").render(**context)
    return [
        {"role": "system", "content": FINAL_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
