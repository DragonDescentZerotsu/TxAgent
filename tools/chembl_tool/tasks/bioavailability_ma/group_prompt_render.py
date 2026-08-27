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
    public_assay_transfer_families,
    public_assay_transfer_score,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    full_record_example,
)
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_field_policy import included_fields

TEMPLATE_DIR = Path(__file__).with_name("group_prompt_templates")
INSTRUCTIONS_DIR = Path(__file__).with_name("prompt_instructions")
GROUP_DESCRIPTIONS_PATH = INSTRUCTIONS_DIR / "group_descriptions.md"

SUPPORTED_FORMATS = ("morganfingerprint", "assay_transfer_tool")
GROUP_OUTPUT_SCHEMA_PROFILES = ("legacy", "assay-transfer")
GROUP_PROMPT_VERSIONS = ("legacy_unversioned", "bioavailability_text_v1")
DEFAULT_GROUP_PROMPT_VERSION = "legacy_unversioned"


def _prompt_asset_paths(
    prompt_format: str,
    *,
    prompt_version: str,
    output_schema_profile: str,
) -> tuple[Path, Path]:
    if prompt_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unknown text group-prompt format: {prompt_format!r}")
    if prompt_version not in GROUP_PROMPT_VERSIONS:
        raise ValueError(f"Unknown Bioavailability group-prompt version: {prompt_version!r}")
    version_dir = "" if prompt_version == "legacy_unversioned" else prompt_version
    instruction_name = (
        "assay_transfer_tool_assay_transfer_schema.txt"
        if prompt_format == "assay_transfer_tool"
        and output_schema_profile == "assay-transfer"
        else f"{prompt_format}.txt"
    )
    return (
        TEMPLATE_DIR / version_dir / f"{prompt_format}.jinja",
        INSTRUCTIONS_DIR / version_dir / instruction_name,
    )


def instruction_file_provenance(
    prompt_format: str,
    instructions_file: str | Path | None = None,
    *,
    output_schema_profile: str = "legacy",
    prompt_version: str = DEFAULT_GROUP_PROMPT_VERSION,
) -> dict[str, Any]:
    """Resolve, validate, and fingerprint one editable prompt-instruction file."""
    if instructions_file:
        path = Path(instructions_file)
    elif prompt_format not in SUPPORTED_FORMATS:
        # Final-synthesis assets predate versioned group prompts and retain their
        # historical root-level names.
        path = INSTRUCTIONS_DIR / f"{prompt_format}.txt"
    else:
        _, path = _prompt_asset_paths(
            prompt_format,
            prompt_version=prompt_version,
            output_schema_profile=output_schema_profile,
        )
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
    prompt_version: str = DEFAULT_GROUP_PROMPT_VERSION,
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
            prompt_version=prompt_version,
        )["instructions"]
    )


def group_prompt_provenance(
    prompt_format: str,
    *,
    prompt_version: str = DEFAULT_GROUP_PROMPT_VERSION,
    instructions_file: str | Path | None = None,
    output_schema_profile: str = "legacy",
) -> dict[str, Any]:
    """Fingerprint the complete rendered prompt contract, including its Jinja."""
    template_path, _ = _prompt_asset_paths(
        prompt_format,
        prompt_version=prompt_version,
        output_schema_profile=output_schema_profile,
    )
    instruction = instruction_file_provenance(
        prompt_format,
        instructions_file,
        output_schema_profile=output_schema_profile,
        prompt_version=prompt_version,
    )
    template_raw = template_path.read_bytes()
    return {
        "prompt_contract_version": "bioavailability_group_prompt.v1",
        "prompt_version": prompt_version,
        "prompt_format": prompt_format,
        "template_path": str(template_path.resolve()),
        "template_sha256": hashlib.sha256(template_raw).hexdigest(),
        "instructions_path": instruction["path"],
        "instructions_sha256": instruction["sha256"],
        "instruction_count": instruction["instruction_count"],
        **group_output_schema_provenance(output_schema_profile),
    }


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
    system_role: str | None = None,
) -> str:
    """System message shared with the legacy branch (kept byte-identical there)."""
    prefetched = group.get("tools_prefetched") or (
        group.get("identity_blind") and group_tools_enabled
    )
    if prefetched:
        middle = "Use the harness-prefetched comparison results; do not call tools. " + (
            "Do not infer query identity. " if group.get("identity_blind") else ""
        )
    elif not group_tools_enabled:
        middle = "No tools are available for this branch. "
        if group.get("identity_blind"):
            middle += "Do not infer query identity. "
        if use_assay_transfer_likelihoods:
            middle += (
                "Use the supplied assay-transfer likelihoods as the best available "
                "transfer estimates. "
            )
    else:
        middle = "You may call the provided molecule comparison tools when structural or property differences matter. "
    return (
        (
            system_role
            or (
                "You are a medicinal chemistry oral bioavailability analog evidence analyst. "
                "Reason about whether analog evidence for one aspect of oral bioavailability is transferable to the query molecule. "
            )
        )
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
            if key == "source_fields":
                parts = [
                    f"{k}: {v if v not in (None, '', [], {}) else 'not reported'}"
                    for k, v in value.items()
                ]
            else:
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
    if source["canonical_smiles"] in {"[hidden]", "[identity hidden]"}:
        source["canonical_smiles"] = ""
    if with_transfer_score and "transfer_selection_score" in neighbor:
        source["assay_transfer_score"] = public_assay_transfer_score(neighbor)
    return source


def _evidence_records(
    neighbor: dict[str, Any], dataset: str, style: str = "legacy"
) -> list[list[tuple[str, str]]]:
    pairs = included_fields("morganfingerprint.record", dataset, style)
    records: list[list[tuple[str, str]]] = []
    for row in neighbor.get("evidence_rows") or []:
        for example in evidence_for_llm(row).get("examples") or []:
            rendered = _render_fields(example, pairs)
            if rendered:
                records.append(rendered)
    return records


def _assay_transfer_evidence_record(
    selected_record: dict[str, Any], dataset: str, group: dict[str, Any], style: str = "legacy"
) -> list[tuple[str, str]]:
    """Normalize one selected catalog record through minimal_evidence.v1 for display."""
    source_contract = selected_record.get("source_contract")
    source_fields = selected_record.get("source_fields")
    if source_contract or source_fields:
        if not isinstance(source_contract, dict) or not isinstance(source_fields, dict):
            raise ValueError("assay-transfer winning record has an incomplete source projection")
        example = {
            "source_contract": source_contract,
            "source_fields": source_fields,
            **(
                {
                    "resolved_measurement_display": selected_record[
                        "resolved_measurement_display"
                    ]
                }
                if selected_record.get("resolved_measurement_display")
                else {}
            ),
        }
    elif dataset == "Starling normalized oral bioavailability":
        raise ValueError(
            "normalized Starling assay-transfer record lacks its source projection; "
            "canonical display fallback is forbidden"
        )
    else:
        example = {
            "endpoint_type": selected_record.get("canonical_endpoint_key")
            or selected_record.get("endpoint_subtype")
            or selected_record.get("measurement_label"),
            "reported_value": selected_record.get("value_display", selected_record.get("value")),
            "reported_units": selected_record.get("unit_basis"),
            "context": selected_record.get("context") or {},
            "support_text": selected_record.get("support_text"),
        }
        # Legacy catalogs do not yet carry an explicit source projection.
        example.update(full_record_example(selected_record))
    normalized = evidence_for_llm(
        {
            "evidence_source": dataset or selected_record.get("source_id") or "unknown",
            "canonical_smiles": selected_record.get("canonical_smiles")
            or selected_record.get("original_smiles"),
            "group_id": group.get("group_id"),
            "tier": group.get("tier"),
            "endpoint_group": group.get("endpoint_group"),
            "standard_type": example.get("endpoint_type", "source_record"),
            "standard_value": example.get("reported_value", ""),
            "standard_units": example.get("reported_units", ""),
            "evidence_text": example.get("support_text", "source-contracted record"),
            "source_record_examples": [example],
        }
    )
    normalized_example = (normalized.get("examples") or [{}])[0]
    return _render_fields(
        normalized_example,
        [
            pair
            for pair in included_fields("morganfingerprint.record", dataset, style)
            if pair[0] != "source_contract"
        ],
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


def _resolve_evidence_source(group: dict[str, Any]) -> str:
    """The neighbor `evidence_source` (used to key the field policy). In the retrieval
    structure it lives on each evidence row, so fall back there when the group/neighbor
    level doesn't carry it."""
    if group.get("evidence_source"):
        return str(group["evidence_source"])
    for neighbor in group.get("neighbors") or []:
        if neighbor.get("evidence_source"):
            return str(neighbor["evidence_source"])
        for row in neighbor.get("evidence_rows") or []:
            if row.get("evidence_source"):
                return str(row["evidence_source"])
            source = (evidence_for_llm(row).get("source") or {}).get("name")
            if source:
                return str(source)
    return ""


def _group_meta(group: dict[str, Any]) -> dict[str, str]:
    group_id = group.get("group_id", "")
    return {
        "group_id": group_id,
        "tier": group.get("tier", ""),
        "endpoint_group": group.get("endpoint_group", ""),
        "evidence_source": _resolve_evidence_source(group),
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
    prompt_version: str = DEFAULT_GROUP_PROMPT_VERSION,
    style: str = "legacy",
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
                "records": _evidence_records(neighbor, dataset, style),
            }
        )
    return {
        "instructions": load_instructions(
            "morganfingerprint",
            instructions_file,
            output_schema_profile=output_schema_profile,
            prompt_version=prompt_version,
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
    prompt_version: str = DEFAULT_GROUP_PROMPT_VERSION,
    style: str = "legacy",
) -> dict[str, Any]:
    dataset = _dataset_key(group)
    header_pairs = included_fields("assay_transfer_tool.neighbor", dataset)
    neighbors_ctx = []
    for neighbor in group.get("neighbors") or []:
        families = public_assay_transfer_families(neighbor, group)
        neighbors_ctx.append(
            {
                "rank": neighbor.get("rank"),
                "header": _render_fields(
                    _neighbor_header_source(neighbor, with_transfer_score=False),
                    header_pairs,
                ),
                "families": [
                    {
                        "group_id": family["group_id"],
                        "family_rank": family["family_rank"],
                        "score_kind": family["score_kind"],
                        "assay_transfer_score": family["assay_transfer_score"],
                        "records": [
                            {
                                "rank": record["record_rank"],
                                "assay_transfer_score": record["assay_transfer_score"],
                                "record": _assay_transfer_evidence_record(
                                    record["record"],
                                    dataset,
                                    {
                                        "group_id": family["group_id"],
                                        "tier": family["tier"],
                                        "endpoint_group": family["endpoint_group"],
                                    },
                                    style,
                                ),
                            }
                            for record in family["records"]
                            if record["record"]
                        ],
                    }
                    for family in families
                ],
            }
        )
    return {
        "instructions": load_instructions(
            "assay_transfer_tool",
            instructions_file,
            output_schema_profile=output_schema_profile,
            prompt_version=prompt_version,
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
    prompt_version = str(
        options.get("prompt_version") or DEFAULT_GROUP_PROMPT_VERSION
    )
    style = str(options.get("presentation_style", "legacy"))
    if prompt_format == "morganfingerprint":
        context = _build_morgan_context(
            query,
            group,
            min_similarity=float(options.get("prompt_min_similarity", 0.0)),
            instructions_file=options.get("instructions_file"),
            output_schema_profile=output_schema_profile,
            prompt_version=prompt_version,
            style=style,
        )
        template_path, _ = _prompt_asset_paths(
            prompt_format,
            prompt_version=prompt_version,
            output_schema_profile=output_schema_profile,
        )
    elif prompt_format == "assay_transfer_tool":
        context = _build_assay_transfer_context(
            query,
            group,
            instructions_file=options.get("instructions_file"),
            output_schema_profile=output_schema_profile,
            prompt_version=prompt_version,
            style=style,
        )
        # The layout is assay-transfer-specific, but record fields still come from
        # the shared minimal_evidence.v1 / morganfingerprint.record policy.
        template_path, _ = _prompt_asset_paths(
            prompt_format,
            prompt_version=prompt_version,
            output_schema_profile=output_schema_profile,
        )
    else:
        raise ValueError(f"Unknown text group-prompt format: {prompt_format!r}")
    context["instructions"] = [
        *list(options.get("additional_instructions") or []),
        *context["instructions"],
    ]
    if options.get("omit_query_tools"):
        context["instructions"] = [
            line
            for line in context["instructions"]
            if "mmp_structure_compare" not in line
            and "properties_compare" not in line
            and "molecule_properties" not in line
        ]
    user_content = _env().get_template(
        str(template_path.relative_to(TEMPLATE_DIR))
    ).render(**context)
    return [
        {
            "role": "system",
            "content": group_system_message(
                group,
                group_tools_enabled=bool(options.get("group_tools_enabled", True)),
                use_assay_transfer_likelihoods=prompt_format == "assay_transfer_tool",
                system_role=options.get("system_role"),
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

ANALOGOUS_REASONING_ONLY_FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    key: value
    for key, value in FINAL_OUTPUT_SCHEMA.items()
    if key != "single_molecule_assessment"
}

FINAL_SYSTEM_MESSAGE = (
    "You are a senior oral bioavailability reasoning model. Integrate group-level analog "
    "evidence into one final oral bioavailability assessment. Return only valid JSON."
)

ANALOGOUS_REASONING_ONLY_FINAL_SYSTEM_MESSAGE = (
    "You are a senior oral bioavailability analog-evidence synthesis model. "
    "Use only the supplied mechanism-branch analog analyses to produce one final "
    "oral bioavailability assessment. Return only valid JSON."
)


def final_prompt_provenance(*, analogous_reasoning_only: bool) -> dict[str, Any]:
    """Fingerprint the selected final prompt instructions, template, and schema."""
    profile = "analogous_reasoning_only" if analogous_reasoning_only else "standard"
    stem = "final_analogous_reasoning_only" if analogous_reasoning_only else "final"
    instruction_path = INSTRUCTIONS_DIR / f"{stem}.txt"
    template_path = TEMPLATE_DIR / f"{stem}.jinja"
    schema = (
        ANALOGOUS_REASONING_ONLY_FINAL_OUTPUT_SCHEMA
        if analogous_reasoning_only
        else FINAL_OUTPUT_SCHEMA
    )
    contract = {
        "profile": profile,
        "contract_version": f"bioavailability_final_prompt.{profile}.v1",
        "instructions_sha256": hashlib.sha256(instruction_path.read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(
                schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return contract


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
    analogous_reasoning_only: bool = False,
) -> list[dict[str, str]]:
    """Compile the final-synthesis [system, user] messages as clean text (no LLM needed)."""
    instruction_name = (
        "final_analogous_reasoning_only" if analogous_reasoning_only else "final"
    )
    instructions = [
        line.replace("{high_f_cutoff}", f"{high_f_cutoff:g}")
        for line in load_instructions(instruction_name)
    ]
    context = {
        "instructions": instructions,
        "groups": [
            {
                "group_id": item.get("group_id"),
                "status": item.get("status"),
                "text": _branch_text(item),
            }
            for item in group_outputs
        ],
        "output_schema": json.dumps(
            (
                ANALOGOUS_REASONING_ONLY_FINAL_OUTPUT_SCHEMA
                if analogous_reasoning_only
                else FINAL_OUTPUT_SCHEMA
            ),
            indent=2,
            ensure_ascii=False,
        ),
    }
    if analogous_reasoning_only:
        template_name = "final_analogous_reasoning_only.jinja"
        system_message = ANALOGOUS_REASONING_ONLY_FINAL_SYSTEM_MESSAGE
    else:
        context["query_smiles"] = _query_smiles(retrieval.get("query") or {})
        context["coverage_text"] = _content_to_text(retrieval.get("coverage") or {})
        context["single_text"] = _branch_text(single_output)
        template_name = "final.jinja"
        system_message = FINAL_SYSTEM_MESSAGE
    user_content = _env().get_template(template_name).render(**context)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
