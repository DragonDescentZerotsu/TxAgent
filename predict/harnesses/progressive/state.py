"""Append-only assay-card selection and progressive reasoning contracts.

This module deliberately knows nothing about benchmark file paths or model
endpoints.  It turns cumulative assay retrievals into a stable molecule/card
surface, selects the bounded evidence delta, renders the model prompt, and
validates the progressive state returned by the reasoning model.

Inputs are normalized retrieval dictionaries plus the previous structured
state. Outputs are deterministic card selections, prompt messages, validation
errors, or the next structured state. ``runner.py`` is the only harness
that orchestrates these operations; this module contains the reusable pure
state transformations so retrieval and model transport remain outside them.

``molecule_card.yaml`` is the authoritative model-visible JSON contract. It
maps the internal molecule and evidence-card state below into each entry of
``active_evidence``; fields beginning with ``_`` are selection metadata and
never reach the model unless the YAML explicitly names them as a source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from predict.harnesses.progressive.prompt import prompt_assets, render_progressive_messages


PROGRESSIVE_PROTOCOL_VERSION = "conditioned_assay_progressive_visible.v8"
MOLECULE_CARD_CONTRACT_PATH = Path(__file__).with_name("molecule_card.yaml")
INITIAL_MOLECULE_LIMIT = 10
INITIAL_CARD_LIMIT = 4
NEW_MOLECULE_LIMIT = 3
AUGMENTED_MOLECULE_LIMIT = 3
DELTA_CARD_LIMIT = 2

_CORE_PROPERTY_NAMES = (
    "neutral fraction",
    "estimated logD",
    "strongest acidic pKa",
    "strongest basic pKa",
    "number of ionizable sites",
    "exact molecular weight",
    "fraction of sp3 carbons",
    "estimated logP",
    "hydrogen-bond acceptor count",
    "hydrogen-bond donor count",
    "rotatable-bond count",
    "ring count",
    "topological polar surface area",
)


@dataclass(frozen=True)
class ProgressiveTaskContract:
    task: str
    endpoint_name: str
    label_scope: str
    prediction_field: str
    positive_prediction: str
    negative_prediction: str
    system_role: str
    task_instructions: tuple[str, ...]

    @property
    def prediction_values(self) -> set[str]:
        return {self.positive_prediction, self.negative_prediction}


def stable_analog_id(neighbor: Mapping[str, Any]) -> str:
    """Return a prompt-safe identifier that remains stable across levels."""
    identity = str(
        neighbor.get("standard_inchi_key")
        or neighbor.get("canonical_smiles")
        or neighbor.get("molecule_chembl_id")
        or ""
    )
    return "analog_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


@lru_cache(maxsize=1)
def molecule_card_contract(
    path: Path = MOLECULE_CARD_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and validate the YAML that defines model-visible molecule cards."""
    from predict.harnesses.progressive.prompt import prompt_directory
    active_path = prompt_directory('standard_v1') / 'card.yaml' if path == MOLECULE_CARD_CONTRACT_PATH else path
    contract = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema_version") != "progressive_molecule_card.v1":
        raise ValueError("molecule_card.yaml has an unsupported schema_version")
    required_outputs = {
        "molecule": {"analog_id", "canonical_smiles", "first_seen_level"},
        "evidence_card": {"card_id", "first_seen_level", "new_this_level"},
    }
    for section, required in required_outputs.items():
        fields = (contract.get(section) or {}).get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"molecule_card.yaml {section}.fields must be a non-empty list")
        names = [str(field.get("name") or "") for field in fields if isinstance(field, Mapping)]
        if len(names) != len(fields) or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"molecule_card.yaml {section}.fields has invalid or duplicate names")
        if not required <= set(names):
            raise ValueError(f"molecule_card.yaml {section}.fields must include {sorted(required)}")
        for field in fields:
            if not _clean(field.get("source")):
                raise ValueError(f"molecule_card.yaml field {field.get('name')!r} needs a source")
    molecule = contract["molecule"]
    for key in ("tool_summaries_field", "evidence_cards_field"):
        if not _clean(molecule.get(key)):
            raise ValueError(f"molecule_card.yaml molecule.{key} is required")
    molecule_outputs = {str(field["name"]) for field in molecule["fields"]}
    section_outputs = {
        str(molecule["tool_summaries_field"]),
        str(molecule["evidence_cards_field"]),
    }
    if len(section_outputs) != 2 or molecule_outputs.intersection(section_outputs):
        raise ValueError("molecule_card.yaml molecule output names must be unique")
    return contract


def _project_card_fields(
    values: Mapping[str, Any], fields: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the ordered YAML field projection and fail on missing required data."""
    projected: dict[str, Any] = {}
    for field in fields:
        name = str(field["name"])
        source = str(field["source"])
        value = values.get(source)
        if value in (None, "", []):
            if field.get("required") is True:
                raise ValueError(f"required molecule-card field is blank: {name}")
            continue
        projected[name] = value
    return projected


def _card_surface(example: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    card = example.get("_progressive_card") or example
    scope = row.get("evidence_scope") or {}
    assay_context = _clean(card.get("assay_context"))
    if not assay_context:
        contexts = scope.get("assay_context") or []
        assay_context = _clean(contexts[0]) if contexts else _clean(row.get("target_pref_name"))
    species = _clean(card.get("species_context"))
    if not species:
        species_values = scope.get("species_context") or []
        species = _clean(species_values[0]) if species_values else _clean(row.get("organism"))
    conditions = _clean(card.get("qualifying_conditions"))
    if not conditions:
        condition_values = scope.get("qualifying_conditions") or []
        conditions = _clean(condition_values[0]) if condition_values else ""
    return {
        "evidence_family": _clean(example.get("evidence_family")),
        "assay_context": assay_context,
        "endpoint": _clean(card.get("endpoint_type")) or _clean(row.get("standard_type")),
        "reported_value": _clean(card.get("reported_value")) or _clean(row.get("standard_value")),
        "reported_unit": _clean(card.get("reported_units")) or _clean(row.get("standard_units")),
        "species": species,
        "qualifying_conditions": conditions,
        "support_text": _clean(card.get("support_text")) or _clean(row.get("assay_description")),
    }


def _card_id(analog_id: str, surface: Mapping[str, Any]) -> str:
    # Exact prompt-visible duplicates for one molecule intentionally collapse,
    # even if they came through more than one physical assay.
    payload = json.dumps(
        {"analog_id": analog_id, **surface},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "card_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def extract_cumulative_evidence(retrieval: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract a de-duplicated, model-visible card set by stable analog id."""
    analogs: dict[str, dict[str, Any]] = {}
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            if not isinstance(neighbor, Mapping):
                continue
            analog_id = stable_analog_id(neighbor)
            analog = analogs.setdefault(
                analog_id,
                {
                    "analog_id": analog_id,
                    "canonical_smiles": _clean(neighbor.get("canonical_smiles")),
                    "similarity": float(neighbor.get("similarity") or 0.0),
                    "similarity_bucket": _clean(neighbor.get("similarity_bucket")),
                    "molecule_relation": _clean(neighbor.get("molecule_relation")),
                    "cards": {},
                },
            )
            for row in neighbor.get("evidence_rows") or []:
                if not isinstance(row, Mapping):
                    continue
                examples = [
                    item
                    for item in row.get("source_record_examples") or []
                    if isinstance(item, Mapping)
                ] or [{}]
                for example in examples:
                    surface = _card_surface(example, row)
                    card_id = _card_id(analog_id, surface)
                    if card_id in analog["cards"]:
                        continue
                    try:
                        family_level = int(example.get("evidence_family_level") or 0)
                    except (TypeError, ValueError):
                        family_level = 0
                    analog["cards"][card_id] = {
                        "card_id": card_id,
                        **surface,
                        "_family_level": family_level,
                        "_assay_key": _clean(row.get("assay_chembl_id"))
                        or _clean(row.get("group_id")),
                        "_confidence": row.get("confidence_score"),
                    }
    return analogs


def _analog_order(analog: Mapping[str, Any]) -> tuple[Any, ...]:
    selection_rank = analog.get("_selection_rank")
    if selection_rank is not None:
        return (0, float(selection_rank), str(analog.get("analog_id") or ""))
    return (1, -float(analog.get("similarity") or 0.0), str(analog.get("analog_id") or ""))


_NUMERIC_PATTERN = re.compile(r"[-+]?\d")


def _card_order(card: Mapping[str, Any]) -> tuple[Any, ...]:
    selection_rank = card.get("_selection_rank")
    if selection_rank is not None:
        return (0, float(selection_rank), str(card.get("card_id") or ""))
    value = _clean(card.get("reported_value"))
    support = _clean(card.get("support_text"))
    confidence = card.get("_confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = -1.0
    return (
        1,
        -int(bool(support) and bool(value)),
        -int(bool(_NUMERIC_PATTERN.search(value))),
        -int(bool(value)),
        -int(bool(support)),
        -int(bool(_clean(card.get("endpoint")))),
        -sum(
            bool(_clean(card.get(field)))
            for field in ("assay_context", "species", "qualifying_conditions")
        ),
        -confidence_value,
        str(card.get("card_id") or ""),
    )


def _select_cards(cards: Iterable[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Prefer informative cards while spanning physical assays first."""
    ordered = sorted((dict(card) for card in cards), key=_card_order)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_assays: set[str] = set()
    for card in ordered:
        assay_key = _clean(card.get("_assay_key"))
        if assay_key and assay_key in used_assays:
            continue
        selected.append(card)
        selected_ids.add(str(card["card_id"]))
        used_assays.add(assay_key)
        if len(selected) >= limit:
            return selected
    for card in ordered:
        if str(card["card_id"]) in selected_ids:
            continue
        selected.append(card)
        if len(selected) >= limit:
            break
    return selected


def _public_card(card: Mapping[str, Any], *, level: int) -> dict[str, Any]:
    public = {
        key: value
        for key, value in card.items()
        if not str(key).startswith("_") and value not in (None, "", [])
    }
    public["first_seen_level"] = level
    return public


def select_initial_evidence(
    cumulative: Mapping[str, Mapping[str, Any]],
    *,
    level: int = 1,
    molecule_limit: int = INITIAL_MOLECULE_LIMIT,
    card_limit: int = INITIAL_CARD_LIMIT,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    candidates = sorted(cumulative.values(), key=_analog_order)
    for analog in candidates[:molecule_limit]:
        cards = _select_cards((analog.get("cards") or {}).values(), card_limit)
        if not cards:
            continue
        analog_id = str(analog["analog_id"])
        selected[analog_id] = {
            **{key: analog.get(key) for key in (
                "analog_id", "canonical_smiles", "similarity", "similarity_bucket", "molecule_relation",
                "_selection_rank",
            )},
            "first_seen_level": level,
            "cards": {str(card["card_id"]): _public_card(card, level=level) for card in cards},
        }
    return selected, {
        "n_candidate_molecules": len(candidates),
        "n_selected_new_molecules": len(selected),
        "n_selected_augmentation_molecules": 0,
        "n_selected_cards": sum(len(row["cards"]) for row in selected.values()),
    }


def select_progressive_delta(
    previous_cumulative: Mapping[str, Mapping[str, Any]],
    current_cumulative: Mapping[str, Mapping[str, Any]],
    active: Mapping[str, Mapping[str, Any]],
    *,
    level: int,
    new_molecule_limit: int = NEW_MOLECULE_LIMIT,
    augmentation_molecule_limit: int = AUGMENTED_MOLECULE_LIMIT,
    card_limit: int = DELTA_CARD_LIMIT,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Select independent new-molecule and active-molecule augmentation pools."""
    new_candidates = []
    augmentation_candidates = []
    for analog_id, analog in current_cumulative.items():
        previous_ids = set((previous_cumulative.get(analog_id) or {}).get("cards") or {})
        active_ids = set((active.get(analog_id) or {}).get("cards") or {})
        delta_cards = [
            card
            for card_id, card in (analog.get("cards") or {}).items()
            if card_id not in previous_ids and card_id not in active_ids
        ]
        if not delta_cards:
            continue
        candidate = {**analog, "delta_cards": delta_cards}
        if analog_id in active:
            augmentation_candidates.append(candidate)
        else:
            new_candidates.append(candidate)

    selected_new: dict[str, dict[str, Any]] = {}
    for analog in sorted(new_candidates, key=_analog_order)[:new_molecule_limit]:
        cards = _select_cards(analog["delta_cards"], card_limit)
        analog_id = str(analog["analog_id"])
        selected_new[analog_id] = {
            **{key: analog.get(key) for key in (
                "analog_id", "canonical_smiles", "similarity", "similarity_bucket", "molecule_relation"
            )},
            "first_seen_level": level,
            "cards": {str(card["card_id"]): _public_card(card, level=level) for card in cards},
        }

    selected_augmentations: dict[str, dict[str, Any]] = {}
    for analog in sorted(augmentation_candidates, key=_analog_order)[:augmentation_molecule_limit]:
        cards = _select_cards(analog["delta_cards"], card_limit)
        analog_id = str(analog["analog_id"])
        selected_augmentations[analog_id] = {
            "analog_id": analog_id,
            "cards": {str(card["card_id"]): _public_card(card, level=level) for card in cards},
        }

    audit = {
        "n_new_molecule_candidates": len(new_candidates),
        "n_augmentation_molecule_candidates": len(augmentation_candidates),
        "n_selected_new_molecules": len(selected_new),
        "n_selected_augmentation_molecules": len(selected_augmentations),
        "n_selected_cards": sum(len(row["cards"]) for row in selected_new.values())
        + sum(len(row["cards"]) for row in selected_augmentations.values()),
        "unused_new_molecule_slots": max(0, new_molecule_limit - len(selected_new)),
        "unused_augmentation_slots": max(
            0, augmentation_molecule_limit - len(selected_augmentations)
        ),
        "slot_borrowing": False,
    }
    return selected_new, selected_augmentations, audit


def append_evidence(
    active: Mapping[str, Mapping[str, Any]],
    new_molecules: Mapping[str, Mapping[str, Any]],
    augmentations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = json.loads(json.dumps(active, ensure_ascii=False))
    for analog_id, analog in new_molecules.items():
        if analog_id in output:
            raise ValueError(f"new molecule is already active: {analog_id}")
        output[analog_id] = json.loads(json.dumps(analog, ensure_ascii=False))
    for analog_id, delta in augmentations.items():
        if analog_id not in output:
            raise ValueError(f"augmentation molecule is not active: {analog_id}")
        cards = output[analog_id].setdefault("cards", {})
        overlap = set(cards) & set(delta.get("cards") or {})
        if overlap:
            raise ValueError(f"augmentation repeats active cards: {sorted(overlap)}")
        cards.update(json.loads(json.dumps(delta.get("cards") or {}, ensure_ascii=False)))
    return output


def attach_analog_tool_summaries(
    active: dict[str, dict[str, Any]],
    summaries: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    for analog_id, results in summaries.items():
        if analog_id not in active:
            continue
        active[analog_id]["query_analog_tool_summaries"] = [
            {
                "tool_name": result.get("tool_name"),
                "status": result.get("status"),
                "content": result.get("content"),
            }
            for result in results
        ]


def _compact_tool_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep full tool receipts in artifacts while bounding their prompt view."""
    compact = dict(result)
    tool_name = _clean(result.get("tool_name"))
    content = _clean(result.get("content"))
    if tool_name not in {"molecule_properties", "properties_compare"} or not content:
        return compact
    lines = content.splitlines()
    selected = [lines[0]] if lines else []
    for line in lines[1:]:
        name = line.split(":", 1)[0].strip()
        if name in _CORE_PROPERTY_NAMES:
            selected.append(line)
    if tool_name == "molecule_properties" and "functional groups:" in lines:
        start = lines.index("functional groups:")
        selected.extend(lines[start:])
    compact["content"] = "\n".join(selected)
    return compact


def card_ids(active: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        str(card_id)
        for analog in active.values()
        for card_id in (analog.get("cards") or {})
    }


def card_alias_maps(
    active: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable prompt aliases without changing artifact card identifiers.

    Cards are ordered by first-seen level and then by their stable artifact ID.
    Append-only later levels therefore add aliases without renumbering cards that
    were already visible in an earlier prompt.
    """
    ordered = sorted(
        (
            (
                int(card.get("first_seen_level") or 0),
                str(card_id),
            )
            for analog in active.values()
            for card_id, card in (analog.get("cards") or {}).items()
        ),
    )
    card_id_to_alias = {
        card_id: f"C{index:02d}"
        for index, (_, card_id) in enumerate(ordered, start=1)
    }
    alias_to_card_id = {
        alias: card_id for card_id, alias in card_id_to_alias.items()
    }
    return card_id_to_alias, alias_to_card_id


def _prior_card_roles(prior_state: Mapping[str, Any] | None) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    if not prior_state:
        return roles
    for field, role in (
        ("supportive_card_ids", "supportive"),
        ("contradictory_card_ids", "contradictory"),
        ("prediction_basis_card_ids", "prediction_basis"),
    ):
        for card_id in prior_state.get(field) or []:
            roles.setdefault(str(card_id), []).append(role)
    return roles


def _map_card_references(
    rows: Any,
    *,
    card_id_map: Mapping[str, str],
) -> Any:
    """Recursively map values under card-reference fields only."""
    if isinstance(rows, list):
        return [_map_card_references(row, card_id_map=card_id_map) for row in rows]
    if not isinstance(rows, Mapping):
        return rows
    mapped: dict[str, Any] = {}
    for key, value in rows.items():
        if (key.endswith("card_ids") or key.endswith("record_ids")) and isinstance(value, list):
            mapped[key] = [card_id_map[str(card_id)] for card_id in value]
        else:
            mapped[key] = _map_card_references(value, card_id_map=card_id_map)
    return mapped


def render_prior_state(
    prior_state: Mapping[str, Any],
    *,
    card_id_to_alias: Mapping[str, str],
) -> dict[str, Any]:
    """Render prior reasoning once, without duplicating card partitions."""
    omitted = {
        "supportive_card_ids",
        "contradictory_card_ids",
        "prediction_basis_card_ids",
        "not_used_card_ids",
    }
    compact = {key: value for key, value in prior_state.items() if key not in omitted}
    return _map_card_references(compact, card_id_map=card_id_to_alias)


def restore_card_ids(
    content: Mapping[str, Any],
    *,
    alias_to_card_id: Mapping[str, str],
) -> dict[str, Any]:
    """Restore model-facing aliases to stable artifact IDs after validation."""
    return _map_card_references(content, card_id_map=alias_to_card_id)


def render_active_evidence(
    active: Mapping[str, Mapping[str, Any]],
    *,
    current_level: int,
    prior_state: Mapping[str, Any] | None,
    card_id_to_alias: Mapping[str, str],
    card_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contract = dict(card_contract or molecule_card_contract())
    molecule_contract = contract["molecule"]
    card_fields = contract["evidence_card"]["fields"]
    prior_roles = _prior_card_roles(prior_state)
    rendered = []
    for analog in sorted(active.values(), key=_analog_order):
        public_analog = _project_card_fields(analog, molecule_contract["fields"])
        if analog.get("query_analog_tool_summaries"):
            public_analog[molecule_contract["tool_summaries_field"]] = [
                _compact_tool_summary(result)
                for result in analog["query_analog_tool_summaries"]
            ]
        cards = []
        card_order = (
            (lambda row: (int(row.get("_selection_rank") or 0), str(row["card_id"])))
            if contract.get("card_order") == "selection_rank"
            else (lambda row: str(row["card_id"]))
        )
        for card in sorted((analog.get("cards") or {}).values(), key=card_order):
            item = dict(card)
            stable_card_id = str(item["card_id"])
            item["card_id"] = card_id_to_alias[stable_card_id]
            item["new_this_level"] = int(item.get("first_seen_level") or 0) == current_level
            if stable_card_id in prior_roles:
                item["prior_use"] = prior_roles[stable_card_id]
            cards.append(_project_card_fields(item, card_fields))
        public_analog[molecule_contract["evidence_cards_field"]] = cards
        rendered.append(public_analog)
    return rendered


def build_progressive_messages(
    *,
    contract: ProgressiveTaskContract,
    levels: list[Mapping[str, Any]],
    current_level: int,
    query_smiles: str,
    query_molecule_description: str | None = None,
    condition_sentence: str,
    query_prior: Mapping[str, Any] | None,
    query_tool_summary: Mapping[str, Any] | None,
    active: Mapping[str, Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
    protocol_version: str = PROGRESSIVE_PROTOCOL_VERSION,
    card_contract: Mapping[str, Any] | None = None,
    prompt_template: str = "progressive.jinja",
    prompt_version: str = "standard_v1",
    protocol_details: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prompt_text = prompt_assets(prompt_version)["user_shared"]
    molecule_contract = dict(card_contract or molecule_card_contract())
    evidence_card_contract = molecule_contract["evidence_card"]
    card_output_by_source = {
        str(field["source"]): str(field["name"])
        for field in evidence_card_contract["fields"]
    }
    card_id_to_alias, _ = card_alias_maps(active)
    new_ids = sorted(
        card_id_to_alias[str(card["card_id"])]
        for analog in active.values()
        for card in (analog.get("cards") or {}).values()
        if int(card.get("first_seen_level") or 0) == current_level
    )
    is_initial = prior_state is None
    level_plan = [
        {
            "level": int(row["level"]),
            "family": row.get("endpoint_group") or row.get("family_id"),
            "description": row.get("description") or row.get("label") or "",
        }
        for row in levels
    ]
    settings = prompt_assets(prompt_version)["settings"]
    full_flat = str(settings.get("output_contract", "")).startswith(
        "full_flat_progressive."
    )
    schema = {
        contract.prediction_field: f"{contract.positive_prediction} | {contract.negative_prediction}",
        "confidence": "high | moderate | low",
        "revision_action": "initial" if is_initial else "keep | strengthen | weaken | flip",
        "supportive_card_ids": ["C01"],
        "contradictory_card_ids": ["C02"],
        "prediction_basis_card_ids": ["C01"],
        "claims": [{"claim": "concise source-grounded statement", "card_ids": ["C01"]}],
        "new_evidence_assessment": [
            {
                "family": "family name",
                "applicability": prompt_text['applicability'],
                "direction": prompt_text['direction'],
                "decision_effect": prompt_text['decision_effect'],
                "card_ids": ["new card alias"],
            }
        ],
        "evidence_gaps": ["string"],
        "decision_summary": "concise string",
    }
    if settings.get("claim_provenance") in {
        "derived_v1", "derived_v2"
    }:
        for field in (
            "supportive_card_ids",
            "contradictory_card_ids",
            "prediction_basis_card_ids",
        ):
            schema.pop(field)
        schema["claims"] = [
            {
                "claim": "concise source-grounded statement",
                "card_ids": ["exactly one C-number"],
                "evidence_role": "supportive | contradictory",
            }
        ]
    if full_flat:
        schema = {
            "claims": [
                {
                    "claim": "concise evidence-grounded statement",
                    "molecule_ids": ["Molecule N"],
                    "record_ids": ["Cxx"],
                    "evidence_role": "supportive | contradictory",
                }
            ],
            "summary": "concise overall conclusion",
            contract.prediction_field: (
                f"{contract.positive_prediction} | {contract.negative_prediction}"
            ),
        }
        if is_initial:
            schema.update(
                confidence="high | moderate | low",
                evidence_gaps=["important unresolved evidence gap"],
            )
        else:
            schema.update(
                revision_action="keep | strengthen | weaken | flip",
                new_evidence_assessment=[
                    {
                        "molecule_ids": ["Molecule N"],
                        "record_ids": ["Cxx"],
                        "applicability": "high | moderate | low | not_applicable",
                        "direction": "supportive | contradictory | neutral_or_unclear",
                        "decision_effect": "changed | strengthened | weakened | no_change",
                    }
                ],
            )
    visible_evidence = render_active_evidence(
        active,
        current_level=current_level,
        prior_state=prior_state,
        card_id_to_alias=card_id_to_alias,
        card_contract=molecule_contract,
    )
    payload: dict[str, Any] = {
        "protocol": {
            "version": protocol_version,
            "mode": "initial decision" if is_initial else "progressive update",
            "architecture": (
                prompt_text['architecture']
            ),
            "card_accounting": (
                prompt_text['card_accounting']
                + (
                    prompt_text['card_accounting_2'].format(value_1=card_output_by_source['prior_use'], value_2=card_output_by_source['prior_use'])
                    if "prior_use" in card_output_by_source
                    else ""
                )
                + prompt_text['card_accounting_3']
            ),
            "claim_rule": (
                prompt_text['claim_rule']
                + (
                    prompt_text['claim_rule_2']
                    if query_prior
                    else ""
                )
            ),
            "update_rule": (
                prompt_text['update_rule']
            ),
            "identity_rule": (
                prompt_text['identity_rule']
            ),
            "flip_rule": (
                prompt_text['flip_rule']
            ),
            "output_control": (
                prompt_text['output_control']
            ),
        },
        "task_definition": {
            "task": contract.task,
            "endpoint": contract.endpoint_name,
            "label_scope": contract.label_scope,
            "prediction_values": {
                contract.positive_prediction: "positive class (label 1)",
                contract.negative_prediction: "negative class (label 0)",
            },
            "instructions": list(contract.task_instructions),
        },
        "level_context": {
            "current_level": current_level,
            "current_family": next(
                row for row in level_plan if int(row["level"]) == current_level
            ),
            "full_level_plan": level_plan,
            "new_card_ids": new_ids,
        },
        "query": {"canonical_smiles": query_smiles},
        **({"query_prior": dict(query_prior)} if query_prior else {}),
        "active_evidence": visible_evidence,
        "required_json_schema": schema,
    }
    if full_flat:
        molecule_number = 0
        if prior_state is not None:
            molecules = []
            for molecule in visible_evidence:
                cards = [dict(card) for card in molecule.get("evidence_cards") or []]
                if not cards:
                    continue
                molecule_number += 1
                public = {
                    key: value
                    for key, value in molecule.items()
                    if key not in {"analog_id", "evidence_cards", "first_seen_level"}
                }
                public.update(number=molecule_number, evidence_cards=cards)
                molecules.append(public)
            payload["flat_molecules"] = molecules
        else:
            sections = []
            task_levels = prompt_assets(prompt_version)["levels"][contract.task]
            for level_name, definition in task_levels.items():
                level = int(level_name[1:])
                if level > current_level:
                    continue
                molecules = []
                for molecule in visible_evidence:
                    cards = [
                        dict(card)
                        for card in molecule.get("evidence_cards") or []
                        if int(card.get("first_seen_level") or 0) == level
                    ]
                    if not cards:
                        continue
                    molecule_number += 1
                    public = {
                        key: value
                        for key, value in molecule.items()
                        if key not in {"analog_id", "evidence_cards", "first_seen_level"}
                    }
                    public.update(number=molecule_number, evidence_cards=cards)
                    molecules.append(public)
                if molecules:
                    sections.append({
                        "level": level_name,
                        "family": definition["evidence_family"],
                        "description": definition["description"],
                        "molecules": molecules,
                    })
            payload["level_sections"] = sections
    if condition_sentence:
        payload["query"]["external_condition"] = condition_sentence
    if query_molecule_description:
        payload["query"]["molecule_description"] = query_molecule_description
    if query_tool_summary:
        payload["query"]["molecule_property_tool_summary"] = _compact_tool_summary(query_tool_summary)
    transfer_field = card_output_by_source.get("transfer_likelihood")
    if transfer_field and any(
        card.get("transfer_likelihood") is not None
        for analog in active.values()
        for card in (analog.get("cards") or {}).values()
    ):
        payload["protocol"]["transfer_likelihood_rule"] = (
            prompt_text['progressive_messages'].format(transfer_field=transfer_field)
        )
    if prior_state is not None:
        payload["prior_state"] = (
            {
                key: prior_state.get(key)
                for key in (
                    "claims", "summary", contract.prediction_field,
                    "confidence", "evidence_gaps",
                )
            }
            if full_flat
            else render_prior_state(
                prior_state,
                card_id_to_alias=card_id_to_alias,
            )
        )
    if protocol_details:
        payload["protocol"].update(dict(protocol_details))
    return render_progressive_messages(
        system_role=contract.system_role,
        payload=payload,
        template_name=prompt_template,
        prompt_version=prompt_version,
    )


def progressive_state_errors(
    content: Mapping[str, Any],
    *,
    contract: ProgressiveTaskContract,
    visible_card_ids: set[str],
    new_card_ids: set[str],
    prior_state: Mapping[str, Any] | None,
    max_claims: int | None = 8,
) -> list[str]:
    errors: list[str] = []
    partitions: dict[str, list[str]] = {}
    for field in ("supportive_card_ids", "contradictory_card_ids"):
        raw = content.get(field)
        if not isinstance(raw, list):
            errors.append(f"{field} must be an array")
            partitions[field] = []
        else:
            partitions[field] = [str(value) for value in raw]
            if len(partitions[field]) != len(set(partitions[field])):
                errors.append(f"{field} contains duplicates")
    union = set().union(*(set(values) for values in partitions.values()))
    total = sum(len(values) for values in partitions.values())
    if not union <= visible_card_ids:
        errors.append("supportive/contradictory cards must be visible")
    if total != len(union):
        errors.append("supportive/contradictory card lists must be disjoint")

    basis_raw = content.get("prediction_basis_card_ids")
    basis = {str(value) for value in basis_raw} if isinstance(basis_raw, list) else set()
    if not isinstance(basis_raw, list):
        errors.append("prediction_basis_card_ids must be an array")
    if not basis <= visible_card_ids:
        errors.append("prediction_basis_card_ids contains unknown card IDs")
    if not basis <= union:
        errors.append("prediction basis cards must be supportive or contradictory")

    prediction = str(content.get(contract.prediction_field) or "")
    if prediction not in contract.prediction_values:
        errors.append(f"invalid {contract.prediction_field}")
    action = str(content.get("revision_action") or "")
    if prior_state is None:
        if action != "initial":
            errors.append("initial level requires revision_action=initial")
    else:
        prior_prediction = str(prior_state.get(contract.prediction_field) or "")
        if action not in {"keep", "strengthen", "weaken", "flip"}:
            errors.append("update revision_action must be keep, strengthen, weaken, or flip")
        if action == "flip":
            if prediction == prior_prediction:
                errors.append("revision_action=flip requires a changed prediction")
            if new_card_ids and not basis.intersection(new_card_ids):
                errors.append("a flip must cite at least one newly added card in prediction basis")
        elif prediction and prior_prediction and prediction != prior_prediction:
            errors.append("changed prediction requires revision_action=flip")

    claims = content.get("claims")
    if not isinstance(claims, list):
        errors.append("claims must be an array")
    else:
        if max_claims is not None and len(claims) > max_claims:
            errors.append(f"claims must contain at most {max_claims} items")
        for claim in claims:
            refs = claim.get("card_ids") if isinstance(claim, Mapping) else None
            if (
                not isinstance(refs, list)
                or not set(map(str, refs)) <= visible_card_ids
            ):
                errors.append("every claim must reference only visible card IDs")
                break

    assessments = content.get("new_evidence_assessment")
    assessed: set[str] = set()
    if not isinstance(assessments, list):
        errors.append("new_evidence_assessment must be an array")
    else:
        for assessment in assessments:
            refs = assessment.get("card_ids") if isinstance(assessment, Mapping) else None
            if not isinstance(refs, list):
                errors.append("every new evidence assessment needs card_ids")
                continue
            assessed.update(map(str, refs))
            if str(assessment.get("applicability") or "") not in {
                "high", "moderate", "low", "not_applicable"
            }:
                errors.append("invalid new evidence applicability")
            if str(assessment.get("direction") or "") not in {
                "supportive", "contradictory", "neutral_or_unclear"
            }:
                errors.append("invalid new evidence direction")
            if str(assessment.get("decision_effect") or "") not in {
                "changed", "strengthened", "weakened", "no_change"
            }:
                errors.append("invalid new evidence decision effect")
        if not assessed <= new_card_ids:
            errors.append("new evidence assessments may cite only newly added card IDs")
    gaps = content.get("evidence_gaps")
    if isinstance(gaps, list) and len(gaps) > 6:
        errors.append("evidence_gaps must contain at most 6 items")
    return errors


def state_from_content(
    content: Mapping[str, Any],
    *,
    contract: ProgressiveTaskContract,
    level: int,
    visible_card_ids: set[str],
) -> dict[str, Any]:
    fields = (
        contract.prediction_field,
        "confidence",
        "revision_action",
        "supportive_card_ids",
        "contradictory_card_ids",
        "prediction_basis_card_ids",
        "claims",
        "new_evidence_assessment",
        "evidence_gaps",
        "decision_summary",
    )
    state = {"level": level, **{field: content.get(field) for field in fields}}
    used = {
        str(card_id)
        for field in ("supportive_card_ids", "contradictory_card_ids")
        for card_id in content.get(field) or []
    }
    state["not_used_card_ids"] = sorted(visible_card_ids - used)
    return state
