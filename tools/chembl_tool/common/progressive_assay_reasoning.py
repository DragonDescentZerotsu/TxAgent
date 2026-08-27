"""Append-only assay-card selection and progressive reasoning contracts.

This module deliberately knows nothing about benchmark file paths or model
endpoints.  It turns cumulative assay retrievals into a stable molecule/card
surface, selects the bounded evidence delta, renders the model prompt, and
validates the progressive state returned by the reasoning model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


PROGRESSIVE_PROTOCOL_VERSION = "conditioned_assay_progressive_visible.v8"
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


def _card_surface(example: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    scope = row.get("evidence_scope") or {}
    assay_context = _clean(example.get("assay_context"))
    if not assay_context:
        contexts = scope.get("assay_context") or []
        assay_context = _clean(contexts[0]) if contexts else _clean(row.get("target_pref_name"))
    species = _clean(example.get("species_context"))
    if not species:
        species_values = scope.get("species_context") or []
        species = _clean(species_values[0]) if species_values else _clean(row.get("organism"))
    conditions = _clean(example.get("qualifying_conditions"))
    if not conditions:
        condition_values = scope.get("qualifying_conditions") or []
        conditions = _clean(condition_values[0]) if condition_values else ""
    return {
        "evidence_family": _clean(example.get("evidence_family")),
        "assay_context": assay_context,
        "endpoint": _clean(example.get("endpoint_type")) or _clean(row.get("standard_type")),
        "reported_value": _clean(example.get("reported_value")) or _clean(row.get("standard_value")),
        "reported_unit": _clean(example.get("reported_units")) or _clean(row.get("standard_units")),
        "species": species,
        "qualifying_conditions": conditions,
        "support_text": _clean(example.get("support_text")) or _clean(row.get("assay_description")),
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
    return (-float(analog.get("similarity") or 0.0), str(analog.get("analog_id") or ""))


_NUMERIC_PATTERN = re.compile(r"[-+]?\d")


def _card_order(card: Mapping[str, Any]) -> tuple[Any, ...]:
    value = _clean(card.get("reported_value"))
    support = _clean(card.get("support_text"))
    confidence = card.get("_confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = -1.0
    return (
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
    return {
        key: value
        for key, value in {
            "card_id": card.get("card_id"),
            "evidence_family": card.get("evidence_family"),
            "assay_context": card.get("assay_context"),
            "endpoint": card.get("endpoint"),
            "reported_value": card.get("reported_value"),
            "reported_unit": card.get("reported_unit"),
            "species": card.get("species"),
            "qualifying_conditions": card.get("qualifying_conditions"),
            "support_text": card.get("support_text"),
            "first_seen_level": level,
        }.items()
        if value not in (None, "")
    }


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
                "analog_id", "canonical_smiles", "similarity", "similarity_bucket", "molecule_relation"
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
        if key.endswith("card_ids") and isinstance(value, list):
            mapped[key] = [card_id_map[str(card_id)] for card_id in value]
        else:
            mapped[key] = _map_card_references(value, card_id_map=card_id_map)
    return mapped


def _render_prior_state(
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
) -> list[dict[str, Any]]:
    prior_roles = _prior_card_roles(prior_state)
    rendered = []
    for analog in sorted(active.values(), key=_analog_order):
        public_analog = {
            key: analog.get(key)
            for key in (
                "analog_id",
                "canonical_smiles",
                "similarity",
                "similarity_bucket",
                "molecule_relation",
                "first_seen_level",
            )
            if analog.get(key) not in (None, "", [])
        }
        if analog.get("query_analog_tool_summaries"):
            public_analog["query_analog_tool_summaries"] = [
                _compact_tool_summary(result)
                for result in analog["query_analog_tool_summaries"]
            ]
        cards = []
        for card in sorted((analog.get("cards") or {}).values(), key=lambda row: str(row["card_id"])):
            item = dict(card)
            stable_card_id = str(item["card_id"])
            item["card_id"] = card_id_to_alias[stable_card_id]
            item["new_this_level"] = int(item.get("first_seen_level") or 0) == current_level
            if stable_card_id in prior_roles:
                item["prior_use"] = prior_roles[stable_card_id]
            cards.append(item)
        public_analog["evidence_cards"] = cards
        rendered.append(public_analog)
    return rendered


def build_progressive_messages(
    *,
    contract: ProgressiveTaskContract,
    levels: list[Mapping[str, Any]],
    current_level: int,
    query_smiles: str,
    condition_sentence: str,
    query_prior: Mapping[str, Any],
    query_tool_summary: Mapping[str, Any] | None,
    active: Mapping[str, Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
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
                "applicability": "high | moderate | low | not_applicable",
                "direction": "supportive | contradictory | neutral_or_unclear",
                "decision_effect": "changed | strengthened | weakened | no_change",
                "card_ids": ["new card alias"],
            }
        ],
        "evidence_gaps": ["string"],
        "decision_summary": "concise string",
    }
    payload: dict[str, Any] = {
        "protocol": {
            "version": PROGRESSIVE_PROTOCOL_VERSION,
            "mode": "initial decision" if is_initial else "progressive update",
            "architecture": (
                "Evidence is append-only. Every selected raw card accumulated through the current level is shown. "
                "Later levels add biologically more indirect families. All older cards remain visible so that a prior "
                "decision can be corrected, but on an update start with the new cards and their effect on the prior "
                "decision. Revisit only relevant older cards when new evidence conflicts with the prior reasoning or "
                "with those cards; do not re-audit every older card by default."
            ),
            "card_accounting": (
                "Cards use short aliases that the workflow maps back to stable artifact IDs. On an older card, prior_use "
                "lists the roles it had in the previous decision; an absent prior_use means that it was not used. "
                "List only cards that materially support or contradict the current decision. Unlisted visible cards are "
                "deterministically recorded as not_used by the workflow. Repeated records are not independent votes."
            ),
            "claim_rule": (
                "Every evidence-card claim must cite its card IDs. A claim based only on query_prior may use an "
                "empty card_ids list, but must say explicitly that it is a query-property prior rather than experimental evidence."
            ),
            "update_rule": (
                "Judge endpoint-to-task relevance, direction, species/condition compatibility, formulation or route "
                "compatibility, and whether structural differences preserve the mechanism. Indirect evidence may "
                "support, contradict, or leave the earlier prediction unchanged."
            ),
            "identity_rule": (
                "Do not identify the query by name even if its structure is recognizable."
            ),
            "flip_rule": (
                "A flip is allowed only when new evidence is strong enough to overturn the prior decision; if you flip, "
                "prediction_basis_card_ids must include at least one card from new_card_ids."
            ),
            "output_control": (
                "Use the minimum sufficient number of claims, evidence gaps, assessments, and card citations. Empty "
                "lists are valid; do not fill arrays merely to appear complete or to approach a target count. Cite only "
                "cards that materially affect the prediction, omit irrelevant cards, and do not repeat card text."
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
            "current_family": level_plan[current_level - 1],
            "full_level_plan": level_plan,
            "new_card_ids": new_ids,
        },
        "query": {"canonical_smiles": query_smiles},
        "query_prior": dict(query_prior),
        "active_evidence": render_active_evidence(
            active,
            current_level=current_level,
            prior_state=prior_state,
            card_id_to_alias=card_id_to_alias,
        ),
        "required_json_schema": schema,
    }
    if condition_sentence:
        payload["query"]["external_condition"] = condition_sentence
    if query_tool_summary:
        payload["query"]["molecule_property_tool_summary"] = _compact_tool_summary(query_tool_summary)
    if prior_state is not None:
        payload["prior_state"] = _render_prior_state(
            prior_state,
            card_id_to_alias=card_id_to_alias,
        )
    return [
        {
            "role": "system",
            "content": (
                contract.system_role
                + " You are operating inside a progressive molecular-evidence experiment. "
                "Use general medicinal-chemistry knowledge to interpret the supplied structures and evidence. "
                "Ground every compound-specific empirical claim and the final prediction in the supplied prior state, "
                "tool summaries, or cited evidence cards. "
                "Return exactly one valid JSON object and do not reveal hidden chain-of-thought."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def progressive_state_errors(
    content: Mapping[str, Any],
    *,
    contract: ProgressiveTaskContract,
    visible_card_ids: set[str],
    new_card_ids: set[str],
    prior_state: Mapping[str, Any] | None,
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
        if len(claims) > 8:
            errors.append("claims must contain at most 8 items")
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
