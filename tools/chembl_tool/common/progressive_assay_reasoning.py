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
PROGRESSIVE_PROMPT_PROFILE = "progressive_evidence_revision.v5"
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
    evidence_grounding_rules: tuple[tuple[str, str], ...] = ()

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


def shortlist_progressive_retrievals(retrievals, family_levels_by_molecule):
    """Materialize cards only for molecules the unchanged budgets can select.

    A card's family is part of its identity, so newly unlocked cards at level L
    belong to family L. Rank both pools before expanding any record text, then
    retain the union at every level (including its earlier, unselected cards).
    The latter is essential to preserve the original delta-card semantics.
    """
    active_ids = set()
    for level, retrieval in sorted(retrievals.items()):
        candidates = {}
        for group in retrieval.get("groups") or []:
            for neighbor in group.get("neighbors") or []:
                aid = stable_analog_id(neighbor)
                row = candidates.setdefault(aid, {
                    "analog_id": aid, "similarity": neighbor["similarity"], "unlocked": False,
                })
                row["unlocked"] |= level in family_levels_by_molecule[neighbor["molecule_chembl_id"]]
        if level == 1:
            active_ids.update(r["analog_id"] for r in sorted(candidates.values(), key=_analog_order)[:INITIAL_MOLECULE_LIMIT])
        else:
            new = [r for aid, r in candidates.items() if aid not in active_ids and r["unlocked"]]
            active_ids.update(r["analog_id"] for r in sorted(new, key=_analog_order)[:NEW_MOLECULE_LIMIT])
    return {
        level: {**retrieval, "groups": [
            {**group, "neighbors": [n for n in group.get("neighbors") or [] if stable_analog_id(n) in active_ids]}
            for group in retrieval.get("groups") or []
        ]}
        for level, retrieval in retrievals.items()
    }


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


def _endpoint_diversity_key(card: Mapping[str, Any]) -> str:
    """Use the reported endpoint type, never the result direction or query label."""
    endpoint = _clean(card.get("endpoint")).lower()
    match = re.search(
        r"(?:^|\|)\s*(assay_family|endpoint_category|assay_domain|phenotype_domain)\s*=\s*([^|]+)",
        endpoint,
    )
    return " ".join(match.group(2).split()) if match else endpoint


def _select_cards(cards: Iterable[Mapping[str, Any]], limit: int, *,
                  prefer_distinct_endpoints: bool = False) -> list[dict[str, Any]]:
    """Prefer informative cards while spanning physical assays first."""
    ordered = sorted((dict(card) for card in cards), key=_card_order)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_assays: set[str] = set()
    if prefer_distinct_endpoints:
        used_endpoints: set[str] = set()
        for card in ordered:
            endpoint_key = _endpoint_diversity_key(card)
            if not endpoint_key or endpoint_key in used_endpoints:
                continue
            selected.append(card)
            selected_ids.add(str(card["card_id"]))
            used_endpoints.add(endpoint_key)
            used_assays.add(_clean(card.get("_assay_key")))
            if len(selected) >= limit:
                return selected
    for card in ordered:
        if str(card["card_id"]) in selected_ids:
            continue
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
    query_condition: str = "",
    card_condition_groups: Mapping[str, list[str]] | None = None,
    condition_priority_scope: str = "molecules_and_cards",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if condition_priority_scope not in {"molecules_and_cards", "cards_only", "cards_exact_only"}:
        raise ValueError("unknown initial condition priority scope")
    # Condition metadata is selection-only; neither votes nor derived labels enter here.
    condition_priority = bool(card_condition_groups is not None and query_condition
                              and query_condition != "no_reported_external_condition")

    def condition_rank(card: Mapping[str, Any]) -> int:
        groups = (card_condition_groups or {}).get(str(card["card_id"]), [])
        if query_condition in groups:
            return 0
        if condition_priority_scope == "cards_exact_only":
            return 1
        if not groups or "no_reported_external_condition" in groups:
            return 1
        return 2

    selected: dict[str, dict[str, Any]] = {}
    candidates = sorted(cumulative.values(), key=_analog_order)
    if condition_priority and condition_priority_scope == "molecules_and_cards":
        # Stable sorting preserves the original Morgan order within each bucket.
        candidates.sort(key=lambda a: min(
            (condition_rank(c) for c in (a.get("cards") or {}).values()), default=1))
    for analog in candidates[:molecule_limit]:
        available = list((analog.get("cards") or {}).values())
        if condition_priority:
            cards = []
            for rank in range(3):
                if len(cards) == card_limit:
                    break
                cards.extend(_select_cards(
                    (c for c in available if condition_rank(c) == rank), card_limit - len(cards)))
        else:
            cards = _select_cards(available, card_limit)
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
    prefer_distinct_endpoints: bool = False,
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
        cards = _select_cards(analog["delta_cards"], card_limit,
                              prefer_distinct_endpoints=prefer_distinct_endpoints)
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
        cards = _select_cards(analog["delta_cards"], card_limit,
                              prefer_distinct_endpoints=prefer_distinct_endpoints)
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


def tool_functional_group_tree(result: Mapping[str, Any], *, reference: bool = False) -> str:
    """Read the tree from tool output.text, never from internal atom mappings."""
    if result.get("status") == "error":
        return "Unavailable; do not infer absence of functional groups."
    section = "reference_functional_group_tree" if reference else "functional_group_tree"
    _, separator, tree = str(result.get("content") or "").partition(f"[{section}]\n")
    return tree.split("\nWarnings:", 1)[0].strip() if separator else ""


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
    # Historical receipts remain renderable for paired replay; new tool results
    # carry a separate tree and have no flat functional-group section.
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
    family_labels: Mapping[str, str] | None = None,
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
            for result in analog["query_analog_tool_summaries"]:
                if result.get("tool_name") == "properties_compare":
                    tree = tool_functional_group_tree(result, reference=True)
                    if tree:
                        public_analog["functional_group_tree"] = tree
        cards = []
        for card in sorted((analog.get("cards") or {}).values(), key=lambda row: str(row["card_id"])):
            item = dict(card)
            if family_labels and item.get("evidence_family") in family_labels:
                item["evidence_family"] = family_labels[item["evidence_family"]]
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
    independent: bool = False,
    omit_query_prior: bool = False,
    evidence_grounding: bool = False,
    excluded_query_name: str = "",
) -> list[dict[str, Any]]:
    if independent and prior_state is not None:
        raise ValueError("independent full-flat reasoning cannot consume prior state")
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
            "family": row.get("family_label") or row.get("endpoint_group") or row.get("family_id"),
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
        "decision_summary": "Explain the decision and why the strongest relevant counterevidence does not prevail, if any.",
    }
    payload: dict[str, Any] = {
        "protocol": {
            "version": PROGRESSIVE_PROTOCOL_VERSION,
            "mode": "initial decision" if is_initial else "progressive update",
            "architecture": (
                "Evidence is append-only. Every selected raw card accumulated through the current level is shown. "
                "Later levels add biologically more indirect families. Start with new cards and revisit relevant older "
                "cards when evidence conflicts or a prior factual, attribution, or inference error is apparent."
            ),
            "card_accounting": (
                "Card aliases are stable. prior_use records previous citation roles, not evidence quality. "
                "Supportive/contradictory lists refer to the current prediction; unlisted cards are recorded as not_used. "
                "Repeated records are not independent votes."
            ),
            "claim_rule": (
                "Query priors and prior states are revisable judgments, not sources of experimental facts. "
                "Ground compound-specific empirical claims in the supplied evidence cards, including inherited claims. "
                "For each decisive claim, name the tested subject and relevant conditions, explain transfer to the "
                "query and its effect on the task label, and cite the cards. Structure/computed-property inferences may have empty card_ids "
                "but must be labeled as inferences, not observed outcomes."
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
                "A flip may follow new evidence or correction of a prior factual, attribution, or inference error. "
                "Explain the new evidence or specific correction in decision_summary and cite the relevant visible "
                "basis cards; correction of an earlier error does not require a new card."
            ),
            "output_control": (
                "Choose one prediction from the schema. Express uncertainty through confidence, evidence_gaps, "
                "and decision_summary. Use only decisive claims and citations; do not repeat card text or assess "
                "every neighbor. Empty lists are valid. Maximum 8 claims and 6 evidence gaps; these are limits, not targets."
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
            # Native query-only branches share these contracts; omit their
            # generic binary-choice instruction only in this shared prompt.
            "instructions": [s for s in contract.task_instructions if not s.startswith("Choose ")],
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
            family_labels={str(row.get("endpoint_group") or row.get("family_id")): str(row["family_label"])
                           for row in levels if row.get("family_label")},
        ),
        "required_json_schema": schema,
    }
    if condition_sentence:
        payload["query"]["external_condition"] = condition_sentence
    if excluded_query_name:
        payload["query"]["identity_exclusion"] = f"The query molecule is not {excluded_query_name}."
    if query_tool_summary:
        payload["query"]["molecule_property_tool_summary"] = _compact_tool_summary(query_tool_summary)
        tree = tool_functional_group_tree(query_tool_summary)
        if tree:
            payload["query"]["functional_group_tree"] = tree
    if "functional_group_tree" in payload["query"] or any(
        "functional_group_tree" in analog for analog in payload["active_evidence"]
    ):
        payload["protocol"]["functional_group_tree_rule"] = (
            "Trees show nested substructure matches, not chemical connectivity. "
            "Counts apply within each branch; do not add parent and child counts."
        )
    if prior_state is not None:
        payload["prior_state"] = _render_prior_state(
            prior_state,
            card_id_to_alias=card_id_to_alias,
        )
    if independent:
        payload["protocol"].update(
            version="conditioned_assay_matched_full_flat.v1",
            mode="independent cumulative full-flat decision",
            architecture=(
                "Judge all supplied cumulative evidence together from scratch. "
                "No previous evidence-based decision is supplied or presumed."
            ),
            card_accounting=(
                "Card aliases map to stable artifact IDs. Cite only materially relevant cards. "
                "Repeated records are not independent votes. All supplied cards are available "
                "to new_evidence_assessment; decision_effect refers only to the query-property prior."
            ),
        )
        payload["protocol"]["evidence_rule"] = (
            "Judge endpoint relevance, direction, species/condition, formulation/route, "
            "and whether structural differences preserve the mechanism."
        )
        for key in ("update_rule", "flip_rule"):
            del payload["protocol"][key]
        del payload["level_context"]["new_card_ids"]
        schema["new_evidence_assessment"][0]["card_ids"] = ["evidence card alias"]
        for analog in payload["active_evidence"]:
            analog.pop("first_seen_level", None)
            for card in analog["evidence_cards"]:
                card.pop("first_seen_level", None)
                card.pop("new_this_level", None)
    if omit_query_prior and active:
        del payload["query_prior"]
        if independent:
            payload["protocol"]["card_accounting"] = (
                "Card aliases map to stable artifact IDs. Cite only materially relevant cards. "
                "Repeated records are not independent votes. All supplied cards are available "
                "to new_evidence_assessment; decision_effect describes their effect on this decision."
            )
    if evidence_grounding:
        payload["protocol"]["identity_rule"] = (
            "Retrieved neighbors are distinct from the query. Attribute each observation to the molecule "
            "actually studied; never present an analog's trial, adverse event, or regulatory outcome as "
            "the query's own result. For passages mentioning multiple molecules, distinguish their roles "
            "and outcomes. Explain the structural and contextual basis for transferring analog evidence. "
            "Do not identify the query by name merely because its structure is recognizable."
        )
        payload["protocol"]["endpoint_decision_rule"] = (
            "Predict the defined endpoint label under the query's reported condition. A structural alert, "
            "hypothetical mechanism, or reason for further safety investigation does not by itself establish "
            "a positive endpoint outcome. Separate measured outcomes from mechanisms and predictions."
        )
        payload["protocol"]["evidence_balance_rule"] = (
            "Assess positive and negative observations using the same standards of endpoint relevance, "
            "study scope and analog transferability. A scoped negative observation is evidence within that "
            "scope; it need not prove universal safety. An untested alternative mechanism is not observed "
            "positive evidence. Preserve conflicting findings and do not equate missing evidence with a negative result."
        )
        payload["protocol"].update(dict(contract.evidence_grounding_rules))
    return [
        {
            "role": "system",
            "content": (
                contract.system_role
                + (" You are making an independent cumulative full-flat decision. " if independent
                   else " You are operating inside a progressive molecular-evidence experiment. ")
                + "Use general medicinal-chemistry knowledge to interpret the supplied structures and evidence. "
                + "Follow the evidence and inference rules below. "
                "Return exactly one valid JSON object matching the schema and do not reveal hidden chain-of-thought."
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
            # Old evidence or a corrected property inference can justify a flip.
            # Validate the explanation's presence, not its scientific truth.
            summary = content.get("decision_summary")
            if not isinstance(summary, str) or not summary.strip():
                errors.append("a flip requires a decision_summary explaining the evidence or correction")
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
