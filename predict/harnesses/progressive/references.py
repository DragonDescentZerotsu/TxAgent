"""Keep progressive prompt identifiers and reasoning diagnostics aligned."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from predict.harnesses.reasoning_references import (
    CARD_RE,
    GROUP_RE,
    MOLECULE_RE,
    match_references,
)


SCHEMA_VERSION = "progressive_reasoning_references.v1"
LAYOUTS = {
    "l1",
    "semantic_l2",
    "weighted_l2",
    "weighted_l2_legacy",
    "flat_global",
    "flat_level_grouped",
}
HEADING_RE = re.compile(
    r"^(#{2,3})\s+(Molecule|Evidence group|Semantic bucket)\s+[1-9][0-9]*$"
)
REFERENCE_LINE_RE = re.compile(
    r"^(?:#{2,3}\s+(?:Molecule|Evidence group|Semantic bucket)\s+[1-9][0-9]*"
    r"|Record\s+(?:C[0-9]{2,}|[1-9][0-9]*-[1-9][0-9]*))$"
)


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract != {
        "schema_version": SCHEMA_VERSION,
        "layout": contract.get("layout"),
        "required_mentions": False,
        "matching": "case_insensitive_exact_visible_id",
    } or contract.get("layout") not in LAYOUTS - {"weighted_l2_legacy"}:
        raise ValueError("invalid progressive reasoning-reference contract")


def normalize_headings(text: str) -> str:
    """Canonicalize evidence headings without constraining model reasoning."""
    molecules = groups = 0
    output = []
    for line in text.splitlines():
        match = HEADING_RE.fullmatch(line)
        if match and match.group(2) == "Molecule":
            molecules += 1
            line = f"{match.group(1)} Molecule {molecules}"
        elif match:
            groups += 1
            line = f"{match.group(1)} Evidence group {groups}"
        output.append(line)
    return "\n".join(output)


def build_reference_index(
    active: Mapping[str, Mapping[str, Any]],
    *,
    card_id_to_alias: Mapping[str, str],
    current_level: int,
    layout: str,
) -> list[dict[str, Any]]:
    """Return the exact molecule, group, and record identifiers in one prompt."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown reasoning-reference layout: {layout}")
    ordered = sorted(
        active.items(),
        key=lambda item: (int(item[1].get("_selection_rank") or 0), item[0]),
    )
    rows: list[dict[str, Any]] = []
    molecule_number = group_number = 0
    legacy_l2_molecule_number = 0

    def cards_at(analog: Mapping[str, Any], level: int) -> list[tuple[str, Mapping[str, Any]]]:
        return sorted(
            (
                (str(card_id), card)
                for card_id, card in (analog.get("cards") or {}).items()
                if int(card.get("first_seen_level") or 0) == level
            ),
            key=lambda item: (int(item[1].get("first_seen_level") or 0), item[0]),
        )

    def add_unit(
        *, kind: str, visible_id: str, stable_id: str, analog_id: str,
        level: int, heading: str, containing_unit_id: str = "",
    ) -> None:
        rows.append({
            "unit_kind": kind,
            "visible_id": visible_id,
            "stable_id": stable_id,
            "source_analog_id": analog_id,
            "containing_unit_id": containing_unit_id,
            "first_visible_level": level,
            "is_new": level == current_level,
            "prompt_heading": heading,
        })

    def add_records(
        cards: list[tuple[str, Mapping[str, Any]]], *, containing_unit_id: str,
    ) -> None:
        for card_id, card in cards:
            alias = card_id_to_alias[card_id]
            rows.append({
                "unit_kind": "record",
                "visible_id": alias,
                "stable_id": card_id,
                "source_analog_id": "",
                "containing_unit_id": containing_unit_id,
                "first_visible_level": int(card.get("first_seen_level") or current_level),
                "is_new": int(card.get("first_seen_level") or 0) == current_level,
                "prompt_heading": f"Record {alias}",
            })

    if layout in {"flat_global", "flat_level_grouped"}:
        if layout == "flat_global":
            level_groups = [(0, ordered)]
        else:
            visible_levels = sorted({
                int(card.get("first_seen_level") or 0)
                for _, analog in ordered
                for card in (analog.get("cards") or {}).values()
            })
            level_groups = [(level, ordered) for level in visible_levels]
        for level, analogs in level_groups:
            for analog_id, analog in analogs:
                cards = (
                    sorted(
                        (
                            (str(card_id), card)
                            for card_id, card in (analog.get("cards") or {}).items()
                        ),
                        key=lambda item: (
                            int(item[1].get("first_seen_level") or 0), item[0]
                        ),
                    )
                    if level == 0
                    else cards_at(analog, level)
                )
                if not cards:
                    continue
                molecule_number += 1
                first_level = min(
                    int(card.get("first_seen_level") or current_level)
                    for _, card in cards
                )
                stable_id = (
                    str(analog_id)
                    if level == 0 else f"{analog_id}@L{level}"
                )
                add_unit(
                    kind=(
                        "context_conditioned_molecule"
                        if first_level == 1 and analog.get("group_kind") == "l1_context"
                        else "molecule"
                    ),
                    visible_id=f"Molecule {molecule_number}",
                    stable_id=stable_id,
                    analog_id=str(analog_id),
                    level=first_level,
                    heading=f"## Molecule {molecule_number}",
                )
                add_records(cards, containing_unit_id=stable_id)
        if current_level > 1:
            for row in rows:
                row["is_new"] = int(row["first_visible_level"]) > 1
        return rows

    # L1-layout prompts can also render one isolated later level as the initial evidence.
    l1_level = 1
    if layout == "l1" and not any(cards_at(analog, 1) for _, analog in ordered):
        l1_level = current_level
    for analog_id, analog in ordered:
        cards = cards_at(analog, l1_level)
        if not cards:
            continue
        molecule_number += 1
        stable_id = f"{analog_id}@L{l1_level}"
        add_unit(
            kind=(
                "context_conditioned_molecule"
                if l1_level == 1 and analog.get("group_kind") == "l1_context"
                else "molecule"
            ),
            visible_id=f"Molecule {molecule_number}",
            stable_id=stable_id,
            analog_id=analog_id,
            level=l1_level,
            heading=f"## Molecule {molecule_number}",
        )
        add_records(cards, containing_unit_id=stable_id)

    if current_level == 1 or layout == "l1":
        return rows

    for analog_id, analog in ordered:
        cards = cards_at(analog, 2)
        if not cards:
            continue
        stable_id = f"{analog_id}@L2"
        if layout == "semantic_l2":
            group_number += 1
            add_unit(
                kind="evidence_group",
                visible_id=f"Evidence group {group_number}",
                stable_id=stable_id,
                analog_id=analog_id,
                level=2,
                heading=f"## Evidence group {group_number}",
            )
            add_records(cards, containing_unit_id=stable_id)
            continue
        if analog.get("group_kind") != "semantic_bucket":
            if layout == "weighted_l2":
                molecule_number += 1
            else:
                legacy_l2_molecule_number += 1
                molecule_number = legacy_l2_molecule_number
            add_unit(
                kind="molecule",
                visible_id=f"Molecule {molecule_number}",
                stable_id=stable_id,
                analog_id=analog_id,
                level=2,
                heading=f"## Molecule {molecule_number}",
            )
            add_records(cards, containing_unit_id=stable_id)
            continue
        group_number += 1
        group_label = (
            f"Evidence group {group_number}"
            if layout == "weighted_l2" else f"Semantic bucket {group_number}"
        )
        add_unit(
            kind="evidence_group",
            visible_id=group_label,
            stable_id=stable_id,
            analog_id=analog_id,
            level=2,
            heading=f"## {group_label}",
        )
        for smiles in sorted({str(card.get("reference_smiles") or "") for _, card in cards}):
            if layout == "weighted_l2":
                molecule_number += 1
            else:
                legacy_l2_molecule_number += 1
                molecule_number = legacy_l2_molecule_number
            parent_id = f"{stable_id}:parent:{smiles}"
            add_unit(
                kind="molecule",
                visible_id=f"Molecule {molecule_number}",
                stable_id=parent_id,
                analog_id=analog_id,
                level=2,
                heading=f"### Molecule {molecule_number}",
                containing_unit_id=stable_id,
            )
            add_records(
                [(card_id, card) for card_id, card in cards
                 if str(card.get("reference_smiles") or "") == smiles],
                containing_unit_id=parent_id,
            )
    return rows


def validate_prompt_index(user_text: str, index: list[Mapping[str, Any]]) -> None:
    expected = Counter(str(row["prompt_heading"]) for row in index)
    actual = Counter(line for line in user_text.splitlines() if REFERENCE_LINE_RE.fullmatch(line))
    if actual != expected:
        raise ValueError("reasoning-reference index disagrees with rendered prompt")


def validate_unique_visible_ids(index: list[Mapping[str, Any]]) -> None:
    labels = Counter(str(row["visible_id"]).casefold() for row in index)
    duplicates = sorted(label for label, count in labels.items() if count > 1)
    if duplicates:
        raise ValueError(
            "contracted prompt has ambiguous reasoning-reference identifiers: "
            + ", ".join(duplicates)
        )
