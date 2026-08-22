"""Small shared primitives for proposing exact external-condition signatures.

Pattern matches only create review candidates.  They are never sufficient to
enter a gold benchmark; the reviewed-conditioned builder requires a terminal
record-level review ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Iterable, Pattern, Sequence


@dataclass(frozen=True, order=True)
class ConditionAtom:
    family: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.family}={self.value}"


@dataclass(frozen=True)
class AtomRule:
    atom: ConditionAtom
    pattern: Pattern[str]

    @classmethod
    def make(cls, family: str, value: str, pattern: str) -> "AtomRule":
        return cls(ConditionAtom(family, value), re.compile(pattern, re.IGNORECASE))


@dataclass(frozen=True)
class ConditionProposal:
    status: str
    signature: str | None
    atoms: tuple[ConditionAtom, ...]
    reason: str
    normalized_text: str


def normalize_condition_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip(" .;,\t\n").lower()


def propose_pattern_condition(
    value: object,
    *,
    rules: Sequence[AtomRule],
    excluded_patterns: Sequence[Pattern[str]] = (),
    incompatible_families: Iterable[str] = (),
    suppress: Callable[[set[ConditionAtom]], set[ConditionAtom]] | None = None,
) -> ConditionProposal:
    text = normalize_condition_text(value)
    if not text or text in {"nan", "none", "null", "n/a", "na"}:
        return ConditionProposal("not_candidate", None, (), "no_reported_condition", text)
    for pattern in excluded_patterns:
        if pattern.search(text):
            return ConditionProposal("not_candidate", None, (), "excluded_semantic_class", text)
    atoms = {rule.atom for rule in rules if rule.pattern.search(text)}
    if suppress:
        atoms = suppress(atoms)
    if not atoms:
        return ConditionProposal("not_candidate", None, (), "no_supported_external_atom", text)
    incompatible = set(incompatible_families)
    by_family: dict[str, set[str]] = {}
    for atom in atoms:
        by_family.setdefault(atom.family, set()).add(atom.value)
    conflicts = sorted(family for family, values in by_family.items() if family in incompatible and len(values) > 1)
    if conflicts:
        ordered = tuple(sorted(atoms))
        return ConditionProposal(
            "needs_review",
            None,
            ordered,
            f"incompatible_values:{'+'.join(conflicts)}",
            text,
        )
    ordered = tuple(sorted(atoms))
    return ConditionProposal(
        "needs_review",
        "+".join(atom.key for atom in ordered),
        ordered,
        "pattern_proposed_external_condition",
        text,
    )


def payload_sha256(fields: Sequence[object]) -> str:
    serialized = "\0".join(str(value or "") for value in fields)
    import hashlib

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
