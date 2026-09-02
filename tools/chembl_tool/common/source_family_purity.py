"""Shared types for retrieval-only source-family reassignment."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FamilyMove:
    """One auditable source-row reassignment."""

    new_group: str
    reason: str


def audit_exact_voter_membership(
    records: Iterable[Mapping[str, Any]],
    voter_ids: set[Hashable] | frozenset[Hashable],
    *,
    direct_group: str,
    record_id: Callable[[Mapping[str, Any]], Hashable | None],
) -> dict[str, Any]:
    """Require source-layer L1 membership to equal the present voter ledger."""

    l1_records = l1_nonvoters = voter_records_outside_l1 = 0
    eligible_l1 = eligible_l1_nonvoters = eligible_voters_outside_l1 = 0
    saw_retrieval_eligibility = False
    present_voters: set[Hashable] = set()
    for row in records:
        key = record_id(row)
        is_voter = key is not None and key in voter_ids
        if is_voter:
            present_voters.add(key)
        is_direct = str(row.get("group_id") or "") == direct_group
        if is_direct:
            l1_records += 1
            l1_nonvoters += int(not is_voter)
        elif is_voter:
            voter_records_outside_l1 += 1
        if "retrieval_eligible" in row:
            saw_retrieval_eligibility = True
            if row.get("retrieval_eligible") is True:
                if is_direct:
                    eligible_l1 += 1
                    eligible_l1_nonvoters += int(not is_voter)
                elif is_voter:
                    eligible_voters_outside_l1 += 1

    absent_voters = sorted(map(str, voter_ids - present_voters))
    passed = l1_nonvoters == 0 and voter_records_outside_l1 == 0
    audit = {
        "l1_exact_vote_membership": passed,
        "n_l1_records": l1_records,
        "n_l1_nonvoter_records": l1_nonvoters,
        "n_voter_records_outside_l1": voter_records_outside_l1,
        "n_present_voter_ids": len(present_voters),
        "n_voter_ids_not_present_in_source": len(absent_voters),
        "voter_ids_not_present_examples": absent_voters[:20],
    }
    if saw_retrieval_eligibility:
        audit.update(
            {
                "retrieval_eligible_l1_exact_vote_membership": (
                    eligible_l1_nonvoters == 0
                    and eligible_voters_outside_l1 == 0
                ),
                "n_retrieval_eligible_l1_records": eligible_l1,
                "n_retrieval_eligible_l1_nonvoter_records": eligible_l1_nonvoters,
                "n_retrieval_eligible_voter_records_outside_l1": (
                    eligible_voters_outside_l1
                ),
            }
        )
    if not passed:
        raise RuntimeError(f"exact voter-membership gate failed: {audit}")
    return audit
