import json

import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tools.chembl_tool.common.coverage_reasoning import (
    COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
    STANDARD_NEIGHBOR_CONTEXT,
    attach_neighbor_context,
    augment_group_messages_with_neighbor_context,
)
from tools.chembl_tool.common.identity_blind import prepare_reasoning_retrieval


MORGAN = {
    "type": "RDKit-Morgan",
    "radius": 2,
    "n_bits": 2048,
    "useFeatures": False,
    "useChirality": False,
    "useBondTypes": True,
}


def _retrieval():
    return {
        "query": {
            "input_smiles": "CCOc1ccccc1C(=O)N",
            "canonical_smiles": "CCOc1ccccc1C(=O)N",
            "fingerprint": MORGAN,
        },
        "experiment": {"mode": "full_mechanism"},
        "groups": [
            {
                "group_id": "Mechanism.test",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "N1",
                        "canonical_smiles": "CCOc1ccccc1",
                        "similarity": 0.7,
                        "evidence_rows": [],
                    },
                    {
                        "rank": 2,
                        "molecule_chembl_id": "N2",
                        "canonical_smiles": "NC(=O)c1ccccc1",
                        "similarity": 0.6,
                        "evidence_rows": [],
                    },
                ],
            }
        ],
    }


def test_standard_neighbor_context_is_exact_noop():
    retrieval = _retrieval()

    result = attach_neighbor_context(
        retrieval,
        profile=STANDARD_NEIGHBOR_CONTEXT,
    )

    assert result is retrieval
    assert "neighbor_set_context" not in retrieval["groups"][0]


def test_coverage_context_is_noop_when_no_neighbors_exist():
    retrieval = {
        "query": {"canonical_smiles": "CCO"},
        "groups": [],
        "experiment": {"mode": "none"},
    }

    result = attach_neighbor_context(
        retrieval,
        profile=COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    )

    assert result is retrieval


def test_coverage_context_reports_sequential_feature_and_atom_coverage():
    retrieval = _retrieval()

    result = attach_neighbor_context(
        retrieval,
        profile=COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    )

    assert result is not retrieval
    assert "neighbor_set_context" not in retrieval["groups"][0]
    context = result["groups"][0]["neighbor_set_context"]
    entries = context["neighbors"]
    assert [entry["rank"] for entry in entries] == [1, 2]
    assert context["query_feature_count"] > 0
    assert context["query_atom_count"] == 12
    assert 0 < context["selected_set_query_feature_coverage"] <= 1
    assert 0 < context["selected_set_query_atom_coverage"] <= 1
    assert entries[1]["cumulative_query_feature_coverage"] >= entries[0][
        "cumulative_query_feature_coverage"
    ]
    assert entries[1]["cumulative_query_atom_coverage"] >= entries[0][
        "cumulative_query_atom_coverage"
    ]
    assert entries[1]["marginal_new_query_feature_count"] > 0
    assert entries[1]["coverage_role"] == "complementary"
    assert all(size > 0 for entry in entries for size in entry["marginal_region_sizes"])

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_bits = set(
        generator.GetFingerprint(Chem.MolFromSmiles("CCOc1ccccc1C(=O)N")).GetOnBits()
    )
    neighbor_union = set().union(
        *(
            set(generator.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits())
            for smiles in ("CCOc1ccccc1", "NC(=O)c1ccccc1")
        )
    )
    expected_feature_coverage = round(len(query_bits & neighbor_union) / len(query_bits), 4)
    assert context["selected_set_query_feature_coverage"] == expected_feature_coverage

    serialized = json.dumps(context)
    assert "CCOc1ccccc1" not in serialized
    assert "NC(=O)c1ccccc1" not in serialized
    assert "bit_ids" not in serialized


def test_coverage_context_requires_morgan_fingerprint():
    retrieval = _retrieval()
    retrieval["query"]["fingerprint"] = {"type": "MiniMol-descriptor"}

    with pytest.raises(ValueError, match="requires an RDKit Morgan"):
        attach_neighbor_context(
            retrieval,
            profile=COVERAGE_AWARE_NEIGHBOR_CONTEXT,
        )


def test_group_message_augmentation_is_opt_in_and_schema_preserving():
    messages = [{"role": "user", "content": "base"}]
    plain_group = _retrieval()["groups"][0]
    aware_group = attach_neighbor_context(
        _retrieval(),
        profile=COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    )["groups"][0]

    assert augment_group_messages_with_neighbor_context(messages, plain_group) is messages
    augmented = augment_group_messages_with_neighbor_context(messages, aware_group)
    assert len(augmented) == 2
    assert messages == [{"role": "user", "content": "base"}]
    payload = json.loads(augmented[-1]["content"])
    assert payload["neighbor_set_context"]["profile"] == "coverage_aware"
    assert "required_json_schema" not in payload


class _FakeToolService:
    def invoke_many(self, calls):
        return [
            {
                "tool_name": tool_name,
                "status": "ok",
                "content": "prefetched result",
                "warnings": [],
                "errors": [],
            }
            for tool_name, _ in calls
        ]


class _RecordingMmpToolService:
    def __init__(self):
        self.calls = []

    def invoke_many(self, calls):
        self.calls.extend(calls)
        return [
            {
                "tool_name": tool_name,
                "status": "ok",
                "content": (
                    "[mmp_structure_compare]\n"
                    "Maximum common substructure coverage: query=0.75, reference=0.80.\n"
                    "mmpdb matched-pair transformation: *N -> *O on shared constant *c1ccccc1."
                ),
                "warnings": [],
                "errors": [],
            }
            for tool_name, _ in calls
        ]


def test_coverage_context_survives_identity_blind_prefetch_without_identity_leak():
    retrieval = _retrieval()

    result = prepare_reasoning_retrieval(
        retrieval,
        _FakeToolService(),
        identity_blind=True,
        harness_prefetch_tools=True,
        neighbor_context_profile=COVERAGE_AWARE_NEIGHBOR_CONTEXT,
    )

    serialized = json.dumps(result)
    assert result["groups"][0]["neighbor_set_context"]["profile"] == "coverage_aware"
    assert "CCOc1ccccc1C(=O)N" not in serialized
    assert "CCOc1ccccc1" not in serialized
    assert "NC(=O)c1ccccc1" not in serialized
    assert "N1" not in serialized
    assert "N2" not in serialized


def test_mmp_ledger_prefetches_existing_tool_once_per_visible_neighbor():
    retrieval = _retrieval()
    service = _RecordingMmpToolService()

    result = prepare_reasoning_retrieval(
        retrieval,
        service,
        identity_blind=False,
        harness_prefetch_tools=False,
        neighbor_context_profile=COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
    )

    assert len(service.calls) == 2
    assert {tool_name for tool_name, _ in service.calls} == {"mmp_structure_compare"}
    assert "neighbor_set_context" not in retrieval["groups"][0]
    context = result["groups"][0]["neighbor_set_context"]
    assert context["profile"] == "coverage_mmp_ledger"
    assert context["version"] == "morgan_coverage_mmp_ledger.v2"
    assert context["pairwise_structure_comparisons_complete"] is True
    assert "selected_set_query_atom_coverage" not in context
    assert all(
        "cumulative_query_atom_coverage" not in entry
        and "marginal_region_sizes" not in entry
        for entry in context["neighbors"]
    )
    comparisons = [
        entry["pairwise_structure_comparison"] for entry in context["neighbors"]
    ]
    assert all(item["tool"] == "mmp_structure_compare.v1" for item in comparisons)
    assert all("shared constant" in item["text"] for item in comparisons)

    messages = augment_group_messages_with_neighbor_context(
        [{"role": "user", "content": "base"}], result["groups"][0]
    )
    payload = json.loads(messages[-1]["content"])
    assert "set-level structural coverage ledger" in payload["task"]
    assert any("uncovered" in instruction for instruction in payload["instructions"])
    assert "required_json_schema" not in payload


def test_mmp_ledger_rejects_identity_blind_visibility():
    with pytest.raises(ValueError, match="visible-only"):
        prepare_reasoning_retrieval(
            _retrieval(),
            _RecordingMmpToolService(),
            identity_blind=True,
            harness_prefetch_tools=True,
            neighbor_context_profile=COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
        )
