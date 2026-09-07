from tools.chembl_tool.tasks.ames.build_conditioned_source import (
    BACTERIAL_CONTEXT,
    _identity,
    collapse_study_votes,
)
from tools.chembl_tool.tasks.ames.identity_review import (
    identity_status,
    load_name_resolutions,
)
import pytest


@pytest.mark.parametrize("existing_partial", [False, True])
@pytest.mark.parametrize("failure", ["interrupted", "wrong_content"])
def test_raw_restore_retries_atomically_and_reuses_verified_file(
    tmp_path, monkeypatch, existing_partial, failure,
):
    import hashlib
    import json
    import subprocess
    from tools.chembl_tool.tasks.ames import build_conditioned_source as source

    payload = b"complete frozen source"
    target = tmp_path / "one.parquet"
    if existing_partial:
        target.write_bytes(b"old partial")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source_commit": source.COMMIT,
        "files": {target.name: {
            "original_path": "original.parquet",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }},
    }))
    monkeypatch.setattr(source, "RAW_ROOT", tmp_path)
    calls = []

    def git_show(command, *, stdout, check):
        calls.append(command)
        assert command == ["git", "show", f"{source.COMMIT}:original.parquet"]
        assert check is True
        stdout.write(b"new partial" if len(calls) == 1 else payload)
        if len(calls) == 1 and failure == "interrupted":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(source.subprocess, "run", git_show)
    error = subprocess.CalledProcessError if failure == "interrupted" else ValueError
    with pytest.raises(error):
        source.restore()
    if existing_partial:
        assert target.read_bytes() == b"old partial"
    else:
        assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
    source.restore()
    assert target.read_bytes() == payload
    source.restore()
    assert len(calls) == 2


def vote(record_id, pmid="123", label=1):
    return {
        "source_record_id": record_id,
        "pmid": pmid,
        "Y": label,
        "molecule_identity_key": "parent",
        "condition_group": "strain_panel=TA98",
    }


def test_paraphrases_do_not_increase_study_support():
    votes, audit = collapse_study_votes([vote("a"), vote("b"), vote("c", pmid="456")])
    assert len(votes) == 2
    assert votes[0]["supporting_source_record_ids"] == ["a", "b"]
    assert len(audit) == 3


def test_conflicting_extractions_cannot_outvote_each_other_within_one_study():
    votes, audit = collapse_study_votes([vote("a"), vote("b"), vote("c", label=0)])
    assert votes == []
    assert {r["reason"] for r in audit} == {"conflicting_study_parent_condition"}


def test_conditions_are_not_collapsed_across_activation_regimes():
    rows = [vote("a"), {**vote("b", label=0), "condition_group": "strain_panel=TA100"}]
    assert len(collapse_study_votes(rows)[0]) == 2


def test_unresolved_ambiguous_and_mismatched_names_are_not_identity_matches():
    resolutions = {
        "benzene": {"parent_keys": ["AAA-BBB-C"]},
        "missing": {"parent_keys": []},
        "ambiguous": {"parent_keys": ["AAA-BBB-C", "DDD-BBB-C"]},
    }
    assert (
        identity_status("benzene", "AAA-BBB-C", resolutions) == "verified_parent_match"
    )
    assert (
        identity_status("benzene", "AAA-FFF-C", resolutions)
        == "stereochemistry_or_isotope_unresolved"
    )
    assert (
        identity_status("benzene", "ZZZ-BBB-C", resolutions)
        == "name_structure_mismatch"
    )
    assert (
        identity_status("missing", "AAA-BBB-C", resolutions)
        == "unresolved_name_or_request"
    )
    assert (
        identity_status("ambiguous", "AAA-BBB-C", resolutions)
        == "ambiguous_name_resolution"
    )
    assert identity_status("unknown", "AAA-BBB-C", resolutions) == "not_queried"


def test_parent_normalization_does_not_authorize_material_or_mixture_votes():
    assert _identity("[Cu+2].[O-2]")[1]["unsupported_material_identity"]
    assert _identity("CCN(CC)C(=S)[S-].[Cd+2]")[1]["unsupported_material_identity"]
    assert _identity("CCO.CCCO")[1]["multiple_organic_components"]
    assert not _identity("CC[NH3+].[Cl-]")[1]["multiple_organic_components"]
    peroxide = _identity("OO")[1]
    assert peroxide["unsupported_material_identity"]  # gold stays strict
    assert peroxide["simple_inorganic_molecule"]
    assert not _identity("[Cd]")[1]["simple_inorganic_molecule"]
    assert not _identity("[O-2].[Zn+2]")[1]["simple_inorganic_molecule"]


def test_indirect_bacterial_scope_covers_ecoli_and_suffix_strains():
    for passage in [
        "E. coli WP2uvrA reverse mutations",
        "TA97A mutagenicity",
        "TA98NR response",
        "S. typhimurium outcome",
    ]:
        assert BACTERIAL_CONTEXT.search(passage)
    assert not BACTERIAL_CONTEXT.search("human hepatocyte DNA adduct formation")


def test_frozen_attempts_preserve_name_ambiguity_and_request_failure(tmp_path):
    import json

    attempts = [
        {
            "ok": True,
            "response": {
                "PropertyTable": {"Properties": [{"CID": cid, "SMILES": smiles}]}
            },
        }
        for cid, smiles in [(702, "CCO"), (887, "CO")]
    ]
    rows = [
        {
            "name": "conflicting",
            "url": "frozen",
            **attempts[-1],
            "request_attempts": attempts,
        },
        {
            "name": "pending",
            "url": "frozen",
            "ok": False,
            "error": {"message": "503 Client Error"},
            "request_attempts": [{"ok": False}] * 4,
        },
        {
            "name": "absent",
            "url": "frozen",
            "ok": False,
            "error": {"message": "404 Client Error"},
        },
    ]
    path = tmp_path / "responses.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    resolved = load_name_resolutions(path)
    assert (
        identity_status("conflicting", "unused", resolved)
        == "ambiguous_name_resolution"
    )
    assert (
        identity_status("pending", "unused", resolved) == "identity_request_unresolved"
    )
    assert resolved["pending"]["n_request_attempts"] == 4
    assert identity_status("absent", "unused", resolved) == "name_not_found"
