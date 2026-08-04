"""Held-out removal is the leakage guard, so it is tested as one.

A held-out molecule that survives into the retrieval index lets the model look
up the answer instead of predicting it, and nothing raises when that happens --
the index simply looks fine and every number computed from it is wrong.  These
tests therefore assert the invariant at the artifact that is actually queried
(the neighbour index), not only at the intermediate record view.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.tasks.bioavailability_ma import (
    build_starling_downstream_artifacts as downstream,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_downstream_artifacts import (
    AUDIT_STAGE,
    DOWNSTREAM_STAGES,
    EXCLUSIONS_FILENAME,
    FILTERED_RECORDS_FILENAME,
    HELDOUT_STAGE,
    LEGACY_DOWNSTREAM_STAGES,
    MOLECULE_EVIDENCE_STAGE,
    NEIGHBOR_INDEX_STAGE,
    TRANSFER_POLICY_STAGE,
    build_downstream_artifacts,
    build_filtered_molecule_evidence,
    build_filtered_neighbor_indices,
    materialize_filtered_record_views,
)
from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    OUTPUT_FIELDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_compact_artifacts import (
    load_compact_neighbor_index,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_bucket_transfer_policy import (
    SOURCE_CANDIDATE_FIELDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    family_assignment,
)

DIRECT = "direct_hf"


def _label_row(smiles: str) -> dict:
    identity = normalize_molecule_identity(smiles)
    return {
        "drug": identity.parent_smiles or identity.canonical_smiles,
        "molecule_identity_key": identity.parent_inchi_key or identity.parent_smiles,
        "molecule_identity": identity.to_dict(),
        "Y": 1,
    }


def _write_labels(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(record_id: str, source_id: str, smiles: str, *, value: float = 40.0) -> dict:
    """One normalized record carrying every column the real builders read."""
    row: dict = {column: None for column in SOURCE_COLUMNS[source_id]}
    row.update({column: None for column in SOURCE_CANDIDATE_FIELDS[source_id]})
    row.update({column: None for column in SOURCE_PAIR_FIELDS[source_id]})
    endpoint = "oral_bioavailability" if source_id == DIRECT else "absorption"
    row.update(
        {
            "normalized_record_id": record_id,
            "source_id": source_id,
            "source_name": source_id,
            "canonical_smiles": smiles,
            # Derived rather than restated so a family rename cannot leave this
            # file asserting against a group id the pipeline no longer emits.
            "group_id": family_assignment(source_id, endpoint).group_id,
            "endpoint_name": endpoint,
            "canonical_endpoint": endpoint,
            "canonical_unit": "%",
            "normalization_validity_status": "valid",
            "retrieval_eligible": True,
            "finite_scalar_value": value,
            "confidence": 0.9,
            "support_text": f"evidence for {record_id}",
            "global_context": "oral administration",
            "global_species_context": "human",
        }
    )
    return row


def _write_splits(split_root: Path, splits: dict[str, tuple[str, str]]) -> None:
    for split, (train_smiles, heldout_smiles) in splits.items():
        _write_labels(
            split_root / split / "train_molecule_labels.jsonl",
            [_label_row(train_smiles)],
        )
        _write_labels(
            split_root / split / "heldout_molecule_labels.jsonl",
            [_label_row(heldout_smiles)],
        )


def _fixture(tmp_path, records: list[dict] | None = None):
    split_root = tmp_path / "splits"
    _write_splits(split_root, {"random": ("CCN", "CCO"), "scaffold": ("CCO", "CCN")})
    if records is None:
        records = [
            _record("direct-ethanol", DIRECT, "CCO", value=40.0),
            _record("direct-ethylamine", DIRECT, "CCN", value=55.0),
            _record("fa-ethanol", "fa", "CCO", value=70.0),
            _record("fg-ethanol", "fg", "CCO", value=80.0),
            _record("fh-ethanol", "fh", "CCO", value=90.0),
            _record("exposure-ethanol", "oral_exposure", "CCO"),
        ]
    records_path = tmp_path / "03_records" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(records_path, index=False)
    return records_path, split_root, records


def _write_auxiliary_manifest(root: Path) -> None:
    manifest = root / "02_normalized" / "auxiliary_mapping_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "mapping_version": MAPPING_VERSION,
                "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
                "output_fields": list(OUTPUT_FIELDS),
                "mapping_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_remove_heldout_overlap_materializes_split_views_and_keeps_assays(tmp_path):
    records_path, split_root, records = _fixture(tmp_path)
    target = tmp_path / "06_remove_heldout_overlap"

    manifest = materialize_filtered_record_views(
        records_path=records_path,
        benchmark_split_root=split_root,
        out_dir=target,
    )

    random = pd.read_parquet(target / "random" / FILTERED_RECORDS_FILENAME)
    scaffold = pd.read_parquet(target / "scaffold" / FILTERED_RECORDS_FILENAME)
    mechanistic = {"fa-ethanol", "fg-ethanol", "fh-ethanol", "exposure-ethanol"}
    assert set(random["normalized_record_id"]) == {"direct-ethylamine", *mechanistic}
    assert set(scaffold["normalized_record_id"]) == {"direct-ethanol", *mechanistic}
    assert list(random.columns) == list(pd.DataFrame(records).columns)
    exclusions = pd.read_parquet(target / EXCLUSIONS_FILENAME)
    assert set(map(tuple, exclusions[["benchmark_split", "normalized_record_id"]].values)) == {
        ("random", "direct-ethanol"),
        ("scaffold", "direct-ethylamine"),
    }
    assert manifest["filter_source_id"] == DIRECT
    assert manifest["policy_statistics_scope"] == "complete_unfiltered_records"
    for split in ("random", "scaffold"):
        validations = manifest["splits"][split]["validations"]
        # `train_heldout_parent_overlap` is a count living among booleans, so a
        # blanket `all(...)` would read its healthy 0 as a failure.
        assert validations["train_heldout_parent_overlap"] == 0
        assert validations["only_filter_source_removed"]
        assert validations[f"only_{DIRECT}_removed"]
        assert validations["excluded_ids_absent"]
        assert validations["schema_matches_stage_03"]
        assert validations["heldout_parent_overlap_zero"]


def test_a_heldout_parent_is_removed_even_as_a_salt(tmp_path):
    """Filtering is parent-based; a salt form must not slip through.

    This is the failure mode that leaks silently: the held-out molecule is
    stored under a different SMILES string, a naive string match misses it, and
    its label lands in the index looking like ordinary evidence.
    """
    records = [
        _record("direct-ethanol-salt", DIRECT, "CCO.Cl"),
        _record("direct-ethylamine", DIRECT, "CCN"),
    ]
    records_path, split_root, _ = _fixture(tmp_path, records)
    target = tmp_path / HELDOUT_STAGE

    materialize_filtered_record_views(
        records_path=records_path,
        benchmark_split_root=split_root,
        out_dir=target,
    )

    random = pd.read_parquet(target / "random" / FILTERED_RECORDS_FILENAME)
    assert "direct-ethanol-salt" not in set(random["normalized_record_id"])
    scaffold = pd.read_parquet(target / "scaffold" / FILTERED_RECORDS_FILENAME)
    assert "direct-ethanol-salt" in set(scaffold["normalized_record_id"])


def test_a_parent_heldout_in_both_splits_is_removed_from_both(tmp_path):
    """The random and scaffold splits share held-out parents in practice."""
    split_root = tmp_path / "splits"
    _write_splits(split_root, {"random": ("CCN", "CCO"), "scaffold": ("CCN", "CCO")})
    records = [
        _record("direct-ethanol", DIRECT, "CCO"),
        _record("direct-ethylamine", DIRECT, "CCN"),
    ]
    records_path = tmp_path / "03_records" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(records_path, index=False)
    target = tmp_path / HELDOUT_STAGE

    materialize_filtered_record_views(
        records_path=records_path,
        benchmark_split_root=split_root,
        out_dir=target,
    )

    for split in ("random", "scaffold"):
        kept = set(
            pd.read_parquet(target / split / FILTERED_RECORDS_FILENAME)[
                "normalized_record_id"
            ]
        )
        assert "direct-ethanol" not in kept
        assert "direct-ethylamine" in kept


def test_an_unparseable_structure_neither_crashes_nor_is_dropped(tmp_path):
    """A record we cannot normalise is not held out, but must not break the run."""
    records = [
        _record("direct-broken", DIRECT, "not-a-smiles"),
        _record("direct-ethanol", DIRECT, "CCO"),
    ]
    records_path, split_root, _ = _fixture(tmp_path, records)
    target = tmp_path / HELDOUT_STAGE

    materialize_filtered_record_views(
        records_path=records_path,
        benchmark_split_root=split_root,
        out_dir=target,
    )

    random = pd.read_parquet(target / "random" / FILTERED_RECORDS_FILENAME)
    assert "direct-broken" in set(random["normalized_record_id"])
    assert "direct-ethanol" not in set(random["normalized_record_id"])


def test_filtered_evidence_and_indices_reference_only_retained_records(tmp_path):
    records_path, split_root, _ = _fixture(tmp_path)
    filtered_root = tmp_path / "06_remove_heldout_overlap"
    evidence_root = tmp_path / "07_molecule_evidence"
    index_root = tmp_path / "08_neighbor_index"
    materialize_filtered_record_views(
        records_path=records_path,
        benchmark_split_root=split_root,
        out_dir=filtered_root,
    )
    build_filtered_molecule_evidence(
        filtered_records_root=filtered_root,
        out_dir=evidence_root,
    )
    build_filtered_neighbor_indices(
        evidence_root=evidence_root,
        overlap_manifest=filtered_root / "manifest.json",
        out_dir=index_root,
    )

    random_bridge = pd.read_parquet(
        evidence_root / "random" / "molecule_family_records.parquet"
    )
    scaffold_bridge = pd.read_parquet(
        evidence_root / "scaffold" / "molecule_family_records.parquet"
    )
    assert "direct-ethanol" not in set(random_bridge["normalized_record_id"])
    assert "direct-ethylamine" not in set(scaffold_bridge["normalized_record_id"])
    assert "fa-ethanol" in set(random_bridge["normalized_record_id"])
    assert "fa-ethanol" in set(scaffold_bridge["normalized_record_id"])

    random_index = load_compact_neighbor_index(index_root / "random")
    scaffold_index = load_compact_neighbor_index(index_root / "scaffold")
    assert random_index["group_to_molecule_indices"]
    assert scaffold_index["group_to_molecule_indices"]


# --- end-to-end success ----------------------------------------------------


def test_complete_build_publishes_a_tree_with_no_heldout_label_evidence(tmp_path):
    """The success path, and the invariant that actually matters.

    Every other test here proves the pipeline fails safely.  This one proves it
    works: run it to completion and assert no held-out molecule's label record
    reaches the artifact the model queries.
    """
    _, split_root, _ = _fixture(tmp_path)
    _write_auxiliary_manifest(tmp_path)

    payload = build_downstream_artifacts(
        normalized_root=tmp_path,
        benchmark_split_root=split_root,
    )

    for stage in DOWNSTREAM_STAGES:
        assert (tmp_path / stage).is_dir(), stage
    assert not list(tmp_path.glob(".downstream-build-*"))
    assert not list(tmp_path.glob(".downstream-backup-*"))

    # The held-out molecule's own label record must be gone from each split's
    # evidence, while the same molecule's mechanistic evidence survives.
    direct_group = family_assignment(DIRECT, "oral_bioavailability").group_id
    heldout_by_split = {
        "random": ("direct-ethanol", "CCO"),
        "scaffold": ("direct-ethylamine", "CCN"),
    }
    for split, (excluded_id, heldout_smiles) in heldout_by_split.items():
        bridge = pd.read_parquet(
            tmp_path / MOLECULE_EVIDENCE_STAGE / split / "molecule_family_records.parquet"
        )
        assert excluded_id not in set(bridge["normalized_record_id"])

        index = load_compact_neighbor_index(tmp_path / NEIGHBOR_INDEX_STAGE / split)
        direct_positions = index["group_to_molecule_indices"].get(direct_group, [])
        direct_smiles = {
            index["molecules"][position]["canonical_smiles"]
            for position in direct_positions
        }
        # This is the leakage assertion: the held-out molecule must not be
        # reachable through the direct-evidence group of the queried index.
        assert heldout_smiles not in direct_smiles, (split, direct_smiles)

    audit = json.loads(
        (tmp_path / AUDIT_STAGE / "heldout_overlap.json").read_text(encoding="utf-8")
    )
    assert audit["validations"]["pair_buckets_precede_split_filter"]
    assert audit["validations"]["filtered_evidence_only"]
    assert audit["validations"]["filtered_neighbor_indices_only"]
    # Deliberate per-task divergence, pinned here so it is not "harmonised"
    # away: Bioavailability (like Skin) declares no `heldout_sources` on its
    # transfer policy.  Held-out filtering is a pipeline stage applied after the
    # policy, so the policy's own calibration deliberately sees the complete
    # unfiltered records and publishes no `heldout_exclusion` block.  BBB
    # Martins deliberately does the opposite: it declares `direct_bbb` as a
    # held-out source and calibrates on
    # `heldout_gold_filtered_transfer_calibration`.  Both are correct for their
    # task; changing either has to be a deliberate decision, not a tidy-up.
    assert audit["validations"]["transfer_policy_has_declared_calibration_filter"] is False
    assert audit["transfer_policy_heldout_exclusion"] is None
    assert audit["policy_statistics_scope"] == "complete_unfiltered_records"
    assert audit["filter_source_id"] == DIRECT
    policy = json.loads(
        gzip.decompress(
            (
                tmp_path / TRANSFER_POLICY_STAGE / "pair_bucket_transfer_policy.json.gz"
            ).read_bytes()
        ).decode("utf-8")
    )
    assert "heldout_exclusion" not in policy

    # Success-path bookkeeping that no other test reaches.
    hashes = payload["downstream_artifact_hashes"]
    assert hashes and all(isinstance(value, str) for value in hashes.values())
    assert payload["downstream_scope"]["filter_source_id"] == DIRECT
    assert payload["downstream_scope"]["unfiltered_record_stage"] == "03_records"

    for split in ("random", "scaffold"):
        index_manifest = json.loads(
            (tmp_path / NEIGHBOR_INDEX_STAGE / split / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert index_manifest["benchmark_split"] == split
        assert index_manifest["unfiltered_index"] is False
        assert index_manifest["filtered_source_overlap"] is True
        assert index_manifest["heldout_overlap_manifest_sha256"]


def test_no_candidate_build_path_leaks_into_the_published_tree(tmp_path):
    """A leaked temp path would make the published manifests unreproducible."""
    _, split_root, _ = _fixture(tmp_path)
    _write_auxiliary_manifest(tmp_path)

    build_downstream_artifacts(
        normalized_root=tmp_path,
        benchmark_split_root=split_root,
    )

    for path in tmp_path.rglob("*.json"):
        assert ".downstream-build-" not in path.read_text(encoding="utf-8"), path
    policy = next((tmp_path / TRANSFER_POLICY_STAGE).glob("*.json.gz"))
    assert ".downstream-build-" not in gzip.decompress(policy.read_bytes()).decode()


# --- transactional guarantees ----------------------------------------------


def test_downstream_build_failure_preserves_active_tree(monkeypatch, tmp_path):
    _, split_root, _ = _fixture(tmp_path)
    auxiliary = tmp_path / "02_normalized" / "auxiliary_mapping_manifest.json"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_text("{}\n", encoding="utf-8")
    _seed_active_tree(tmp_path, prefix="old")
    before = _active_tree_bytes(tmp_path)

    def fail_after_preflight(**_kwargs):
        raise RuntimeError("injected pair-bucket failure")

    monkeypatch.setattr(downstream, "build_sidecar", fail_after_preflight)
    with pytest.raises(RuntimeError, match="injected pair-bucket failure"):
        build_downstream_artifacts(
            normalized_root=tmp_path,
            benchmark_split_root=split_root,
        )

    assert _active_tree_bytes(tmp_path) == before
    assert not list(tmp_path.glob(".downstream-build-*"))
    assert not list(tmp_path.glob(".downstream-backup-*"))


def test_late_index_build_failure_preserves_active_tree(monkeypatch, tmp_path):
    _, split_root, _ = _fixture(tmp_path)
    auxiliary = tmp_path / "02_normalized" / "auxiliary_mapping_manifest.json"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_text("{}\n", encoding="utf-8")
    _seed_active_tree(tmp_path, prefix="old")
    before = _active_tree_bytes(tmp_path)
    _stub_candidate_build_until_index(monkeypatch)

    with pytest.raises(RuntimeError, match="injected index failure"):
        build_downstream_artifacts(
            normalized_root=tmp_path,
            benchmark_split_root=split_root,
        )

    assert _active_tree_bytes(tmp_path) == before
    assert not list(tmp_path.glob(".downstream-build-*"))
    assert not list(tmp_path.glob(".downstream-backup-*"))


def test_publication_failure_rolls_back_every_stage(monkeypatch, tmp_path):
    root = tmp_path / "published"
    root.mkdir()
    _seed_active_tree(root, prefix="old")
    candidate = root / ".candidate"
    _seed_candidate_tree(candidate, prefix="new")
    before = _active_tree_bytes(root)
    real_replace = downstream.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(source) == candidate / downstream.HELDOUT_STAGE and not failed:
            failed = True
            raise OSError("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(downstream.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected publication failure"):
        downstream._publish_downstream_candidate(root, candidate)

    assert failed
    assert _active_tree_bytes(root) == before
    assert not list(root.glob(".downstream-backup-*"))


def test_successful_publication_replaces_complete_tree(tmp_path):
    root = tmp_path / "published"
    root.mkdir()
    _seed_active_tree(root, prefix="old")
    candidate = root / ".candidate"
    _seed_candidate_tree(candidate, prefix="new")

    downstream._publish_downstream_candidate(root, candidate)

    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == {
        "generation": "new"
    }
    assert all(
        (root / stage / "marker.txt").read_text(encoding="utf-8")
        == f"new:{stage}\n"
        for stage in DOWNSTREAM_STAGES
    )
    assert all(not (root / stage).exists() for stage in LEGACY_DOWNSTREAM_STAGES)
    assert not list(root.glob(".downstream-backup-*"))


def _seed_active_tree(root: Path, *, prefix: str) -> None:
    for stage in DOWNSTREAM_STAGES:
        target = root / stage
        target.mkdir(parents=True, exist_ok=True)
        (target / "marker.txt").write_text(f"{prefix}:{stage}\n", encoding="utf-8")
    legacy = root / LEGACY_DOWNSTREAM_STAGES[0]
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "marker.txt").write_text(f"{prefix}:legacy\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"generation": prefix}) + "\n", encoding="utf-8"
    )


def _seed_candidate_tree(root: Path, *, prefix: str) -> None:
    root.mkdir(parents=True)
    for stage in DOWNSTREAM_STAGES:
        target = root / stage
        target.mkdir(parents=True)
        (target / "marker.txt").write_text(f"{prefix}:{stage}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"generation": prefix}) + "\n", encoding="utf-8"
    )


def _active_tree_bytes(root: Path) -> dict[str, bytes]:
    paths = [root / "manifest.json"]
    for stage in (*DOWNSTREAM_STAGES, *LEGACY_DOWNSTREAM_STAGES):
        target = root / stage
        if target.is_dir():
            paths.extend(path for path in target.rglob("*") if path.is_file())
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(paths)
        if path.is_file()
    }


def _stub_candidate_build_until_index(monkeypatch) -> None:
    def sidecar(*, out_dir, **_kwargs):
        target = Path(out_dir)
        target.mkdir(parents=True)
        (target / downstream.PAIR_BUCKET_RECORDS_FILENAME).write_bytes(b"pairs")
        metadata = {
            "stats": {"buckets": 1},
            "output": {
                "path": str(target / downstream.PAIR_BUCKET_RECORDS_FILENAME)
            },
        }
        (target / downstream.PAIR_BUCKET_METADATA_FILENAME).write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return metadata

    def transfer(*, out_dir, **_kwargs):
        Path(out_dir).mkdir(parents=True)
        return {"summary": {"assay_transfer_eligible_buckets": 1}}

    def filtered(*_args, out_dir, **_kwargs):
        Path(out_dir).mkdir(parents=True)
        return {
            "splits": {
                split: {"excluded_direct_hf_records": 1, "filtered_records": 3}
                for split in downstream.BENCHMARK_SPLITS
            }
        }

    def evidence(*_args, out_dir, **_kwargs):
        target = Path(out_dir)
        for split in downstream.BENCHMARK_SPLITS:
            (target / split).mkdir(parents=True)
        return {
            split: {"families": 2} for split in downstream.BENCHMARK_SPLITS
        }

    def fail_index(*_args, **_kwargs):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(downstream, "build_sidecar", sidecar)
    monkeypatch.setattr(downstream, "build_pair_bucket_transfer_policy", transfer)
    monkeypatch.setattr(
        downstream._shared, "materialize_filtered_record_views", filtered
    )
    monkeypatch.setattr(
        downstream._shared, "build_filtered_molecule_evidence", evidence
    )
    monkeypatch.setattr(
        downstream._shared, "build_filtered_neighbor_indices", fail_index
    )
