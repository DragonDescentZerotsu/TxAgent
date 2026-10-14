"""Verify, materialize and publish reviewed semantic successors explicitly."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable
import pandas as pd
from .inputs import REPOSITORY_ROOT, RELEASE_ROOT, normalize_task
from predict.utils.json import write_json_atomic

SCHEMA_VERSION = "semantic_buckets.release.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> Iterable[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()))


def tree_inventory(root: Path) -> dict[str, Any]:
    files = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})
        total_bytes += size
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\x00")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\x00")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": files,
    }


def _id(*parts: object) -> str:
    return (
        "sbref_"
        + hashlib.sha256(
            json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:24]
    )


def _reviewed_changes(
    mapping: pd.DataFrame,
    sizes: pd.DataFrame,
    review: dict,
    proposals: pd.DataFrame,
    flags: pd.DataFrame,
) -> tuple[dict[str, str], dict]:
    buckets = set(mapping.semantic_bucket_id.astype(str))
    atom_sets = mapping.groupby("semantic_bucket_id").atom_id.apply(set).to_dict()
    size_lookup = dict(zip(sizes.semantic_bucket_id.astype(str), sizes.record_count))
    scope = {
        bucket: (str(rows.level.iloc[0]), str(rows.source_id.iloc[0]))
        for bucket, rows in mapping.groupby("semantic_bucket_id")
    }
    proposed = {
        frozenset(json.loads(row.member_bucket_ids_json))
        for row in proposals.itertuples(index=False)
    }
    risky = set(
        flags[flags.risk.isin(["medium", "high"])].semantic_bucket_id.astype(str)
    )
    replacement: dict[str, str] = {}
    touched: set[str] = set()
    for entry in review.get("merges", []):
        members = [str(value) for value in entry["member_bucket_ids"]]
        if (
            len(members) < 2
            or len(set(members)) != len(members)
            or frozenset(members) not in proposed
            or any(
                (
                    bucket not in buckets
                    or size_lookup[bucket] >= 100
                    or bucket in touched
                    for bucket in members
                )
            )
            or (len({scope[bucket] for bucket in members}) != 1)
            or (not str(entry.get("rationale", "")).strip())
        ):
            raise ValueError(f"invalid or unreviewed merge: {members}")
        successor = _id("merge", sorted(members))
        for bucket in members:
            touched.add(bucket)
            replacement.update({atom: successor for atom in atom_sets[bucket]})
    for entry in review.get("splits", []):
        bucket = str(entry["old_bucket_id"])
        groups = [set(map(str, group)) for group in entry["atom_groups"]]
        if (
            bucket not in risky
            or bucket in touched
            or len(groups) < 2
            or any((not group for group in groups))
            or (sum((len(group) for group in groups)) != len(atom_sets[bucket]))
            or (set().union(*groups) != atom_sets[bucket])
            or (not str(entry.get("rationale", "")).strip())
        ):
            raise ValueError(f"invalid or unreviewed split: {bucket}")
        touched.add(bucket)
        for group in groups:
            successor = _id("split", bucket, sorted(group))
            replacement.update({atom: successor for atom in group})
    return (
        replacement,
        {
            "merged_families": len(review.get("merges", [])),
            "split_families": len(review.get("splits", [])),
            "changed_atoms": len(replacement),
        },
    )


def materialize(snapshot_root: Path, decisions_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    snapshot_path = snapshot_root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    review_path = snapshot_root / "review_manifest.json"
    reviewed = json.loads(review_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if (
        snapshot.get("state") != "complete"
        or reviewed.get("status") != "candidate_review_required"
        or decisions.get("task") != snapshot["task"]
        or (decisions.get("review_manifest_sha256") != sha256_file(review_path))
        or (not str(decisions.get("reviewer", "")).strip())
    ):
        raise ValueError("decisions do not target a complete reviewed snapshot")
    for entry in snapshot["inputs"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"snapshot input changed: {entry['path']}")
    paths = snapshot["inputs"]
    mapping = pd.read_parquet(paths["semantic_map"]["path"])
    records = pd.read_parquet(paths["record_map"]["path"])
    weights = pd.read_parquet(paths["weights"]["path"])
    sizes = pd.read_csv(snapshot_root / "family_sizes.tsv", sep="\t")
    proposals = pd.read_csv(snapshot_root / "merge_proposals.tsv", sep="\t")
    flags = pd.read_csv(snapshot_root / "large_family_reviews.tsv", sep="\t")
    if mapping.atom_id.duplicated().any() or records.source_row_uid.duplicated().any():
        raise ValueError("selected map repeats an atom or physical UID")
    replacement, counts = _reviewed_changes(mapping, sizes, decisions, proposals, flags)
    successor = mapping.copy()
    successor["semantic_bucket_id"] = successor.atom_id.map(replacement).fillna(
        successor.semantic_bucket_id
    )
    successor["source_semantic_bucket_id"] = successor.semantic_bucket_id
    record_successor = records.drop(columns="semantic_bucket_id").merge(
        successor[["atom_id", "semantic_bucket_id"]],
        on="atom_id",
        how="left",
        validate="many_to_one",
    )
    if (
        len(record_successor) != len(records)
        or record_successor.semantic_bucket_id.isna().any()
        or (not record_successor.source_row_uid.equals(records.source_row_uid))
    ):
        raise ValueError("candidate changed physical UID coverage")
    original_by_atom = dict(zip(mapping.atom_id, mapping.semantic_bucket_id))
    lineage = successor[["level", "semantic_bucket_id", "atom_id"]].copy()
    lineage["predecessor_bucket_id"] = lineage.atom_id.map(original_by_atom)
    old_weights = weights[["level", "semantic_bucket_id", "weight"]].drop_duplicates(
        ["level", "semantic_bucket_id"]
    )
    joined = lineage.drop_duplicates(
        ["level", "semantic_bucket_id", "predecessor_bucket_id"]
    ).merge(
        old_weights,
        left_on=["level", "predecessor_bucket_id"],
        right_on=["level", "semantic_bucket_id"],
        how="left",
        suffixes=("", "_old"),
        validate="many_to_one",
    )
    seeds = (
        joined.groupby(["level", "semantic_bucket_id"], dropna=False)
        .agg(
            seed_weight=("weight", "mean"),
            weighted_predecessors=("weight", "count"),
            predecessors=(
                "predecessor_bucket_id",
                lambda values: json.dumps(sorted(set(values))),
            ),
        )
        .reset_index()
    )
    output.mkdir(parents=True)
    successor.sort_values(
        ["level", "source_id", "semantic_bucket_id", "atom_id"]
    ).to_parquet(output / "semantic_bucket_map.parquet", index=False)
    record_successor.sort_values(["level", "source_row_uid"]).to_parquet(
        output / "record_semantic_bucket_map.parquet", index=False
    )
    seeds.to_parquet(output / "inherited_seed_weights.parquet", index=False)
    write_json_atomic(output / "reviewed_decisions.json", decisions)
    manifest = {
        "version": "task_bucket_refinement_candidate.v1",
        "status": "candidate_unselected_awaiting_calibration",
        "task": snapshot["task"],
        "evidence_release": snapshot["release"],
        "snapshot_sha256": sha256_file(snapshot_path),
        "review_manifest_sha256": sha256_file(review_path),
        "decision_sha256": sha256_file(output / "reviewed_decisions.json"),
        "counts": {
            **counts,
            "atoms": len(successor),
            "records": len(record_successor),
            "families": int(successor.semantic_bucket_id.nunique()),
        },
        "files": {
            name: sha256_file(output / name)
            for name in (
                "semantic_bucket_map.parquet",
                "record_semantic_bucket_map.parquet",
                "inherited_seed_weights.parquet",
            )
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _selected_files(root: Path, manifest: dict) -> dict[str, Path]:
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(
        manifest.get("selected"), dict
    ):
        raise ValueError("unsupported semantic release manifest")
    paths = {}
    for role, relative in manifest["selected"].items():
        if relative is None:
            continue
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"selected semantic artifact must be relative: {role}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"missing or escaping selected semantic artifact: {role}")
        paths[role] = path
    required = {"semantic_map", "semantic_map_manifest", "semantic_bucket_weights"}
    if not required <= paths.keys() or not {
        "record_semantic_bucket_map",
        "record_readout_bucket_map",
    }.intersection(paths):
        raise ValueError("semantic release lacks maps or reviewed weights")
    return paths


def _coverage(paths: dict[str, Path]) -> dict:
    mapping = pd.read_parquet(
        paths["semantic_map"], columns=["level", "atom_id", "semantic_bucket_id"]
    )
    if (
        mapping.duplicated(["level", "atom_id"]).any()
        or mapping.semantic_bucket_id.isna().any()
    ):
        raise ValueError("semantic map repeats or omits a level/atom assignment")
    records = pd.read_parquet(
        paths.get("record_semantic_bucket_map", paths.get("record_readout_bucket_map"))
    )
    if (
        records.duplicated(["source_row_uid", "level"]).any()
        or records.semantic_bucket_id.isna().any()
    ):
        raise ValueError("record map repeats or omits physical UID assignments")
    weights = pd.read_parquet(
        paths["semantic_bucket_weights"],
        columns=["level", "semantic_bucket_id", "weight"],
    )
    if (
        weights.duplicated(["level", "semantic_bucket_id"]).any()
        or not weights.weight.between(0, 1).all()
    ):
        raise ValueError("semantic weights repeat families or contain invalid values")
    map_keys = set(zip(mapping.level, mapping.semantic_bucket_id))
    if not set(zip(weights.level, weights.semantic_bucket_id)) <= map_keys:
        raise ValueError("weights name foreign semantic families")
    if not set(zip(records.level, records.semantic_bucket_id)) <= map_keys:
        raise ValueError("record map names foreign semantic families")
    return {
        "atoms": len(mapping),
        "physical_records": len(records),
        "weighted_buckets": len(weights),
        "semantic_buckets": len(map_keys),
    }


def build_manifest(task: str, root: Path, selection_manifest: Path) -> dict:
    """Freeze an explicit reviewed selection descriptor in a fresh staged root."""
    if (root / "manifest.json").exists():
        raise FileExistsError("release manifest exists; prepare a fresh staging root")
    descriptor = json.loads(selection_manifest.read_text())
    task = normalize_task(task)
    if (
        descriptor.get("task") != task
        or descriptor.get("status") != "complete_reviewed"
    ):
        raise ValueError("publication requires an explicitly reviewed task selection")
    paths = _selected_files(root, descriptor)
    manifest = {
        **descriptor,
        "coverage": _coverage(paths),
        "selected_sha256": {role: sha256_file(path) for role, path in paths.items()},
        "selection_descriptor_sha256": sha256_file(selection_manifest),
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    paths = _selected_files(root, manifest)
    coverage = _coverage(paths)
    for role, expected in manifest.get("selected_sha256", {}).items():
        if sha256_file(paths[role]) != expected:
            raise ValueError(f"selected artifact changed: {role}")
    return {
        "task": manifest["task"],
        "evidence_release": manifest["evidence_library_version"],
        "coverage": coverage,
    }


def select(root: Path, approved_manifest_sha256: str) -> dict:
    """Install a verified reviewed successor; never modify evidence CURRENT."""
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != approved_manifest_sha256:
        raise ValueError("approved publication manifest changed")
    verify_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete_reviewed" or not manifest.get(
        "selected_sha256"
    ):
        raise ValueError("only a newly frozen reviewed selection can be installed")
    release = manifest["evidence_library_version"]
    if (
        not isinstance(release, str)
        or release in {"", ".", ".."}
        or Path(release).name != release
    ):
        raise ValueError("evidence release must be a single named version")
    target = (
        RELEASE_ROOT
        / normalize_task(manifest["task"])
        / manifest["evidence_library_version"]
    )
    if not (
        REPOSITORY_ROOT
        / "data/evidence_libraries"
        / manifest["task"]
        / manifest["evidence_library_version"]
    ).is_dir():
        raise ValueError("named evidence release is missing")
    files = [p for p in root.rglob("*") if p.is_file() and p != manifest_path]
    if root.resolve() == target.resolve() or any(
        ((target / p.relative_to(root)).exists() for p in files)
    ):
        raise FileExistsError(
            "publication would overwrite an existing semantic payload"
        )
    target.mkdir(parents=True, exist_ok=True)
    previous = target / "manifest.json"
    if previous.exists():
        receipt = target / "selection_receipts" / f"{sha256_file(previous)}.json"
        if receipt.exists() and receipt.read_bytes() != previous.read_bytes():
            raise ValueError("previous selection receipt differs")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if not receipt.exists():
            receipt.write_bytes(previous.read_bytes())
    for path in files:
        destination = target / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for role, path in _selected_files(target, manifest).items():
        if sha256_file(path) != manifest["selected_sha256"][role]:
            raise ValueError(f"installed semantic payload differs: {role}")
    write_json_atomic(previous, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--task", required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--selection-manifest", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    publish = commands.add_parser("select")
    publish.add_argument("--root", type=Path, required=True)
    publish.add_argument("--approved-manifest-sha256", required=True)
    changes = commands.add_parser("materialize")
    changes.add_argument("--snapshot-root", type=Path, required=True)
    changes.add_argument("--decisions", type=Path, required=True)
    changes.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest(args.task, args.root, args.selection_manifest)
    elif args.command == "verify":
        result = verify_manifest(args.root)
    elif args.command == "select":
        result = select(args.root, args.approved_manifest_sha256)
    else:
        result = materialize(args.snapshot_root, args.decisions, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
