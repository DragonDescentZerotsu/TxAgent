"""Lossless raw-to-Stage-03 adapter using the shared accelerated build runtime.

No vote policy or semantic exclusion is applied here. All acquisition rows and
permanent UIDs survive; molecular search eligibility is an independent column.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from tools.chembl_tool.common.build_runtime import worker_pool, local_workdir, publish_file, sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity, bemis_murcko_scaffold

COMMIT = "45663daaad7fd8392a0254bd78793a9e0a581a7a"
CONTRACT = "starling_lossless_source_records.v1"
_identities = None


def text(value):
    return "" if value is None else str(value).strip()


def identity(smiles):
    result = normalize_molecule_identity(smiles).to_dict()
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        organic = sum(any(a.GetAtomicNum() == 6 for a in f.GetAtoms()) for f in Chem.GetMolFrags(mol, asMols=True)) if mol else 0
        material = bool(mol and any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()))
    result["bemis_murcko_scaffold"] = bemis_murcko_scaffold(result["parent_smiles"]) if result["parent_smiles"] else ""
    result["identity_review_reason"] = (
        "invalid_structure_or_parent" if not result["canonical_smiles"] or not result["parent_inchi_key"]
        else "multiple_organic_components" if organic > 1
        else "unspecified_atom" if material else ""
    )
    result["molecule_id"] = "STARLING_" + hashlib.sha256(result["canonical_smiles"].encode()).hexdigest()[:16].upper()
    return smiles, result


def restore(task, cache):
    root = Path("data/starling_data") / task / "raw_v1"
    prefix = f"data/raw/starling/{task}/"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT, "--", prefix], text=True).splitlines()
    paths = [p for p in paths if "/compressed/" in p or p.endswith("extraction_guidance.json")]
    for path in paths:
        rel = path.removeprefix(prefix)
        local = cache / task / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            with local.open("wb") as handle:
                subprocess.run(["git", "show", f"{COMMIT}:{path}"], stdout=handle, check=True)
    manifest = json.loads((cache / task / "compressed/manifest.json").read_text())
    sources = []
    for source in manifest["sources"]:
        offset = 0
        for part in source["parts"]:
            local = cache / task / "compressed" / part["name"]
            if sha256_file(local) != part["sha256"]:
                raise ValueError(f"Raw source hash mismatch: {local}")
            rows = pq.ParquetFile(local).metadata.num_rows
            sources.append((source["source_id"], local, offset, rows))
            offset += rows
        if offset != source["rows"]:
            raise ValueError(f"Raw row count mismatch: {source['source_id']}")
    for path in paths:
        rel = path.removeprefix(prefix)
        publish_file(cache / task / rel, root / rel)
    write_json_atomic(root / "manifest.json", {
        "source_commit": COMMIT, "source_branch": "joseph", "original_root": prefix,
        "format": "ordinary_lossless_parquet_parts", "uid_policy": "preserve_source_row_uid_verbatim",
        "sources": manifest["sources"],
        "files": {p.removeprefix(prefix): {"sha256": sha256_file(cache / task / p.removeprefix(prefix)), "git_path": p} for p in paths},
    })
    return sources


def _joined(raw, fields):
    return " | ".join(f"{key}={text(raw.get(key))}" for key in fields if text(raw.get(key)))


def _convert(job):
    task, source, path, offset, row_group, target = job
    policy = importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_source")
    raw_table = pq.ParquetFile(path).read_row_group(row_group)
    columns = []
    stats = Counter()
    for index, raw in enumerate(raw_table.to_pylist()):
        smiles = text(raw.get("SMILES"))
        molecule = _identities[smiles]
        uid = raw.get("source_row_uid")
        if not uid:
            raise ValueError("Missing permanent source_row_uid")
        fields = policy.FIELDS[source.rsplit("_", 1)[1]]
        mapped = {key: _joined(raw, names) for key, names in fields.items()}
        used = {x for names in fields.values() for x in names}
        excluded = used | {"SMILES", "source_row_uid", "pmid", "paragraph_idx", "extraction_id", "support_text", "confidence", "molecule_name", "agent_name", "entity_name"}
        # All unmapped study fields remain in the model-visible context card.
        context = _joined(raw, [k for k in raw if k not in excluded])
        reason = molecule["identity_review_reason"]
        row = {
            "source_id": source, "source_name": f"starling/{task}/{source}",
            "source_row_number": offset + index, "source_row_uid": uid,
            "source_record_id": uid, "canonical_record_id": uid,
            "source_smiles": smiles, "canonical_smiles": molecule["canonical_smiles"],
            "molecule_identity_key": molecule["parent_inchi_key"],
            "molecule_id": molecule["molecule_id"], "parent_smiles": molecule["parent_smiles"],
            "bemis_murcko_scaffold": molecule["bemis_murcko_scaffold"],
            "molecule_name": next((text(raw.get(k)) for k in ("molecule_name", "agent_name", "entity_name") if text(raw.get(k))), ""),
            **mapped, "qualifying_conditions": context,
            "support_text": text(raw.get("support_text")), "confidence": raw.get("confidence"),
            "pmid": text(raw.get("pmid")), "extraction_id": text(raw.get("extraction_id")),
            "paragraph_idx": text(raw.get("paragraph_idx")),
            "group_id": f"Source.{source}", "retrieval_eligible": not reason,
            "identity_review_reason": reason, "identity_verification": "source_structure_only",
            "is_gold_voter": False, "record_contract_version": CONTRACT,
            "raw_record_json": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        }
        columns.append(row)
        stats[reason or "retrieval_eligible"] += 1
    pq.write_table(pa.Table.from_pylist(columns), target, compression="zstd", compression_level=3)
    return str(target), len(columns), dict(stats)


def build(tasks, workers=128, cache=Path("/local/tmp/txagent-new-starling")):
    started = time.monotonic()
    sources = {task: restore(task, cache) for task in tasks}
    smiles = set()
    for task_sources in sources.values():
        for _, path, _, _ in task_sources:
            smiles.update(text(s) for s in pc.unique(pq.read_table(path, columns=["SMILES"])["SMILES"]).to_pylist())
    print(f"Identity normalization: {len(smiles):,} distinct SMILES across {sum(x[3] for ss in sources.values() for x in ss):,} rows; workers={workers}", flush=True)
    global _identities
    with worker_pool(workers) as pool:
        _identities = dict(pool.map(identity, sorted(smiles), chunksize=64))
    print(f"Identity normalization finished at {time.monotonic()-started:.1f}s", flush=True)
    for task, task_sources in sources.items():
        root = Path("data/starling_data") / task / "canonical_v1"
        with local_workdir() as staging:
            pq.write_table(pa.Table.from_pylist(list(_identities.values())), staging / "molecule_identities.parquet", compression="zstd")
            identity_hash = publish_file(staging / "molecule_identities.parquet", root / "molecule_identities.parquet")
            jobs = []
            for source, path, offset, _ in task_sources:
                pf = pq.ParquetFile(path)
                for rg in range(pf.num_row_groups):
                    jobs.append((task, source, path, offset, rg, staging / f"shard-{len(jobs):06d}.parquet"))
                    offset += pf.metadata.row_group(rg).num_rows
            stats = Counter()
            count = 0
            writer = None
            try:
                with worker_pool(workers) as pool:
                    for part, n, stat in pool.map(_convert, jobs, chunksize=1):
                        table = pq.read_table(part)
                        if writer is None:
                            writer = pq.ParquetWriter(staging / "records.parquet", table.schema, compression="zstd", compression_level=3)
                        writer.write_table(table)
                        count += n
                        stats.update(stat)
                        Path(part).unlink()
                        if count % 250000 < n:
                            print(f"{task}: {count:,} records converted ({time.monotonic()-started:.1f}s)", flush=True)
            finally:
                if writer:
                    writer.close()
            if count != sum(s[3] for s in task_sources):
                raise ValueError("Source conservation failed")
            uids = pq.read_table(staging / "records.parquet", columns=["source_row_uid"])["source_row_uid"]
            if uids.null_count or len(pc.unique(uids)) != count:
                raise ValueError("UID uniqueness failed")
            records_hash = publish_file(staging / "records.parquet", root / "records.parquet")
        manifest = {
            "contract": CONTRACT, "source_commit": COMMIT, "n_source_rows": count,
            "n_records": count, "n_deleted_records": 0, "identity_status_counts": dict(stats),
            "n_retrieval_eligible": stats["retrieval_eligible"], "n_gold_voters": 0,
            "gold_status": "separate_vote_review_required", "family_status": "source_groups_only_pending_record_level_mapping",
            "raw_manifest_sha256": sha256_file(Path("data/starling_data") / task / "raw_v1/manifest.json"),
            "files": {"records.parquet": records_hash, "molecule_identities.parquet": identity_hash},
            "builder_sha256": sha256_file(Path(__file__)),
            "adapter_sha256": sha256_file(Path(importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_source").__file__)),
            "workers": workers, "elapsed_seconds": round(time.monotonic()-started, 2),
            "pyarrow_version": pa.__version__, "rdkit_version": rdBase.rdkitVersion,
        }
        write_json_atomic(root / "manifest.json", manifest)
        print(json.dumps({"task": task, **manifest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("dili", "carcinogens"), default=["dili", "carcinogens"])
    parser.add_argument("--workers", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        parser.error("workers must fit available CPUs")
    build(args.tasks, args.workers)
