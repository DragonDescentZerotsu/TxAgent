"""Select progressive evidence with explicit, manifest-backed level membership.

Read frozen canonical records, level membership and relevance rankings; construct
scaffold-disjoint Morgan pools and select molecule/record cards. The progressive
runner owns rendering and inference. No source classifier or model runs here.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import inspect
import math
from pathlib import Path
import sqlite3
import yaml

import pandas as pd
import pyarrow.parquet as pq
from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.retrieval.policies import (
    decide_candidate, normalize_molecule_identity, standardize_smiles_and_fp,
    seeded_rank_tie_key,
)
from predict.retrieval.assay_reranking.progressive_levels import (
    ProgressiveV191PromptRenderer, _record_payload,
)
from predict.utils.json import sha256_file, write_json_atomic


ASSETS = Path("data/artifacts/evidence_library_assets")
IMPORTED = Path("data/evidence_libraries/level_mappings.v1.json")
LEDGER = Path("data/raw/starling/source_row_uid_ledger")
LOCAL = ASSETS / "bbb_martins/construction_assets/progressive_level_mapping_v1/manifest.json"
RANKINGS = ASSETS / "bbb_relevance_levels_baidu_degree10_v1"
RELEVANCE_ELIGIBILITY = ASSETS / "bbb_relevance_retrieval_eligibility_v1"
CACHE = Path("predict/retrieval/cache/assay_reranking/archive/mapped_progressive")
VERSION = "mapped_progressive_sampling.v2"
RELEVANCE_FILTER_VERSION = "mapped_progressive_relevance_filter.v1"
TASKS = {"bbb_martins": ("BBB_Martins", 5), "bioavailability_ma": ("Bioavailability_Ma", 6)}


def _active_heldout_parents(task):
    """Resolve held-out parents and hashes from the active gold release."""
    parents, inputs = set(), {}
    for split in ("valid", "test"):
        path = split_path(task, split).with_name(
            f"{split}_molecule_condition_labels.jsonl"
        )
        inputs[str(path)] = sha256_file(path)
        for line in path.read_text().splitlines():
            identity = normalize_molecule_identity(json.loads(line)["drug"])
            parents.add(identity.parent_inchi_key or identity.parent_smiles)
    return parents, inputs


def load_retrieval_policy(path, task, max_level=0, *, reranking=None):
    """Resolve an explicit stage policy; absent scores never imply Morgan fallback."""
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    allowed = {'version', 'score_caches'} if reranking else {'version', 'stages', 'score_caches'}
    if not isinstance(document, dict) or set(document) - allowed:
        raise ValueError('Invalid retrieval policy fields')
    if reranking:
        if reranking not in {'joint', 'morgan', 'assay-transfer'}:
            raise ValueError(f'Invalid reranking mode: {reranking}')
        document['stages'] = {f'L{i}': 'morgan' if reranking == 'morgan' or i == 5
                              else 'assay_transfer' for i in range(1, 7)}
        if reranking == 'joint':
            document['stages']['L1'] = 'joint'
    if document.get('version') != 1 or not isinstance(document.get('stages'), dict):
        raise ValueError('Retrieval policy requires version: 1 and stages')
    if task not in TASKS:
        raise ValueError(f'Unsupported policy task: {task}')
    last = max_level or TASKS[task][1]
    if not 1 <= last <= TASKS[task][1]:
        raise ValueError(f'Unsupported final level for {task}: {last}')
    stages = document['stages']
    for level, method in stages.items():
        if level not in {f'L{i}' for i in range(1, 7)} or method not in {'morgan', 'assay_transfer', 'joint'}:
            raise ValueError(f'Invalid stage ranking: {level}={method}')
        if method == 'joint' and level != 'L1':
            raise ValueError('Joint retrieval is supported only at L1')
    required = [f'L{i}' for i in range(1, last + 1)]
    if set(required) - stages.keys():
        raise ValueError(f'Missing stage rankings: {set(required) - stages.keys()}')
    caches = document.get('score_caches') or {}
    if not isinstance(caches, dict) or set(caches) - TASKS.keys():
        raise ValueError('score_caches must map supported tasks to cache paths')
    cache = caches.get(task)
    if cache is not None and (not isinstance(cache, str) or not cache.strip()):
        raise ValueError('Score cache paths must be nonempty strings')
    return dict(stages={k: stages[k] for k in required},
                **({'harness_version': 'reranked-progressive-v1', 'reranking': reranking} if reranking else {}),
                score_cache=str((path.parent / cache).resolve()) if cache else None,
                inputs={str(path): sha256_file(path)})


def select_policy_records(by_level, pools, similarities, scores, policy, *, task,
                          query_id, molecule_limit=10, l1_limit=10, later_limit=50, tie_seed=0):
    """Select L1 molecules and independent L2+ records before any card grouping."""
    def tie(level, mid, rid):
        return seeded_rank_tie_key(tie_seed, task, query_id, level, mid, rid)

    def rows_for(level, method):
        rows = []
        for mid in pools[level]:
            for rid in by_level[level].get(mid, []):
                if method == 'assay_transfer' and rid not in scores:
                    raise ValueError(f'Missing mapping-compatible score: {query_id}/{level}/{rid}')
                row = dict(record_id=rid, reference_molecule_id=mid,
                           morgan_similarity=similarities[mid], ranking_method=method)
                if method == 'assay_transfer':
                    score = scores[rid]
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f'Invalid transfer score: {rid}')
                    row['transfer_likelihood'] = score
                rows.append(row)
        field = 'morgan_similarity' if method == 'morgan' else 'transfer_likelihood'
        return sorted(rows, key=lambda r: (-r[field], tie(level, r['reference_molecule_id'], r['record_id'])))

    mode = policy['L1']
    methods = ['morgan', 'assay_transfer'] if mode == 'joint' else [mode]
    if mode == 'joint' and molecule_limit != 10:
        raise ValueError('Joint L1 requires ten slots: five per method')
    selected, ranks = {}, {}
    for method in methods:
        rows = rows_for('L1', method)
        mids = list(dict.fromkeys(r['reference_molecule_id'] for r in rows))[:5 if mode == 'joint' else molecule_limit]
        for rank, mid in enumerate(mids, 1):
            ranks.setdefault(mid, {})[method] = rank
            # Within-molecule sampling is identical across both panels.
            chosen = sorted((r for r in rows if r['reference_molecule_id'] == mid),
                            key=lambda r: tie('L1', mid, r['record_id']))[:l1_limit]
            if mid in selected:
                assert [r['record_id'] for r in selected[mid]] == [r['record_id'] for r in chosen]
            selected[mid] = chosen
    if not selected:
        raise ValueError(f'No eligible mapped L1 molecules for {query_id}')
    molecules = []
    for mid, rows in selected.items():
        molecule = dict(reference_molecule_id=mid, selection_rank=len(molecules),
                        morgan_similarity=similarities[mid], ranking_method=mode,
                        available_l1=len(by_level['L1'][mid]), available_l2=0,
                        l1_records=rows, l2_records=[])
        if 'assay_transfer' in ranks[mid]:
            molecule['transfer_likelihood'] = max(scores[rid] for rid in by_level['L1'][mid])
        if mode == 'joint':
            molecule.update(morgan_top5_rank=ranks[mid].get('morgan', 'not_selected_in_top5'),
                            assay_transfer_top5_rank=ranks[mid].get('assay_transfer', 'not_selected_in_top5'))
        molecules.append(molecule)
    seen = {r['record_id'] for rows in selected.values() for r in rows}
    later = {}
    for level, method in policy.items():
        if level == 'L1':
            continue
        rows = [r for r in rows_for(level, method) if r['record_id'] not in seen]
        chosen = rows[:later_limit]
        seen.update(r['record_id'] for r in chosen)
        later[level] = dict(records=chosen, available_record_count=len(rows),
                            candidate_molecules=len(pools[level]), shortfall=later_limit-len(chosen),
                            ranking_method=method, allow_shortfall=True,
                            selection_policy='stage_ranked_records.v1')
    return molecules, later


def load_level_mapping(records: pd.DataFrame, manifest_path: Path, task="bbb_martins"):
    """Resolve imported acquisition membership or a canonical-record sidecar."""
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(manifest_path): sha256_file(manifest_path)}
    if records.canonical_record_id.duplicated().any():
        raise ValueError("Duplicate canonical records")
    if manifest.get("version") == "gold_original_level_mapping.v1":
        if manifest.get("task") != task:
            raise ValueError("Gold level mapping targets a different task")
        item = manifest["outputs"]["level_mapping"]
        if item["kind"] != "parquet_file":
            raise ValueError("Progressive retrieval requires a single Parquet level map")
    elif "tasks" in manifest:
        item = manifest["tasks"][task]
    else:
        item = None

    if item is not None:
        path = manifest_path.parent / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Imported level hash mismatch")
        if item.get("manifest"):
            release_manifest_path = manifest_path.parent / item["manifest"]
            if sha256_file(release_manifest_path) != item["manifest_sha256"]:
                raise ValueError("Imported level manifest hash mismatch")
            release_manifest = json.loads(release_manifest_path.read_text())
            output = release_manifest.get("output")
            if (
                release_manifest.get("version") != "evidence_library_level_mapping.v1"
                or release_manifest.get("task") != task
                or output is None
                or output["sha256"] != item["sha256"]
                or (release_manifest_path.parent / output["path"]).resolve()
                != path.resolve()
            ):
                raise ValueError("Imported level manifest disagrees with index")
            inputs[str(release_manifest_path)] = item["manifest_sha256"]
        ledger_manifest = json.loads((LEDGER / "manifest.json").read_text())
        if sha256_file(LEDGER / "manifest.json") != manifest["uid_ledger_manifest_sha256"]:
            raise ValueError("Imported levels target a different UID ledger")
        inputs[str(LEDGER / "manifest.json")] = manifest["uid_ledger_manifest_sha256"]
        for name, expected in ledger_manifest["partition_sha256"].items():
            if sha256_file(LEDGER / name) != expected:
                raise ValueError(f"UID ledger changed: {name}")
            inputs[str(LEDGER / name)] = expected
        ledger = pq.read_table(LEDGER, filters=[("task_id", "=", task)],
                               ignore_prefixes=[".", "_", "manifest"]).to_pandas()
        levels = pd.read_parquet(path)
        if levels.source_row_uid.duplicated().any():
            raise ValueError("Conflicting or duplicate UID membership")
        joined = records.merge(
            ledger[["source_id", "source_record_id", "acquisition_source_row_number", "source_row_uid"]],
            left_on=["source_id", "source_record_id", "source_row_number"],
            right_on=["source_id", "source_record_id", "acquisition_source_row_number"],
            how="left", validate="many_to_one",
        )
        if joined.source_row_uid.isna().any():
            raise ValueError("Canonical record lacks an exact acquisition UID")
        if not joined.source_row_uid.str.fullmatch(r"sr_[0-9a-f]{32}").all():
            raise ValueError("Invalid acquisition UID")
        joined = joined.merge(levels[["source_row_uid", "level", "family_key"]],
                              on="source_row_uid", how="left", validate="many_to_one")
        joined["level"] = joined.level.map(lambda x: None if pd.isna(x) else f"L{int(x)}")
    else:
        path = Path(manifest["output"]["path"])
        if sha256_file(path) != manifest["output"]["sha256"]:
            raise ValueError("Canonical level hash mismatch")
        levels = pd.read_parquet(path)
        joined = records.merge(levels[["canonical_record_id", "progressive_level"]],
                               on="canonical_record_id", how="left", validate="one_to_one")
        joined = joined.rename(columns={"progressive_level": "level"})
    inputs[str(path)] = sha256_file(path)
    if not set(joined.level.dropna()) <= {f"L{i}" for i in range(1, TASKS[task][1]+1)}:
        raise ValueError(f"Unsupported {task} level")
    return joined, inputs


def select_bucket_records(rows, *, limit, per_bucket, repeat=False):
    """Rows already carry within-bucket score order; bucket rank is primary."""
    buckets = defaultdict(list)
    seen = set()
    for row in rows:
        if row["record_id"] not in seen:
            buckets[row["node_key"]].append(row)
            seen.add(row["record_id"])
    ordered = sorted(buckets, key=lambda key: (buckets[key][0]["level_rank"], key))
    result = []
    offset = 0
    while len(result) < limit:
        batch = [row for key in ordered for row in buckets[key][offset:offset + per_bucket]]
        if not batch:
            break
        result.extend(batch[:limit - len(result)])
        if not repeat:
            break
        offset += per_bucket
    return result


def load_mapped_candidates(queries, *, mapping=None, ranking_root=RANKINGS,
                           relevance_eligibility_root=RELEVANCE_ELIGIBILITY,
                           v7_root=Path("data/evidence_libraries"), molecule_limit=10,
                           l1_limit=10, l2_limit=10, later_limit=50, tie_seed=0,
                           ranking="morgan", score_cache=None, cache_root=CACHE,
                           task="bbb_martins", sampler="relevance", retrieval_policy=None):
    """Return existing molecule-card/record-bundle shapes plus selection provenance.

Mapping and ranking hashes participate in the reusable selection-cache identity.
Assay-transfer selection only reads a supplied complete, mapping-compatible cache.
"""
    if min(molecule_limit, l1_limit, l2_limit, later_limit) < 1 or molecule_limit > 75:
        raise ValueError("Positive limits and at most 75 molecules are required")
    if sampler not in {"plain", "relevance", "relevance_filter"}:
        raise ValueError(f"Unsupported sampler: {sampler}")
    if sampler == "relevance_filter" and (task != "bbb_martins" or ranking != "morgan"):
        raise ValueError("The BBB relevance filter requires Morgan ranking")
    _, last_level = TASKS[task]
    stage_rankings = retrieval_policy['stages'] if retrieval_policy else None
    if retrieval_policy:
        if sampler != 'plain':
            raise ValueError('Stage ranking policies require plain record sampling')
        last_level = len(stage_rankings)
        score_cache = retrieval_policy['score_cache']
        if any(m in {'assay_transfer', 'joint'} for m in stage_rankings.values()) and not score_cache:
            raise ValueError(f'{task}: configure a complete mapping-compatible score_caches entry for assay-transfer/joint stages')
    default_mapping = IMPORTED
    mapping = Path(mapping) if mapping is not None else default_mapping
    source = v7_root / f"{task}/v7/03_pair_buckets/records.parquet"
    source_manifest = json.loads(source.with_name("manifest.json").read_text())
    source_hash = sha256_file(source)
    if source_manifest["outputs"]["records.parquet"] != source_hash:
        raise ValueError("Stage 3 source hash mismatch")
    columns = ["canonical_record_id", "source_id", "source_record_id", "source_row_number", "canonical_smiles"]
    inventory = pd.read_parquet(source, columns=columns)
    mapped, inputs = load_level_mapping(inventory, Path(mapping), task)
    if retrieval_policy:
        inputs.update(retrieval_policy['inputs'])
    inputs[str(source)] = source_hash
    unmapped = mapped[mapped.level.isna()].canonical_record_id.tolist()
    if Path(mapping).resolve() in {IMPORTED.resolve(), default_mapping.resolve()} and len(unmapped) != {"bbb_martins": 60, "bioavailability_ma": 117}[task]:
        raise ValueError(f"Unexpected imported mapping coverage for {task}: {len(unmapped)} unmapped")
    info = {}
    relevance_eligible = set()
    relevance_filter_counts = None
    if sampler == "relevance":
        rank_manifest_path = Path(ranking_root) / "manifest.json"
        rank_manifest = json.loads(rank_manifest_path.read_text())
        if rank_manifest["status"] != "complete" or rank_manifest["input_sha256"] != source_hash:
            raise ValueError("Relevance rankings must be complete and match Stage 3")
        rank_path = Path(ranking_root) / "relevance_bucket_rankings.parquet"
        record_map_path = Path(ranking_root) / "record_relevance_map.parquet"
        for path, expected in [(rank_path, rank_manifest["ranking_sha256"]),
                               (record_map_path, rank_manifest["record_map_sha256"])]:
            if sha256_file(path) != expected:
                raise ValueError(f"Relevance artifact hash mismatch: {path}")
            inputs[str(path)] = expected
        inputs[str(rank_manifest_path)] = sha256_file(rank_manifest_path)
        ranks = pd.read_parquet(rank_path)
        record_map = pd.read_parquet(record_map_path)
        comparison = mapped[mapped.level.isin([f"L{i}" for i in range(2, last_level+1)])].merge(
            record_map[["canonical_record_id", "level", "node_key"]],
            on="canonical_record_id", how="outer", suffixes=("", "_ranked"), validate="one_to_one")
        if comparison.level.isna().any() or comparison.level.ne(comparison.level_ranked).any():
            raise ValueError("Relevance rankings and configured level membership disagree")
        membership = comparison.merge(ranks[["node_key", "level_rank"]],
                                      on="node_key", how="left", validate="many_to_one")
        if membership.level_rank.isna().any():
            raise ValueError("Missing within-level bucket ranking")
        if len(unmapped) != rank_manifest["unmapped_level_records"]:
            raise ValueError("Unexpected unmapped record coverage")
        info = membership.set_index("canonical_record_id")[["node_key", "level_rank"]].to_dict("index")
    elif sampler == "relevance_filter":
        eligibility_root = Path(relevance_eligibility_root)
        eligibility_manifest_path = eligibility_root / "manifest.json"
        eligibility_path = eligibility_root / "record_relevance_eligibility.parquet"
        eligibility_manifest = json.loads(eligibility_manifest_path.read_text())
        expected_rule = {
            "field": "level_percentile", "operator": ">=", "value": 20.0,
            "output_field": "relevance_bucket_retrieval_eligible",
        }
        if (eligibility_manifest.get("version") != "bbb_relevance_retrieval_eligibility.v1"
                or eligibility_manifest.get("status") != "complete"
                or eligibility_manifest.get("task") != task
                or eligibility_manifest.get("stage3_input_sha256") != source_hash
                or eligibility_manifest.get("rule") != expected_rule):
            raise ValueError("BBB relevance eligibility contract mismatch")
        ranking_inputs = eligibility_manifest.get("ranking_inputs")
        if not isinstance(ranking_inputs, dict) or len(ranking_inputs) != 3:
            raise ValueError("BBB relevance eligibility lacks ranking provenance")
        for path, expected in ranking_inputs.items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"BBB relevance ranking input changed: {path}")
        if (Path(eligibility_manifest["output"]["path"]).name != eligibility_path.name
                or sha256_file(eligibility_path) != eligibility_manifest["output"]["sha256"]):
            raise ValueError("BBB relevance eligibility hash mismatch")
        flags = pd.read_parquet(eligibility_path)
        expected_columns = {"canonical_record_id", "level", "relevance_bucket_retrieval_eligible"}
        if (
            set(flags.columns) != expected_columns
            or flags.canonical_record_id.duplicated().any()
            or not pd.api.types.is_bool_dtype(flags.relevance_bucket_retrieval_eligible)
            or len(flags) != eligibility_manifest.get("records")
            or set(flags.level) != {"L2", "L3", "L4", "L5"}
        ):
            raise ValueError("Invalid BBB relevance eligibility schema")
        comparison = mapped[mapped.level.isin([f"L{i}" for i in range(2, last_level + 1)])].merge(
            flags, on="canonical_record_id", how="outer", suffixes=("", "_flagged"), validate="one_to_one"
        )
        if (comparison.level.isna().any() or comparison.level_flagged.isna().any()
                or comparison.level.ne(comparison.level_flagged).any()
                or comparison.relevance_bucket_retrieval_eligible.isna().any()):
            raise ValueError("Relevance eligibility and configured level membership disagree")
        relevance_eligible = set(comparison.loc[
            comparison.relevance_bucket_retrieval_eligible, "canonical_record_id"
        ])
        relevance_filter_counts = eligibility_manifest["counts_by_level"]
        actual_counts = {
            level: {
                "eligible": int(group.relevance_bucket_retrieval_eligible.sum()),
                "ineligible": int((~group.relevance_bucket_retrieval_eligible).sum()),
            }
            for level, group in flags.groupby("level", sort=True)
        }
        if actual_counts != relevance_filter_counts:
            raise ValueError("BBB relevance eligibility counts do not match the manifest")
        inputs[str(eligibility_manifest_path)] = sha256_file(eligibility_manifest_path)
        inputs[str(eligibility_path)] = eligibility_manifest["output"]["sha256"]
    local_path = ASSETS / f"{task}/construction_assets/progressive_level_mapping_v1/manifest.json"
    local_manifest = json.loads(local_path.read_text())
    quality_path = Path(local_manifest["output"]["path"])
    if sha256_file(quality_path) != local_manifest["output"]["sha256"]:
        raise ValueError("Quality sidecar changed")
    quality = pd.read_parquet(quality_path)
    # Membership is not eligibility: an old L1 exclusion must not survive a swap.
    excluded = quality[quality.score_cache_exclusion_reason.str.contains(
        "unresolved", case=False, na=False)].canonical_record_id.tolist()
    inputs[str(quality_path)] = sha256_file(quality_path)
    inputs[str(local_path)] = sha256_file(local_path)
    if source_hash not in local_manifest["inputs"].values():
        raise ValueError("Quality sidecar targets a different Stage 3 source")
    heldout_parents, heldout_inputs = _active_heldout_parents(task)
    inputs.update(heldout_inputs)
    score_version = None
    needs_scores = any(m in {'assay_transfer', 'joint'} for m in stage_rankings.values()) if stage_rankings else ranking == 'assay_transfer'
    if needs_scores:
        if score_cache is None:
            raise ValueError("Assay-transfer requires a complete mapping-compatible --level-score-cache")
        score_cache = Path(score_cache)
        score_version = json.loads(score_cache.with_name("VERSION.json").read_text())
        if score_version.get("level_mapping_sha256") != sha256_file(Path(mapping)):
            raise ValueError("Assay-transfer cache level mapping mismatch")
        if source_hash not in score_version.get("inputs", {}).values():
            raise ValueError("Assay-transfer cache source mismatch")
        if score_version.get("status") != "complete":
            raise ValueError("Assay-transfer cache is not complete")
        cache_hash = sha256_file(score_cache)
        if score_version.get("cache_sha256") != cache_hash or score_version.get("cache_quick_check") != "ok":
            raise ValueError("Assay-transfer cache integrity mismatch")
        inputs[str(score_cache)] = cache_hash
        inputs[str(score_cache.with_name("VERSION.json"))] = sha256_file(score_cache.with_name("VERSION.json"))
    elif ranking != "morgan":
        raise ValueError(f"Unsupported record ranking: {ranking}")
    renderer = ProgressiveV191PromptRenderer(task)
    contract_version = RELEVANCE_FILTER_VERSION if sampler == "relevance_filter" else VERSION
    contract = dict(version=contract_version, inputs=inputs, queries=queries, ranking=ranking, task=task, sampler=sampler,
                    molecule_limit=molecule_limit, l1_limit=l1_limit, l2_limit=l2_limit,
                    later_limit=later_limit, tie_seed=tie_seed, pool_size=75,
                    projection_sha256=renderer.projection_hash,
                    payload_code_sha256=sha256_file(Path(inspect.getfile(_record_payload))),
                    assembly_sha256=sha256_file(Path(__file__)))
    if retrieval_policy:
        contract.update(version='stage_ranked_records.v1', retrieval_policy=retrieval_policy)
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_root) / f"{key}.json"
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["audit"]["contract"] != contract:
            raise ValueError("Selection cache contract mismatch")
        return saved["molecules"], saved["later"], saved["audit"]
    eligible = mapped[mapped.level.isin([f'L{i}' for i in range(1, last_level+1)]) & ~mapped.canonical_record_id.isin(excluded)]
    if sampler == "relevance_filter":
        eligible = eligible[eligible.level.eq("L1") | eligible.canonical_record_id.isin(relevance_eligible)]
    molecules, fps, by_level = {}, {}, {f"L{i}": defaultdict(list) for i in range(1, last_level+1)}
    identities = {}
    heldout_l1 = 0
    for row in eligible.itertuples(index=False):
        smiles = row.canonical_smiles
        if smiles not in identities:
            identities[smiles] = normalize_molecule_identity(smiles)
        identity = identities[smiles]
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"Unresolved eligible parent: {row.canonical_record_id}")
        mid = identity.parent_inchi_key or identity.parent_smiles
        if row.level == "L1" and mid in heldout_parents:
            heldout_l1 += 1
            continue
        if mid not in molecules:
            molecules[mid] = dict(canonical_smiles=identity.parent_smiles, molecule_identity=identity.to_dict())
            _, _, fps[mid] = standardize_smiles_and_fp(identity.parent_smiles)
        by_level[row.level][mid].append(row.canonical_record_id)
    selected_molecules, selected_later = {}, {}
    molecule_ids = sorted(molecules)
    fingerprints = [fps[mid] for mid in molecule_ids]
    for query_id, smiles in queries.items():
        query_identity = normalize_molecule_identity(smiles)
        _, _, query_fp = standardize_smiles_and_fp(query_identity.parent_smiles)
        if query_fp is None:
            raise ValueError(f"Cannot fingerprint query {query_id}")
        similarities = dict(zip(molecule_ids, DataStructs.BulkTanimotoSimilarity(query_fp, fingerprints)))
        ordered = sorted(molecule_ids, key=lambda mid: (-similarities[mid], mid))
        pools = {level: [] for level in by_level}
        for mid in ordered:
            relevant = [level for level in pools if len(pools[level]) < 75 and mid in by_level[level]]
            if relevant and not decide_candidate(query_identity, molecules[mid], "scaffold_disjoint").excluded:
                for level in relevant:
                    pools[level].append(mid)
            if all(len(pool) == 75 for pool in pools.values()):
                break
        scores = {}
        if score_version is not None:
            with sqlite3.connect(f"file:{score_cache.resolve()}?mode=ro", uri=True) as connection:
                for rid, score in connection.execute(
                    "SELECT r.external_record_id, s.transfer_probability FROM assignments a "
                    "JOIN queries q USING(query_id) JOIN records r USING(record_key) "
                    "JOIN scores s USING(score_key) WHERE q.query_smiles=?", (query_identity.parent_smiles,)):
                    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid cached transfer score: {rid}")
                    if rid in scores and scores[rid] != score:
                        raise ValueError(f"Conflicting cached scores: {rid}")
                    scores[rid] = score
        if retrieval_policy:
            cards, later = select_policy_records(by_level, pools, similarities, scores, stage_rankings,
                task=task, query_id=query_id, molecule_limit=molecule_limit,
                l1_limit=l1_limit, later_limit=later_limit, tie_seed=tie_seed)
            for card in cards:
                card['canonical_smiles'] = molecules[card['reference_molecule_id']]['canonical_smiles']
            selected_molecules[query_id], selected_later[query_id] = cards, later
            continue
        score_field = "morgan_similarity" if ranking == "morgan" else "transfer_likelihood"

        def rows_for(level, mids):
            rows = []
            for mid in mids:
                for rid in by_level[level].get(mid, []):
                    if ranking == "assay_transfer" and rid not in scores:
                        raise ValueError(f"Missing mapping-compatible score: {query_id}/{rid}")
                    rows.append(dict(record_id=rid, reference_molecule_id=mid,
                                     **{score_field: similarities[mid] if ranking == "morgan" else float(scores[rid])},
                                     **info.get(rid, {})))
            return sorted(rows, key=lambda r: (-r[score_field], seeded_rank_tie_key(
                tie_seed, task, query_id, level, r["reference_molecule_id"], r["record_id"])))

        l1_rows = rows_for("L1", pools["L1"])
        mids = list(dict.fromkeys(r["reference_molecule_id"] for r in l1_rows))[:molecule_limit]
        if not mids:
            raise ValueError(f"No eligible imported L1 molecules for {query_id}")
        cards = []
        for mid in mids:
            l1 = [r for r in l1_rows if r["reference_molecule_id"] == mid]
            l2 = rows_for("L2", [mid])
            cards.append(dict(reference_molecule_id=mid, canonical_smiles=molecules[mid]["canonical_smiles"],
                              **{score_field: l1[0][score_field]}, selection_rank=len(cards),
                              available_l1=len(l1), available_l2=len(l2), l1_records=l1[:l1_limit],
                              l2_records=(select_bucket_records(l2, limit=l2_limit, per_bucket=2, repeat=True)
                                          if sampler == "relevance" else l2[:l2_limit])))
        selected_molecules[query_id] = cards
        seen = {r["record_id"] for card in cards for level in ("l1_records", "l2_records") for r in card[level]}
        selected_later[query_id] = {}
        for level in (f"L{i}" for i in range(3, last_level+1)):
            rows = [r for r in rows_for(level, pools[level]) if r["record_id"] not in seen]
            chosen = (select_bucket_records(rows, limit=later_limit, per_bucket=5)
                      if sampler == "relevance" else rows[:later_limit])
            seen.update(r["record_id"] for r in chosen)
            selected_later[query_id][level] = dict(records=chosen, available_record_count=len(rows),
                allow_shortfall=True, selection_policy=contract_version, candidate_molecules=len(pools[level]),
                shortfall=later_limit-len(chosen))
    selected = [r for cards in selected_molecules.values() for card in cards
                for level in ("l1_records", "l2_records") for r in card[level]]
    selected += [r for levels in selected_later.values() for selection in levels.values() for r in selection["records"]]
    wanted = {r["record_id"] for r in selected}
    level_by_id = mapped.set_index("canonical_record_id").level.to_dict()
    payloads = {}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=10000):
        ids = batch.column(batch.schema.get_field_index("canonical_record_id")).to_pylist()
        for row in batch.filter([rid in wanted for rid in ids]).to_pylist():
            rid = row["canonical_record_id"]
            identity = identities[row["canonical_smiles"]]
            payload = _record_payload(renderer, row, level_by_id[rid], identity.parent_smiles, task)
            payload.update({k: v for k, v in row.items() if k.startswith("canonical_") and k != "canonical_smiles"})
            payloads[rid] = payload
    if wanted != set(payloads):
        raise ValueError("Selected records missing canonical payloads")
    for row in selected:
        row["payload"] = payloads[row["record_id"]]
    audit = dict(contract=contract, inputs=inputs, ranking=stage_rankings or ranking, cache=str(cache_path),
                 unmapped_record_ids=unmapped, quality_excluded_records=len(excluded),
                 heldout_l1_excluded_records=heldout_l1,
                 mapped_level_counts=mapped.level.value_counts().to_dict(),
                 neighbor_identity_policy="scaffold_disjoint", selection_policy=contract['version'])
    if relevance_filter_counts is not None:
        audit["relevance_filter_counts_by_level"] = relevance_filter_counts
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_path, dict(molecules=selected_molecules, later=selected_later, audit=audit))
    return selected_molecules, selected_later, audit
