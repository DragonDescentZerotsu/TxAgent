"""Read-only Ames retrieval smoke; write the requested receipt only on success."""

import gc
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import multiprocessing
import os
import time
import hashlib
import importlib
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from tools.chembl_tool.common.assay_retrieval import (
    build_family_molecule_prefix_view,
    retrieve_family_molecule_prefixes,
)
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.common.molecule_identity import (
    normalize_molecule_identity,
    bemis_murcko_scaffold,
)
from tools.chembl_tool.common.progressive_assay_reasoning import (
    extract_cumulative_evidence,
    select_initial_evidence,
    select_progressive_delta,
    append_evidence,
)
from tools.chembl_tool.common.retrieval_policy import decide_candidate
from tools.chembl_tool.common.reasoning_payload import external_condition_sentence
from tools.chembl_tool.common.starling.assay_catalog import assay_id, assay_unit
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    NO_REPORTED_CONDITION,
    task_root,
)
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles
from tools.chembl_tool.tasks.ames.build_dataset import (
    MANIFEST,
    RECORDS,
    VOTES,
    CATALOG_ROOT,
    INDEX_ROOT,
)
from tools.chembl_tool.tasks.ames.experiment_config import STARLING
from tools.chembl_tool.common.build_runtime import local_input, worker_pool

LEVELS = [1, 2, 3, 4, 5]
FAMILIES = [spec.family_key for spec in STARLING.mechanism_groups]
GROUPS = {
    group: (level, spec.family_key)
    for level, spec in enumerate(STARLING.mechanism_groups, 1)
    for group in spec.source_group_ids
}
FIELDS = {
    "endpoint_type": "canonical_endpoint_name",
    "reported_value": "canonical_measurement_text",
    "reported_units": "canonical_unit_text",
    "assay_context": "canonical_assay_context",
    "species_context": "canonical_species_context",
    "qualifying_conditions": "qualifying_conditions",
    "support_text": "support_text",
}
# Independent semantic guard: do not import the production scope classifier.
INDEPENDENT_BACTERIAL_PATTERNS = {
    "bacterial_mutation_cooccurrence": re.compile(
        r"\A(?=[\s\S]*\bbacteri(?:al|a|um)\b)(?=[\s\S]*\b(?:mutat|mutagen|revertant))",
        re.I,
    ),
    "ames_or_reverse_mutation": re.compile(
        r"(?<![a-z])ames(?![a-z])|\brevers(?:e|ion)[\s_-]+mutat|\brevertant", re.I
    ),
    "ecoli_reversion": re.compile(
        r"\A(?=[\s\S]*(?:\bescherichia\b|\be\.?\s*coli\b))(?=[\s\S]*\b(?:reverse[\s_-]+mutat|reversions?\b|revertant))",
        re.I,
    ),
    "salmonella_or_bacterial_mutation": re.compile(
        r"salmonella|\bbacteri(?:al|a|um)\b.{0,30}\b(?:mutat|mutagen)", re.I
    ),
    "wp2": re.compile(r"\bWP2", re.I),
    "ta_strain_including_suffix": re.compile(r"\bTA[- ]?\d{2,4}[a-z0-9]*\b", re.I),
    "abbreviated_s_typhi_or_typhimurium": re.compile(
        r"\bs\.?\s*typhi(?:murium)?\b", re.I
    ),
}


def bacterial_mentions(values):
    text = " | ".join(clean(v) for v in values).replace("_", " ")
    return tuple(
        name
        for name, pattern in INDEPENDENT_BACTERIAL_PATTERNS.items()
        if pattern.search(text)
    )


def placement_exceptions():
    """Validate authored exceptions independently against immutable raw payloads.

    The broad lexical guard remains active for every unreviewed record. Neither
    a group edit nor a production-classifier decision can suppress it.
    """
    path = MANIFEST.parent / "source_review_v6" / "placement_decisions.jsonl"
    decisions = read_jsonl(path)
    meta = json.loads(RECORDS.with_name("manifest.json").read_text())
    assert track(path) == meta["placement_reviews_sha256"]
    assert len(decisions) == meta["n_applied_placement_reviews"]
    samples, expected = {}, {}
    review_origins = {"source_review_v5", "source_review_v6"} | {
        row["review_origin"] for row in decisions
    }
    for origin in sorted(review_origins):
        assert Path(origin).name == origin, "Review origin must name a local audit directory"
        root = MANIFEST.parent / origin
        for name in ("sample.jsonl", "audit_annotations.jsonl"):
            track(root / name)
        for r in read_jsonl(root / "sample.jsonl"):
            assert r["source_record_id"] not in samples
            samples[r["source_record_id"]] = r
        for r in read_jsonl(root / "audit_annotations.jsonl"):
            rid = r["source_record_id"]
            assert rid in samples and rid not in expected
            expected[rid] = (
                (
                    r["proposed_groups"][0]
                    if r["disposition"] == "move"
                    else r["group_id"]
                )
                if origin == "source_review_v5"
                else r["group_id"]
            )
    assert samples.keys() == expected.keys()
    by_id = {r["source_record_id"]: r for r in decisions}
    assert len(by_id) == len(decisions) and by_id.keys() <= samples.keys()
    result = {}
    for source in ("ames_base", "ames_v1", "ames_v2", "ames_v3"):
        selected = [
            r for r in samples.values() if r["source_record_id"].split(":")[0] == source
        ]
        if not selected:
            continue
        path = MANIFEST.parent / "raw_v1" / f"{source}.parquet"
        track(path)
        ordinals = [int(r["source_record_id"].split(":")[1]) for r in selected]
        for sample, raw in zip(
            selected, pq.read_table(local_input(path)).take(ordinals).to_pylist()
        ):
            assert raw == sample["raw"], sample["source_record_id"]
            r = by_id.get(sample["source_record_id"])
            if r is None:
                continue
            assert (
                hashlib.sha256(
                    json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                == r["raw_sha256"]
            )
            assert r["rationale"] and r["canonical_surface_sha256"]
            assert r["source_record_id"] not in result
            assert r["group_id"] in GROUPS and GROUPS[r["group_id"]][0] > 1
            assert r["group_id"] == expected[r["source_record_id"]]
            result[r["source_record_id"]] = r
    assert len(result) == len(decisions)
    return result, expected


tracked = {}
tracked_stamps = {}


def track(path):
    path = Path(path)
    key = str(path.resolve())
    info = path.stat()
    before = (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    prior = tracked_stamps.get(key)
    if prior and prior[0] == before and time.monotonic() - prior[1] >= 1.0:
        return tracked[key]
    # Independent initial content scan; never accept a producer's digest cache.
    digest = sha256_file(path)
    info = path.stat()
    after = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    assert after == before, ("input changed while hashing", key)
    assert key not in tracked or tracked[key] == digest, ("input changed", key)
    tracked[key] = digest
    tracked_stamps[key] = before, (
        prior[1] if prior and prior[0] == before else time.monotonic()
    )
    return digest


def clean(value):
    return "" if value is None else str(value).strip()


def signature(smiles, family, values):
    return hashlib.sha256(
        json.dumps([smiles, family, *values], ensure_ascii=False).encode()
    ).hexdigest()


@lru_cache(None)
def identity(smiles):
    result = normalize_molecule_identity(smiles)
    assert result.status == "ok", smiles
    return result


@lru_cache(None)
def scaffold(smiles):
    return bemis_murcko_scaffold(identity(smiles).parent_smiles)


def choose_queries(rows):
    remaining = list(enumerate(rows))
    selected, labels, activations, conditions, parents = [], set(), set(), set(), set()

    def activation(row):
        return row["condition_group"].split("+", 1)[0]

    while remaining and len(selected) < 3:
        eligible = [
            (i, r) for i, r in remaining if r["molecule_identity_key"] not in parents
        ]
        assert eligible
        i, row = min(
            eligible,
            key=lambda x: (
                x[1]["Y"] in labels,
                activation(x[1]) in activations,
                x[1]["condition_group"] in conditions,
                hashlib.sha256(x[1]["benchmark_row_id"].encode()).hexdigest(),
            ),
        )
        selected.append((i, row))
        labels.add(row["Y"])
        conditions.add(row["condition_group"])
        parents.add(row["molecule_identity_key"])
        activations.add(activation(row))
        remaining = [(j, r) for j, r in remaining if j != i]
    assert labels == {0, 1}
    assert len(activations) == 3
    return selected


def validate_benchmark_votes(splits, votes):
    """Independently replay agreement/group gates from the frozen study votes."""
    units = defaultdict(list)
    for vote in votes:
        units[vote["molecule_identity_key"], vote["condition_group"]].append(vote)
    eligible = {}
    group_support = defaultdict(list)
    for key, sources in units.items():
        counts = Counter(r["Y"] for r in sources)
        threshold = 0.7 if key[1] == NO_REPORTED_CONDITION else 0.6
        agreement = max(counts.values()) / len(sources)
        if counts[0] == counts[1] or agreement < threshold:
            continue
        eligible[key] = (sources, counts, threshold, agreement)
        group_support[key[1]].append(sources[0])
    allowed = {
        group
        for group, rows in group_support.items()
        if len({r["molecule_identity_key"] for r in rows}) >= 3
        and len(
            {r["bemis_murcko_scaffold"] for r in rows if r["bemis_murcko_scaffold"]}
        )
        >= 3
    }
    expected = {key for key in eligible if key[1] in allowed}
    rows = [row for split in splits.values() for row in split]
    assert len(rows) == len(expected)
    assert {
        (r["molecule_identity_key"], r["condition_group"]) for r in rows
    } == expected
    for row in rows:
        sources, counts, threshold, agreement = eligible[
            row["molecule_identity_key"], row["condition_group"]
        ]
        assert row["Y"] == int(counts[1] > counts[0])
        assert row["agreement_threshold"] == threshold
        assert row["agreement_fraction"] == agreement
        assert row["label_counts"] == {"0": counts[0], "1": counts[1]}
        assert row["source_record_count"] == len(sources)
        assert row["source_votes"] == sorted(
            sources, key=lambda r: r["source_record_id"]
        )
    return {"rows": len(rows), "conditions": len(allowed), "source_votes": len(votes)}


_validation_rows = None


def _source_features(bounds):
    start, stop = bounds
    result = []
    for row in _validation_rows[start:stop]:
        values = [clean(row[f]) for f in FIELDS.values()]
        result.append(
            (
                bacterial_mentions(values),
                signature(row["molecule_id"], GROUPS[row["group_id"]][1], values),
                assay_id(
                    "ames",
                    assay_unit(
                        row["canonical_assay_context"], row["canonical_endpoint_name"]
                    )[0],
                ),
            )
        )
    return result


def main():
    dataset = json.loads(MANIFEST.read_text())
    assert (
        dataset["phase"] in {"all", "retrieval", "refresh-retrieval"}
        and "retrieval" in dataset
    )
    assert dataset["model_evaluation_performed"] is False
    track(Path(__file__))
    track(Path(importlib.import_module(local_input.__module__).__file__))
    track(MANIFEST)
    review_root = MANIFEST.parent / "source_review_v4"
    review = json.loads((review_root / "summary.json").read_text())
    track(review_root / "summary.json")
    for name, digest in review["artifacts"].items():
        assert track(review_root / name) == digest
    source_manifest = RECORDS.with_name("manifest.json")
    track(source_manifest)
    assert track(VOTES) == review["source_votes_sha256_after"]
    placements, reviewed_memberships = placement_exceptions()
    applied = read_jsonl(review_root / "applied_decisions.jsonl")
    assert (
        track(review_root / "applied_decisions.jsonl")
        == json.loads(source_manifest.read_text())["semantic_reviews_sha256"]
    )
    current_audit = {
        r["source_record_id"]: r
        for r in pq.read_table(
            local_input(RECORDS.with_name("record_audit.parquet"))
        ).to_pylist()
    }
    for action in applied:
        row = current_audit[action["source_record_id"]]
        if action["action"] == "exclude":
            assert not row["group_id"] and not row["is_voter"]
        elif action["action"] == "withhold":
            assert not row["is_voter"]
        elif action["action"] == "correct_structure":
            assert not row["is_voter"]
            assert row["parent_inchi_key"] == action["corrected_parent_inchi_key"]
            assert row["identity_verification"] == "payload_pinned_structure_correction"
    del current_audit
    preservation = {
        "source_votes_sha256": track(VOTES),
        "source_votes_unchanged_since_retrieval_revision": True,
        "placement_revision": "source_review_v6",
        "n_payload_pinned_placement_decisions": len(placements),
        "individually_reviewed_payloads_and_memberships": len(reviewed_memberships),
        "semantic_review_revision": str(review_root / "summary.json"),
        "source_votes_sha256_before": review["source_vote_sha256_before"],
        "current_benchmark_validation": "current dataset hashes and independent vote/group-gate replay",
    }
    for name in (
        "assay_retrieval",
        "progressive_assay_reasoning",
        "retrieval_policy",
        "molecule_identity",
        "reasoning_payload",
        "task_workflows.retrieve_neighbors",
        "task_workflows.evidence_library",
    ):
        track(
            Path(importlib.import_module("tools.chembl_tool.common." + name).__file__)
        )
    artifacts = (
        dataset["source"]
        + dataset["benchmark"]
        + dataset.get("retrieval", {}).get("catalog", [])
    )
    for items in dataset.get("retrieval", {}).get("indices", {}).values():
        artifacts += items
    for item in artifacts:
        assert track(item["path"]) == item["sha256"], item["path"]
    catalog_path = CATALOG_ROOT / "family_assays.jsonl"
    catalog = read_jsonl(catalog_path)
    catalog_meta = json.loads((CATALOG_ROOT / "manifest.json").read_text())
    track(CATALOG_ROOT / "manifest.json")
    track(
        Path(
            importlib.import_module(
                "tools.chembl_tool.tasks.ames.experiment_config"
            ).__file__
        )
    )
    assert catalog_meta["records_sha256"] == track(RECORDS)
    assert catalog_meta["catalog_sha256"] == track(catalog_path)
    assert [r["endpoint_group"] for r in catalog_meta["levels"]] == FAMILIES
    allowed_assays = {r["assay_id"] for r in catalog}
    columns = [
        "canonical_record_id",
        "source_record_id",
        "canonical_smiles",
        "molecule_id",
        "molecule_identity_key",
        "group_id",
        "heldout_filter_scope",
        "retrieval_eligible",
        *FIELDS.values(),
    ]
    raw = pq.read_table(RECORDS, columns=list(dict.fromkeys(columns))).to_pylist()
    source_votes = read_jsonl(VOTES)
    voters = {r["source_record_id"] for r in source_votes}
    assert voters == {
        r["source_record_id"] for r in raw if GROUPS[r["group_id"]][0] == 1
    }
    (
        source_signatures,
        canonical_lookup,
        source_molecules,
        source_rows,
        family_counts,
    ) = (
        {},
        {},
        {},
        [],
        Counter(),
    )
    workers = max(
        1,
        min(
            128,
            (
                len(os.sched_getaffinity(0)) // 2
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 2) // 2
            ),
        ),
    )
    global _validation_rows
    _validation_rows = raw
    was_frozen = bool(gc.get_freeze_count())
    if not was_frozen:
        gc.freeze()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    )
    features = chain.from_iterable(
        pool.map(
            _source_features,
            [
                (start, min(start + 2048, len(raw)))
                for start in range(0, len(raw), 2048)
            ],
        )
    )
    surface_mentions = {}
    reviewed_canonical_ids = set()
    seen_placements = set()
    seen_reviewed, lexical_exemptions = set(), 0
    try:
        for row, (raw_mentions, sig, aid) in zip(raw, features, strict=True):
            if not row["retrieval_eligible"]:
                continue
            level, family = GROUPS[row["group_id"]]
            bacterial_scope = row["heldout_filter_scope"] == "bacterial_outcome"
            independent_context = raw_mentions
            assert surface_mentions.setdefault(sig, raw_mentions) == raw_mentions
            if row["source_record_id"] in reviewed_memberships:
                assert row["group_id"] == reviewed_memberships[row["source_record_id"]]
                seen_reviewed.add(row["source_record_id"])
            placement = placements.get(row["source_record_id"])
            if placement:
                assert row["group_id"] == placement["group_id"]
                surface = [clean(row[f]) for f in FIELDS.values()]
                assert (
                    hashlib.sha256(
                        json.dumps(surface, ensure_ascii=False).encode()
                    ).hexdigest()
                    == placement["canonical_surface_sha256"]
                )
                seen_placements.add(row["source_record_id"])
                if level > 2:
                    lexical_exemptions += bool(independent_context)
                    independent_context = ()
                    reviewed_canonical_ids.add(row["canonical_record_id"])
            assert (level <= 2) == bacterial_scope
            assert level <= 2 or not independent_context, (
                "direct content outside L1/L2",
                row["source_record_id"],
            )
            assert (
                source_molecules.setdefault(row["molecule_id"], row["canonical_smiles"])
                == row["canonical_smiles"]
            )
            canonical_id = row["canonical_record_id"]
            assert len(canonical_id) == 64 and int(canonical_id, 16) >= 0
            assert canonical_id not in canonical_lookup
            canonical_lookup[canonical_id] = (
                row["molecule_identity_key"],
                bacterial_scope,
                level,
                independent_context,
            )
            source_signatures.setdefault(sig, set()).add(canonical_id)
            assert aid in allowed_assays
            source_rows.append(
                (
                    aid,
                    row["molecule_id"],
                    row["molecule_identity_key"],
                    level,
                    bacterial_scope,
                    independent_context,
                )
            )
            family_counts[level] += 1
    finally:
        pool.shutdown(cancel_futures=True)
        if not was_frozen:
            gc.unfreeze()
        _validation_rows = None
    assert seen_placements == placements.keys()
    assert seen_reviewed == reviewed_memberships.keys()
    preservation["reviewed_lexical_hits_outside_l1_l2"] = lexical_exemptions
    del raw
    originals = sorted(set(source_molecules.values()))
    with worker_pool(workers) as identity_pool:
        standardized_molecules = dict(
            zip(
                originals,
                identity_pool.map(standardize_smiles, originals, chunksize=128),
            )
        )
    assert set(family_counts) == set(LEVELS)
    receipt = {
        "schema_version": "ames_retrieval_validation.v2",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_evaluation_performed": False,
        "llm_calls": 0,
        "tool_service_calls": 0,
        "benchmark_scope": dataset.get("benchmark_scope"),
        "review_method": dataset["review_method"],
        "scope": "Complete indexed representative-card/source-count audit plus three valid queries per split, all five cumulative levels and deterministic 4/2 progressive selection; no inference.",
        "independent_bacterial_context_guard": {
            "patterns": {
                name: pattern.pattern
                for name, pattern in INDEPENDENT_BACTERIAL_PATTERNS.items()
            },
            "case_insensitive": True,
            "fields": list(FIELDS.values()),
            "policy": "Direct-related passages must be in L1/L2 globally and excluded for heldout parents. Independently scan source fields and indexed cards and verify complete assay-molecule source counts.",
            "production_scope_classifier_used": False,
        },
        "sample_selection": "Greedy unseen label, activation, exact condition, then SHA256 benchmark_row_id; three unique parents per split cover both labels and all three activation states; no prediction-based selection.",
        "source_trace_method": "Actual hashed canonical_record_id lookup after exact source molecule_id + semantic family + seven raw card-field matching. Index molecule SMILES/InChIKeys are separately checked against the shared source-SMILES standardizer. Representative cards omit record IDs; matching canonical IDs and ambiguity are reported, never inferred from source_record_id or assay IDs.",
        "source_records_by_level": dict(family_counts),
        "actual_l1_voters": len(voters),
        "direct_related_records_above_l2": 0,
        "families": FAMILIES,
        "min_similarity": 0.3,
        "splits": {},
    }
    for scheme, policy in [
        ("scaffold", "scaffold_disjoint"),
        ("random", "parent_disjoint"),
    ]:
        print("Auditing", scheme, flush=True)
        root = task_root("ames", scheme)
        splits = {
            s: read_jsonl(root / f"{s}_molecule_condition_labels.jsonl")
            for s in ("train", "valid", "test")
        }
        vote_validation = validate_benchmark_votes(splits, source_votes)
        parents = {
            s: {r["molecule_identity_key"] for r in rows} for s, rows in splits.items()
        }
        scaffolds = {
            s: {r["bemis_murcko_scaffold"] for r in rows} for s, rows in splits.items()
        }
        conditions = {
            s: {r["condition_group"] for r in rows} for s, rows in splits.items()
        }
        parent_overlap, scaffold_overlap = {}, {}
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            parent_overlap[a + "__" + b] = len(parents[a] & parents[b])
            scaffold_overlap[a + "__" + b] = len(scaffolds[a] & scaffolds[b])
        assert not any(parent_overlap.values())
        if scheme == "scaffold":
            assert not any(scaffold_overlap.values())
        assert conditions["train"] == conditions["valid"] == conditions["test"]
        assert len(conditions["train"]) == vote_validation["conditions"]
        for condition in conditions["train"]:
            rendered = external_condition_sentence({"condition_group": condition})
            assert (
                "metabolic activation: " in rendered
                and " and strain panel: " in rendered
            )
            assert "+strain_panel=" not in rendered
        heldout_file = root / "heldout_molecule_condition_labels.jsonl"
        heldout = parents["valid"] | parents["test"]
        assert {r["benchmark_row_id"] for r in read_jsonl(heldout_file)} == {
            r["benchmark_row_id"] for r in splits["valid"] + splits["test"]
        }
        independent_by_pattern = {
            name: dict(
                Counter(
                    level
                    for _, _, parent, level, bacterial, mentions in source_rows
                    if parent in heldout and name in mentions
                )
            )
            for name in INDEPENDENT_BACTERIAL_PATTERNS
        }
        independent_heldout = Counter(
            level
            for _, _, parent, level, bacterial, mentions in source_rows
            if parent in heldout and mentions
        )
        independent_missing_scope = Counter(
            level
            for _, _, parent, level, bacterial, mentions in source_rows
            if parent in heldout and mentions and not bacterial
        )
        assert not independent_missing_scope, (
            "independent bacterial context guard: unscoped heldout source records",
            scheme,
            dict(independent_missing_scope),
        )
        expected_counts = Counter(
            (aid, mid)
            for aid, mid, parent, level, bacterial, mentions in source_rows
            if not (parent in heldout and (bacterial or mentions))
        )
        expected_excluded = Counter(
            level
            for _, _, parent, level, bacterial, mentions in source_rows
            if parent in heldout and (bacterial or mentions)
        )
        idx_path = INDEX_ROOT / scheme / "assay_neighbor_index.pkl"
        meta = json.loads(idx_path.with_name("manifest.json").read_text())
        track(idx_path.with_name("manifest.json"))
        assert meta["evidence_sha256"] == track(
            idx_path.with_name("assay_molecule_evidence.jsonl")
        )
        for key, value in {
            "records_sha256": track(RECORDS),
            "ranked_assays_sha256": track(catalog_path),
            "heldout_molecules_jsonl_sha256": track(heldout_file),
            "index_sha256": track(idx_path),
            "filter_scope_field": "heldout_filter_scope",
            "filter_scope_value": "bacterial_outcome",
            "neighbor_identity_policy_default": policy,
            "n_direct_heldout_records_after_filter": 0,
            "n_direct_heldout_records_excluded": sum(expected_excluded.values()),
        }.items():
            assert meta[key] == value, (scheme, key, meta[key], value)
        with local_input(idx_path).open("rb") as f:
            index = pickle.load(f)
        recanonicalized = 0
        for molecule in index["molecules"]:
            original = source_molecules[molecule["molecule_chembl_id"]]
            canonical, inchi = standardized_molecules[original]
            assert (canonical, inchi) == (
                molecule["canonical_smiles"],
                molecule["standard_inchi_key"],
            )
            recanonicalized += original != canonical
        indexed_counts, examples_by_level, match_cardinalities = (
            Counter(),
            Counter(),
            Counter(),
        )
        trace_examples = {}
        for mid, groups in index["evidence_by_molecule_group"].items():
            for rows in groups.values():
                for row in rows:
                    indexed_counts[row["assay_chembl_id"], mid] += row[
                        "source_record_count"
                    ]
                    for card in row["source_record_examples"]:
                        level = int(card["evidence_family_level"])
                        assert card["evidence_family"] == FAMILIES[level - 1]
                        sig = signature(
                            mid,
                            card["evidence_family"],
                            [clean(card[f]) for f in FIELDS],
                        )
                        assert sig in source_signatures, (
                            "untraceable card",
                            scheme,
                            mid,
                        )
                        matching_ids = source_signatures[sig]
                        facts = {canonical_lookup[rid] for rid in matching_ids}
                        assert len(facts) == 1
                        parent, bacterial, canonical_level, independent_context = next(
                            iter(facts)
                        )
                        assert canonical_level == level
                        assert not (parent in heldout and bacterial)
                        # sig already proves every complete card field equals its
                        # independently scanned canonical surface; reuse that scan.
                        card_mentions = surface_mentions[sig]
                        if matching_ids <= reviewed_canonical_ids:
                            card_mentions = ()
                        assert card_mentions == independent_context
                        assert not (parent in heldout and card_mentions), (
                            "independent bacterial context heldout card leak",
                            scheme,
                            sorted(matching_ids),
                        )
                        examples_by_level[level] += 1
                        match_cardinalities[len(matching_ids)] += 1
                        trace_examples.setdefault(
                            level,
                            {
                                "matching_canonical_record_ids": sorted(matching_ids)[
                                    :3
                                ],
                                "n_matching_canonical_record_ids": len(matching_ids),
                                "surface_sha256": sig,
                            },
                        )
        assert indexed_counts == expected_counts, (
            "indexed source-count mismatch",
            scheme,
        )
        assert set(examples_by_level) == set(LEVELS)
        view = build_family_molecule_prefix_view(index, levels=LEVELS)
        # Audit all cumulative pools, including cards not reached by the smoke queries.
        pool_cards = Counter()
        for groups in view["evidence_by_molecule_group"].values():
            for level in LEVELS:
                gid = view["family_molecule_prefix_view"]["group_ids"][str(level)]
                for row in groups.get(gid, []):
                    for card in row["source_record_examples"]:
                        assert 1 <= int(card["evidence_family_level"]) <= level
                        pool_cards[level] += 1
        assert dict(pool_cards) == {
            level: sum(
                n
                for family_level, n in examples_by_level.items()
                if family_level <= level
            )
            for level in LEVELS
        }
        query_receipts = []
        for qi, query in choose_queries(splits["valid"]):
            retrievals = retrieve_family_molecule_prefixes(
                query["drug"],
                view,
                levels=LEVELS,
                min_similarity=0.3,
                neighbor_identity_policy=policy,
            )
            qidentity = identity(query["drug"])
            assert qidentity.parent_inchi_key == query["molecule_identity_key"]
            active, previous, previous_ids = {}, {}, set()
            level_receipts = []
            for level in LEVELS:
                result = retrievals[level]
                assert result["status"] == "ok"
                neighbors = [n for g in result["groups"] for n in g["neighbors"]]
                visible_families = Counter()
                for neighbor in neighbors:
                    assert not decide_candidate(qidentity, neighbor, policy).excluded
                    nidentity = identity(neighbor["canonical_smiles"])
                    assert nidentity.parent_inchi_key != qidentity.parent_inchi_key
                    assert (
                        nidentity.standard_inchi_key.split("-")[0]
                        != qidentity.standard_inchi_key.split("-")[0]
                    )
                    if scheme == "scaffold":
                        assert not scaffold(query["drug"]) or scaffold(
                            query["drug"]
                        ) != scaffold(neighbor["canonical_smiles"])
                    for row in neighbor["evidence_rows"]:
                        for card in row["source_record_examples"]:
                            assert 1 <= int(card["evidence_family_level"]) <= level
                            visible_families[card["evidence_family"]] += 1
                cumulative = extract_cumulative_evidence(result)
                ids = {(a, c) for a, data in cumulative.items() for c in data["cards"]}
                assert previous_ids <= ids
                if level == 1:
                    active, _ = select_initial_evidence(cumulative, card_limit=4)
                else:
                    new, augmented, _ = select_progressive_delta(
                        previous, cumulative, active, level=level, card_limit=2
                    )
                    old_ids = {
                        (a, c) for a, data in active.items() for c in data["cards"]
                    }
                    active = append_evidence(active, new, augmented)
                    assert old_ids <= {
                        (a, c) for a, data in active.items() for c in data["cards"]
                    }
                for data in active.values():
                    for card in data["cards"].values():
                        assert FAMILIES.index(card["evidence_family"]) + 1 <= level
                level_receipts.append(
                    {
                        "level": level,
                        "neighbors": len(neighbors),
                        "visible_cards": len(ids),
                        "cards_by_family": dict(visible_families),
                        "active_molecules_4_2": len(active),
                        "active_cards_4_2": sum(
                            len(a["cards"]) for a in active.values()
                        ),
                    }
                )
                previous, previous_ids = cumulative, ids
            assert any(r["neighbors"] for r in level_receipts), (
                "vacuous query",
                scheme,
                qi,
            )
            query_receipts.append(
                {
                    "valid_index": qi,
                    "benchmark_row_id": query["benchmark_row_id"],
                    "Y": query["Y"],
                    "condition_group": query["condition_group"],
                    "rendered_condition": external_condition_sentence(query),
                    "levels": level_receipts,
                }
            )
            print(
                scheme,
                "query",
                qi,
                "neighbors",
                [r["neighbors"] for r in level_receipts],
                flush=True,
            )
        receipt["splits"][scheme] = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "identity_policy": policy,
            "heldout_parent_count": len(heldout),
            "split_rows": {split: len(rows) for split, rows in splits.items()},
            "conditions_per_split": vote_validation["conditions"],
            "condition_rendering_checked": vote_validation["conditions"],
            "independent_vote_validation": vote_validation,
            "split_parent_overlaps": parent_overlap,
            "split_scaffold_overlaps": scaffold_overlap,
            "indexed_source_records": sum(indexed_counts.values()),
            "indexed_assay_molecule_rows": len(indexed_counts),
            "index_molecule_structures_checked": len(index["molecules"]),
            "recanonicalized_smiles_strings": recanonicalized,
            "excluded_heldout_bacterial_records_by_level": dict(expected_excluded),
            "indexed_representative_cards_by_level": dict(examples_by_level),
            "canonical_trace_examples_by_level": trace_examples,
            "representative_card_source_match_cardinalities": dict(match_cardinalities),
            "cumulative_pool_card_counts": dict(pool_cards),
            "heldout_bacterial_outcome_overlap": 0,
            "future_family_visibility_violations": 0,
            "sample_neighbor_identity_overlap": 0,
            "untraceable_representative_cards": 0,
            "independent_bacterial_context_guard": {
                "heldout_source_records_by_pattern_and_level": independent_by_pattern,
                "heldout_source_records_by_level": dict(independent_heldout),
                "unscoped_heldout_source_records": sum(
                    independent_missing_scope.values()
                ),
                "retained_heldout_representative_cards": 0,
                "complete_source_count_audit_passed": True,
            },
            "queries": query_receipts,
        }
        del view, index, retrievals, cumulative, active
        gc.collect()
    for path, expected in tracked.items():
        assert track(Path(path)) == expected, (
            "input drift during validation",
            path,
        )
    receipt["input_hashes"] = tracked
    receipt["validator_sha256"] = sha256_file(Path(__file__))
    receipt["validator_path"] = str(Path(__file__).resolve())
    target = Path("data/starling_data/ames/retrieval_validation.json")
    write_json_atomic(target, receipt)
    source_meta = json.loads(source_manifest.read_text())
    write_json_atomic(
        MANIFEST.with_name("dataset_validation.json"),
        {
            "schema_version": "ames_dataset_validation.v2",
            "status": "passed",
            "dataset_manifest_sha256": sha256_file(MANIFEST),
            "source_manifest_sha256": sha256_file(source_manifest),
            "retrieval_validation_sha256": sha256_file(target),
            "source_review_sha256": sha256_file(review_root / "summary.json"),
            "placement_decisions_sha256": source_meta["placement_reviews_sha256"],
            "source_checks": {
                "raw_rows_accounted_for": source_meta["n_source_rows"],
                "retrievable_records": sum(family_counts.values()),
                "actual_l1_voters": len(voters),
                "direct_related_records_above_l2": 0,
                "gold_votes_unchanged": True,
                "review_method": review["review_method"],
            },
            "preservation": preservation,
            "model_evaluation_performed": False,
        },
    )
    print(
        "PASSED",
        target,
        "queries=" + str(3 * len(receipt["splits"])),
        "level_results=" + str(15 * len(receipt["splits"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
