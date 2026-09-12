"""Build five-group Starling-only Carcinogens gold from the completed ledger.

No LLM calls and no old direction-eligibility replay. Shared builders own
consensus and split allocation; this adapter owns conditions and source votes.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    sha256_file,
)
from tools.chembl_tool.common.starling.source_gold_review import payload_hash
from tools.chembl_tool.tasks.carcinogens.reviewed_conditions import (
    classify_condition,
    GROUPS,
    VERSION as CONDITION_VERSION,
)

ROOT = Path("data/starling_data/carcinogens/gold_v4")
SOURCE = Path("data/starling_data/carcinogens")
RAW = SOURCE / "raw_v1/compressed/carcinogens_base.parquet"
LABELS = Path(
    "data/starling_data/new_tasks_gold_audit/targeted_review_v2/carcinogens_source_labels.parquet"
)
VERSION = "carcinogens_reviewed_starling_only_five_groups.v1"
VOTE_UNIT = "reporting_publication_parent_organism_with_exact_passage_deduplication"


def normalize_identity(smiles):
    from tools.chembl_tool.common.starling.new_task_identity import (
        normalize_new_task_identity,
    )

    return smiles, normalize_new_task_identity("carcinogens", smiles)


def frozen_inputs():
    roots = [
        Path("data/conditioned_benchmark") / t / s
        for t in ("Carcinogens", "DILI")
        for s in ("scaffold", "random")
    ]
    return {str(p): sha256_file(p) for root in roots for p in root.glob("*.jsonl")}


def identity_evidence(raw, labels, root, fetch=False):
    frozen = root / "name_identity_evidence.jsonl"
    use_frozen = frozen.exists() and not fetch
    molecules = {
        r["input_smiles"]: r
        for r in pq.read_table(
            SOURCE / "canonical_v1/molecule_identities.parquet"
        ).to_pylist()
    }
    legacy_holds = {
        r["source_row_uid"]: r["identity_review_reason"]
        for r in read_jsonl(SOURCE / "gold_v4/identity_source_holds.jsonl")
    }
    known = {
        r["name"].casefold(): r
        for r in read_jsonl(SOURCE / "gold_v2/name_resolutions.jsonl")
    }
    for p in (
        ()
        if use_frozen
        else Path("/local/tmp/txagent-new-starling/name_resolution_cache").glob(
            "*.json"
        )
    ):
        if p.name.endswith(".response.json"):
            continue
        entry = json.loads(p.read_text())
        if entry.get("ok") and entry.get("parent_keys"):
            known.setdefault(entry["name"].casefold(), entry)
    existing = {name: entry["parent_keys"] for name, entry in known.items()}
    needed = {
        molecules[r["SMILES"]]["parent_inchi_key"]
        for r in raw
        if labels[r["source_row_uid"]]["final_candidate_label"] is not None
    }
    entries = {}
    with gzip.open(SOURCE / "gold_v2/identity_evidence.jsonl.gz", "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("kind") == "parent_synonyms":
                entries[entry["parent_key"]] = entry
    if frozen.exists():
        for entry in read_jsonl(frozen):
            if entry.get("kind") == "parent_synonyms":
                entries[entry["parent_key"]] = entry
            elif entry.get("kind") == "cached_name" and entry.get("parent_keys"):
                known[entry["name"].casefold()] = entry
    existing = {name: entry["parent_keys"] for name, entry in known.items()}
    cache = Path("/local/tmp/txagent-new-starling/parent_synonym_cache")
    for key in [] if use_frozen else sorted(needed):
        paths = (root / "identity_cache" / (key + ".json"), cache / (key + ".json"))
        failed = None
        for p in paths:
            if p.is_file():
                d = json.loads(p.read_text())
                if d.get("ok"):
                    entries[key] = d
                    break
                if failed is None:
                    failed = d
        if key not in entries and failed is not None:
            entries[key] = failed
    network_calls = 0
    if fetch:
        import requests

        local = root / "identity_cache"
        local.mkdir(exist_ok=True)
        wanted = {
            molecules[r["SMILES"]]["parent_inchi_key"]
            for r in raw
            if labels[r["source_row_uid"]]["final_candidate_label"] is not None
            and classify_condition(r)["group"]
            and not molecules[r["SMILES"]]["identity_review_reason"]
        }
        missing = []
        verified = {key for key, entry in entries.items() if entry.get("ok")}
        for key in sorted(wanted - verified - {""}):
            p = local / (key + ".json")
            if p.exists() and json.loads(p.read_text()).get("http_status") == 404:
                continue
            missing.append(key)
        next_request = 0.0

        def request_json(url):
            nonlocal network_calls, next_request
            response = None
            for attempt in range(3):
                while time.monotonic() < next_request:
                    time.sleep(max(0, min(30, next_request - time.monotonic())))
                network_calls += 1
                next_request = time.monotonic() + 2
                try:
                    response = requests.get(url, timeout=40)
                    throttling = response.headers.get("X-Throttling-Control", "")
                    if response.status_code in (429, 500, 502, 503, 504):
                        cooldown = float(response.headers.get("Retry-After") or 10)
                        blocked = re.search(
                            r"Remaining blocking time:\s*(\d+):(\d+):(\d+)", throttling
                        )
                        if blocked:
                            cooldown = max(
                                cooldown,
                                sum(
                                    int(n) * unit
                                    for n, unit in zip(blocked.groups(), (3600, 60, 1))
                                )
                                + 10,
                            )
                        next_request = max(next_request, time.monotonic() + cooldown)
                        print(
                            json.dumps(
                                {
                                    "phase": "identity_service_backoff",
                                    "seconds": cooldown,
                                    "throttling": throttling,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    return response, response.json()
                except (requests.RequestException, ValueError):
                    next_request = max(next_request, time.monotonic() + 10)
            return response, {}

        print(
            json.dumps(
                {
                    "phase": "identity_lookup",
                    "cached": len(entries),
                    "pending": len(missing),
                    "batch_size": 50,
                }
            ),
            flush=True,
        )
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            property_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                + ",".join(batch)
                + "/property/InChIKey/JSON"
            )
            response, body = request_json(property_url)
            status = response.status_code if response is not None else 0
            properties = body.get("PropertyTable", {}).get("Properties", [])
            by_key = defaultdict(list)
            for prop in properties:
                if prop["InChIKey"] in batch:
                    by_key[prop["InChIKey"]].append(prop)
            cids = sorted({p["CID"] for ps in by_key.values() for p in ps})
            synonyms_url = None
            by_cid = {}
            if cids:
                synonyms_url = (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    + ",".join(map(str, cids))
                    + "/synonyms/JSON"
                )
                synonym_response, synonym_body = request_json(synonyms_url)
                status = (
                    synonym_response.status_code if synonym_response is not None else 0
                )
                by_cid = {
                    d["CID"]: d
                    for d in synonym_body.get("InformationList", {}).get(
                        "Information", []
                    )
                }
            for key in batch:
                matched = [p["CID"] for p in by_key[key]]
                info = [by_cid[c] for c in matched if c in by_cid]
                entry = {
                    "parent_key": key,
                    "ok": bool(info),
                    "synonyms": sorted({n for d in info for n in d.get("Synonym", [])}),
                    "cids": matched,
                    "properties": by_key[key],
                    "url": synonyms_url or property_url,
                    "identity_url": property_url,
                    "http_status": status,
                    "reason": ""
                    if info
                    else "parent_or_synonyms_not_returned"
                    if status in (200, 404)
                    else "lookup_service_unavailable",
                }
                write_json_atomic(local / (key + ".json"), entry)
                entries[key] = entry
            print(
                json.dumps(
                    {
                        "phase": "identity_lookup",
                        "completed": min(start + 50, len(missing)),
                        "total": len(missing),
                        "http_status": status,
                    }
                ),
                flush=True,
            )
    synonyms = {
        k: {n.casefold() for n in d.get("synonyms", [])}
        for k, d in entries.items()
        if d.get("ok")
    }
    ledger = {}
    clearances = []
    for r in raw:
        uid = r["source_row_uid"]
        y = labels[uid]["final_candidate_label"]
        if y is None:
            continue
        mol = molecules[r["SMILES"]]
        key = mol["parent_inchi_key"]
        name = str(r.get("agent_name") or "").casefold()
        match = existing.get(name) == [key] or name in synonyms.get(key, set())
        pair = re.fullmatch(r"(.+?)\s+\(([^()]*)\)", name)
        if pair and all(s in synonyms.get(key, set()) for s in pair.groups()):
            match = True
        previous = legacy_holds.get(uid, "")
        if (
            previous == "pubchem_name_structure_mismatch"
            and name == "urethane (ethyl carbamate)"
            and key == "JOYRKODLDBILNP-UHFFFAOYSA-N"
            and pair
            and all(s in synonyms.get(key, set()) for s in pair.groups())
        ):
            clearances.append(
                {
                    "source_row_uid": uid,
                    "source_payload_sha256": payload_hash(r),
                    "previous_reason": previous,
                    "canonical_smiles": mol["parent_smiles"],
                    "reason": "both_explicit_aliases_verified_as_ethyl_carbamate_CID5641; composite_name_lookup_also_returns_unrelated_mixture_CID86737834",
                    "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/5641",
                }
            )
            previous = ""
        reason = (
            mol["identity_review_reason"]
            or previous
            or ("" if key and match else "name_not_verified_for_source_structure")
        )
        ledger[uid] = {
            "source_row_uid": uid,
            "Y": y,
            "parent_smiles": mol["parent_smiles"],
            "source_parent_key": key,
            "reason": reason,
        }
    pq.write_table(
        pa.Table.from_pylist(list(ledger.values())),
        root / "identity_ledger.parquet",
        compression="zstd",
    )
    write_jsonl_atomic(root / "source_identity_clearances.jsonl", clearances)
    write_jsonl_atomic(
        root / "name_identity_evidence.jsonl",
        [{"kind": "cached_name", **d} for d in known.values()]
        + [{"kind": "parent_synonyms", **d} for d in entries.values()],
    )
    write_json_atomic(
        root / "identity_summary.json",
        {
            "records": len(ledger),
            "counts": dict(
                Counter(
                    (r["reason"] or "accepted") + ":" + str(r["Y"])
                    for r in ledger.values()
                )
            ),
            "new_network_calls": network_calls,
            "cached_synonym_parents": sum(bool(d.get("ok")) for d in entries.values()),
            "lookup_outcomes": dict(
                Counter(
                    "verified" if d.get("ok") else d.get("reason", "not_verified")
                    for d in entries.values()
                )
            ),
            "source_name_resolutions_sha256": sha256_file(
                SOURCE / "gold_v2/name_resolutions.jsonl"
            ),
        },
    )
    return ledger


def source_units(candidates, raw_by_uid):
    """One publication-parent-group vote; exact repeated passages join units."""
    parents = {}
    for r in candidates:
        key = r["molecule_identity_key"]
        pair = r["drug"], r["bemis_murcko_scaffold"]
        parents[key] = min(pair, parents.get(key, pair))
    roots = {}

    def find(k):
        roots.setdefault(k, k)
        while roots[k] != k:
            roots[k] = roots[roots[k]]
            k = roots[k]
        return k

    seen = {}
    links = []
    for r in candidates:
        k = (r["study_id"], r["molecule_identity_key"], r["condition_group"])
        text = " ".join(
            str(raw_by_uid[r["source_record_id"]].get("support_text") or "")
            .casefold()
            .split()
        )
        if len(text) < 80:
            continue
        signature = (k[1], k[2], text)
        other = seen.setdefault(signature, k)
        a, b = find(k), find(other)
        if a != b:
            roots[max(a, b)] = min(a, b)
            links.append(
                {
                    "unit_a": list(a),
                    "unit_b": list(b),
                    "support_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    units = defaultdict(list)
    for r in candidates:
        r["drug"], r["bemis_murcko_scaffold"] = parents[r["molecule_identity_key"]]
        units[
            find((r["study_id"], r["molecule_identity_key"], r["condition_group"]))
        ].append(r)
    votes = []
    conflicts = []
    dispositions = {}
    for k, rs in sorted(units.items()):
        uids = sorted(r["source_record_id"] for r in rs)
        if len({r["Y"] for r in rs}) > 1:
            conflicts.append(
                {
                    "study_id": k[0],
                    "molecule_identity_key": k[1],
                    "condition_group": k[2],
                    "source_record_ids": uids,
                    "label_counts": dict(Counter(r["Y"] for r in rs)),
                }
            )
            for uid in uids:
                dispositions[uid] = {
                    "disposition": "within_publication_label_conflict",
                    "representative_uid": None,
                }
        else:
            rep = dict(min(rs, key=lambda r: r["source_record_id"]))
            rep.update(
                study_source_row_uids=uids,
                study_source_pmids=sorted({r["pmid"] for r in rs}),
                deduplication_unit=list(k),
            )
            votes.append(rep)
            for uid in uids:
                dispositions[uid] = {
                    "disposition": "study_vote_representative"
                    if uid == rep["source_record_id"]
                    else "duplicate_support_of_study_vote",
                    "representative_uid": rep["source_record_id"],
                }
    return votes, conflicts, links, dispositions


def build(root, fetch_identities=False, identities_only=False):
    from tools.chembl_tool.common.starling.fresh_conditioned_benchmark import (
        build_fresh_conditioned_benchmark,
    )
    from tools.chembl_tool.common.starling import conditioned_benchmark as paths
    from tools.chembl_tool.common.starling import (
        build_conditioned_random_split as random_builder,
    )

    root.mkdir(parents=True, exist_ok=True)
    before = frozen_inputs()
    write_json_atomic(root / "frozen_inputs.json", before)
    labels = {r["source_row_uid"]: r for r in pq.read_table(LABELS).to_pylist()}
    raw = pq.read_table(RAW).to_pylist()
    raw_by_uid = {r["source_row_uid"]: r for r in raw}
    assert len(raw_by_uid) == len(raw) == len(labels) == 354386
    conditions = []
    for r in raw:
        uid = r["source_row_uid"]
        assert payload_hash(r) == labels[uid]["source_payload_sha256"]
        conditions.append(
            {
                "source_row_uid": uid,
                "source_payload_sha256": labels[uid]["source_payload_sha256"],
                "Y": labels[uid]["final_candidate_label"],
                **classify_condition(r),
            }
        )
    by_condition = {r["source_row_uid"]: r for r in conditions}
    pq.write_table(
        pa.Table.from_pylist(conditions),
        root / "record_conditions.parquet",
        compression="zstd",
    )
    print(
        json.dumps(
            {
                "phase": "conditions",
                "counts": dict(Counter((r["group"] or r["basis"]) for r in conditions)),
            }
        ),
        flush=True,
    )
    identity = identity_evidence(raw, labels, root, fetch=fetch_identities)
    if identities_only:
        return
    # The old Methanal conflict is an extraction-subject mismatch: PMID 33636299
    # studies methyl acrylate (MA). Never flip its direction or silently reassign it.
    holds = []
    for r in raw:
        if str(r["pmid"]) == "33636299" and str(
            r.get("agent_name") or ""
        ).casefold() in {"methanal", "formaldehyde", "methyl isothiocyanate"}:
            holds.append(
                {
                    "source_row_uid": r["source_row_uid"],
                    "source_payload_sha256": payload_hash(r),
                    "reason": "confirmed_MA_methyl_acrylate_misexpanded_as_another_chemical",
                    "source_url": "https://pubmed.ncbi.nlm.nih.gov/33636299/",
                    "source_title": "Comprehensive analysis of chronic rodent inhalation toxicity studies for methyl acrylate with attention to test conditions exceeding a maximum tolerated concentration",
                    "decision": "exclude misattributed record from gold and retrieval; retain raw source and direction",
                }
            )
    write_jsonl_atomic(root / "source_identity_holds.jsonl", holds)
    held = {r["source_row_uid"] for r in holds}
    needed = {
        r["parent_smiles"]
        for uid, r in identity.items()
        if not r["reason"] and by_condition[uid]["group"] and uid not in held
    }
    cache_path = root / "tautomer_identities.jsonl"
    canonical = (
        {r["input_smiles"]: r["identity"] for r in read_jsonl(cache_path)}
        if cache_path.exists()
        else {}
    )
    for r in read_jsonl(SOURCE / "gold_v3/identity_lineage.jsonl"):
        if r.get("input_drug"):
            canonical.setdefault(
                r["input_drug"], {k: v for k, v in r.items() if k != "input_drug"}
            )
    print(
        json.dumps(
            {
                "phase": "identities",
                "needed": len(needed),
                "new_normalizations": len(needed - canonical.keys()),
            }
        ),
        flush=True,
    )
    with ProcessPoolExecutor(16) as pool:
        for s, d in pool.map(
            normalize_identity, sorted(needed - canonical.keys()), chunksize=4
        ):
            canonical[s] = d
    write_jsonl_atomic(
        cache_path,
        ({"input_smiles": s, "identity": d} for s, d in sorted(canonical.items())),
    )
    candidates = []
    dispositions = {}
    for r in raw:
        uid = r["source_row_uid"]
        label = labels[uid]
        y = label["final_candidate_label"]
        cond = by_condition[uid]
        reason = None
        if y is None:
            reason = "source_" + label["final_direction"]
        elif uid in held:
            reason = "confirmed_source_subject_mismatch"
        elif not cond["group"]:
            reason = cond["basis"]
        elif identity[uid]["reason"]:
            reason = identity[uid]["reason"]
        elif not re.fullmatch(r"[1-9][0-9]*", str(r.get("pmid") or "")):
            reason = "reporting_pmid_missing"
        if reason:
            dispositions[uid] = {
                "source_row_uid": uid,
                "Y": y,
                "source_direction": label["final_direction"],
                "condition_group": cond["group"],
                "disposition": reason,
                "representative_uid": None,
            }
            continue
        ident = canonical[identity[uid]["parent_smiles"]]
        group = "species=" + cond["group"]
        pmid = str(r["pmid"])
        candidates.append(
            {
                "source_record_id": uid,
                "source_row_uid": uid,
                "source_id": "carcinogens_base",
                "drug": ident["drug"],
                "molecule_identity_key": ident["molecule_identity_key"],
                "bemis_murcko_scaffold": ident["scaffold_group_smiles"],
                "original_bemis_murcko_scaffold": ident["bemis_murcko_scaffold"],
                "leakage_group": ident["leakage_group"],
                "scaffold_leakage_group": ident["scaffold_leakage_group"],
                "Y": y,
                "condition_group": group,
                "condition_atoms": [group],
                "condition_text": cond["quote"],
                "condition_normalization": cond,
                "pmid": pmid,
                "study_id": "pmid:" + pmid,
                "molecule_name": r["agent_name"],
                "label_method": VERSION,
                "reviewer": label["decision_origin"]
                + "; source_field_organism_normalization",
                "raw_value": r["carcinogenicity_conclusion"],
                "source_payload_sha256": label["source_payload_sha256"],
                "vote_unit": VOTE_UNIT,
                "identity_verification": "exact_name_or_official_parent_synonym",
                "raw_source_path": str(RAW),
            }
        )
    votes, conflicts, links, unit_dispositions = source_units(candidates, raw_by_uid)
    for r in candidates:
        uid = r["source_record_id"]
        dispositions[uid] = {
            "source_row_uid": uid,
            "Y": r["Y"],
            "source_direction": labels[uid]["final_direction"],
            "condition_group": r["condition_group"],
            **unit_dispositions[uid],
        }
    assert set(dispositions) == set(labels)
    pq.write_table(
        pa.Table.from_pylist(list(dispositions.values())),
        root / "record_dispositions.parquet",
        compression="zstd",
    )
    write_jsonl_atomic(root / "source_votes.jsonl", votes)
    write_jsonl_atomic(root / "study_conflicts.jsonl", conflicts)
    write_jsonl_atomic(root / "exact_passage_duplicate_links.jsonl", links)
    write_json_atomic(
        root / "build_policy.json",
        {
            "version": VERSION,
            "direction_ledger": str(LABELS),
            "new_model_calls": 0,
            "condition_policy": CONDITION_VERSION,
            "allowed_groups": GROUPS,
            "missing_or_cross_group_species": "pending, never infer rodent",
            "other_axes": "pooled for source-consensus target; original restrictions retained; not universal carcinogenicity across exposures",
            "in_vitro": "retrieval/background only, no organism gold",
            "vote_unit": VOTE_UNIT,
            "within_publication_conflict": "no vote",
            "condition_agreement": 0.6,
            "ties": "unclassified",
            "tdc_labels_used": False,
            "provenance_limit": "Publication-level consensus, not a full audit of independent primary experiments",
        },
    )
    print(
        json.dumps(
            {
                "phase": "source_votes",
                "candidates": len(candidates),
                "votes": len(votes),
                "labels": dict(Counter(r["Y"] for r in votes)),
                "conflicts": len(conflicts),
            }
        ),
        flush=True,
    )
    artifacts = [
        RAW,
        LABELS,
        root / "record_conditions.parquet",
        root / "identity_ledger.parquet",
        root / "name_identity_evidence.jsonl",
        root / "build_policy.json",
        root / "source_votes.jsonl",
        root / "source_identity_holds.jsonl",
        root / "source_identity_clearances.jsonl",
        Path(__file__),
        Path(__file__).with_name("reviewed_conditions.py"),
    ]
    benchmark = root / "source_only_benchmark"
    build_fresh_conditioned_benchmark(
        task="carcinogens",
        record_votes=votes,
        output_root=benchmark / "Carcinogens/scaffold",
        required_labels=(0, 1),
        source_artifacts=artifacts,
    )
    old_root = paths.BENCHMARK_ROOT
    old_allocator = random_builder.allocate_parent_groups
    leakage = {r["molecule_identity_key"]: r["leakage_group"] for r in votes}

    def allocate(rows, **kwargs):
        grouped = [
            {**r, "molecule_identity_key": leakage[r["molecule_identity_key"]]}
            for r in rows
        ]
        assignment, diagnostics = old_allocator(grouped, **kwargs)
        return {
            r["molecule_identity_key"]: assignment[leakage[r["molecule_identity_key"]]]
            for r in rows
        }, diagnostics

    try:
        paths.BENCHMARK_ROOT = benchmark
        random_builder.allocate_parent_groups = allocate
        random_builder.build_task("carcinogens", minimum_feasible_eval_size=True)
    finally:
        paths.BENCHMARK_ROOT = old_root
        random_builder.allocate_parent_groups = old_allocator
    from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
        _minimal_row,
    )

    b = benchmark / "Carcinogens"
    gold = read_jsonl(b / "scaffold/accepted_parent_conditions_before_group_gate.jsonl")
    selected = read_jsonl(b / "scaffold/accepted_parent_conditions.jsonl")
    selected_ids = {r["benchmark_row_id"] for r in selected}
    rejected = read_jsonl(b / "scaffold/rejected_parent_conditions.jsonl")
    for r in gold:
        r.update(
            vote_unit=VOTE_UNIT,
            gold_contract=VERSION,
            split_eligible=r["benchmark_row_id"] in selected_ids,
        )
    write_jsonl_atomic(
        root / "gold_labels.jsonl",
        (
            {
                **_minimal_row(r),
                "split_eligible": r["split_eligible"],
                "gold_contract": VERSION,
                "positive_votes": sum(v["Y"] == 1 for v in r["source_votes"]),
                "negative_votes": sum(v["Y"] == 0 for v in r["source_votes"]),
            }
            for r in gold
        ),
    )
    write_jsonl_atomic(root / "gold_label_provenance.jsonl", gold)
    write_jsonl_atomic(
        root / "gold_rare_conditions.jsonl",
        (r for r in gold if not r["split_eligible"]),
    )
    write_jsonl_atomic(
        root / "unclassified_parent_conditions.jsonl",
        (r for r in rejected if "Y" not in r),
    )
    voting_pairs = {(r["molecule_identity_key"], r["condition_group"]) for r in votes}
    only_conflicts = [
        r
        for r in conflicts
        if (r["molecule_identity_key"], r["condition_group"]) not in voting_pairs
    ]
    write_jsonl_atomic(root / "only_conflicted_parent_conditions.jsonl", only_conflicts)
    unions = {}
    split_counts = {}
    overlap = {}
    for scheme in ("scaffold", "random"):
        splits = {
            s: read_jsonl(b / scheme / (s + ".jsonl"))
            for s in ("train", "valid", "test")
        }
        unions[scheme] = {
            (r["benchmark_row_id"], r["Y"]) for rs in splits.values() for r in rs
        }
        assert sum(map(len, splits.values())) == len(unions[scheme])
        conditions = [{r["condition_group"] for r in rs} for rs in splits.values()]
        assert conditions[0] == conditions[1] == conditions[2]
        ids = [
            {leakage[r["molecule_identity_key"]] for r in rs} for rs in splits.values()
        ]
        overlap[scheme] = [len(ids[i] & ids[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        assert overlap[scheme] == [0, 0, 0]
        if scheme == "scaffold":
            ids = [
                {r["bemis_murcko_scaffold"] for r in rs if r["bemis_murcko_scaffold"]}
                for rs in splits.values()
            ]
            assert not any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        split_counts[scheme] = {
            s: dict(Counter(r["Y"] for r in rs)) for s, rs in splits.items()
        }
    assert (
        unions["scaffold"]
        == unions["random"]
        == {(r["benchmark_row_id"], r["Y"]) for r in selected}
    )
    seen = set()
    for v in votes:
        uids = set(v["study_source_row_uids"])
        assert not seen & uids
        seen.update(uids)
        assert all(labels[uid]["final_candidate_label"] == v["Y"] for uid in uids)
    for r in gold:
        ys = [v["Y"] for v in r["source_votes"]]
        a = sum(ys)
        z = len(ys) - a
        assert a != z and r["Y"] == int(a > z) and max(a, z) / len(ys) >= 0.6
    for p, h in before.items():
        assert sha256_file(Path(p)) == h
    summary = {
        "version": VERSION,
        "raw_records": len(raw),
        "source_directions": dict(
            Counter(r["final_direction"] for r in labels.values())
        ),
        "dispositions": dict(Counter(r["disposition"] for r in dispositions.values())),
        "dispositions_by_label": dict(
            Counter(r["disposition"] + ":" + str(r["Y"]) for r in dispositions.values())
        ),
        "source_votes": len(votes),
        "source_vote_labels": dict(Counter(r["Y"] for r in votes)),
        "within_publication_conflicts": len(conflicts),
        "exact_passage_duplicate_links": len(links),
        "gold_rows": len(gold),
        "gold_labels": dict(Counter(r["Y"] for r in gold)),
        "gold_by_condition": {
            g: dict(
                Counter(r["Y"] for r in gold if r["condition_group"] == "species=" + g)
            )
            for g in GROUPS
        },
        "split_cohort_rows": len(selected),
        "split_cohort_conditions": len({r["condition_group"] for r in selected}),
        "splits": split_counts,
        "unclassified_parent_conditions": sum("Y" not in r for r in rejected),
        "only_conflicted_parent_conditions": len(
            {(r["molecule_identity_key"], r["condition_group"]) for r in only_conflicts}
        ),
        "tdc_labels_used": False,
        "new_model_calls": 0,
        "retrieval_status": "pending",
        "inputs": {str(p): sha256_file(p) for p in artifacts},
    }
    write_json_atomic(root / "summary.json", summary)
    write_json_atomic(
        root / "validation.json",
        {
            "status": "passed",
            "all_raw_uids_accounted_once": True,
            "source_directions_preserved": True,
            "gold_majorities_recomputed": True,
            "same_cohort_labels_across_splits": True,
            "leakage_group_overlap": overlap,
            "scaffold_overlap": 0,
            "condition_coverage_all_splits": True,
            "canonical_inputs_unchanged": True,
            "tdc_votes": 0,
            "new_model_calls": 0,
            "gold_sha256": sha256_file(root / "gold_labels.jsonl"),
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--fetch-identities",
        action="store_true",
        help="Resolve uncached candidate structures through official PubChem synonyms",
    )
    p.add_argument("--phase", choices=("build", "identities"), default="build")
    args = p.parse_args()
    build(args.root, args.fetch_identities, args.phase == "identities")


if __name__ == "__main__":
    main()
