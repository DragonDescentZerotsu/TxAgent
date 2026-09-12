"""Strict direct-source votes and shared conditioned splits for new tasks.

The complete source records remain untouched. Vote decisions, name resolution,
study deduplication and rejected conditions are separately auditable artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib
import importlib.util
import json
from pathlib import Path
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import read_jsonl, write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.build_runtime import sha256_file, local_workdir, local_input, publish_file
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

ROOT = Path("data/starling_data")


def clean(value):
    return "" if value is None else str(value).strip()


_rows = _identities = _policy = _reviews = _task = None
GOLD_DIR = "gold_v2"


def _prepare_range(bounds):
    from tools.chembl_tool.common.starling.source_gold_review import payload_hash, study_identity, outcome_in_condition, answer_bearing_atoms
    decisions, candidates = [], []
    for index in range(*bounds):
        raw = _rows[index]
        uid = raw["source_row_uid"]
        molecule = _identities[clean(raw["SMILES"])]
        label, reason, atoms = _policy.decide(raw)
        initial_reason, initial_label = reason, label
        name = clean(raw.get("molecule_name") or raw.get("agent_name"))
        study, study_reason = study_identity(raw) if label is not None else (None, "")
        review = _reviews.get(uid)
        if review:
            if review["source_payload_sha256"] != payload_hash(raw):
                raise ValueError("Stale source review: " + uid)
            label, reason = review["Y"], review["reason"]
            atoms = review.get("condition_atoms", atoms)
            study = review.get("study_id", study)
            study_reason = "" if study else study_reason
        if label is not None:
            if molecule["identity_review_reason"]:
                label, reason = None, molecule["identity_review_reason"]
            elif not study:
                label, reason = None, study_reason or "original_study_unresolved"
            elif not review and outcome_in_condition(raw, _task):
                label, reason = None, "answer_bearing_condition_requires_semantic_review"
            elif answer_bearing_atoms(atoms, _task):
                label, reason = None, "answer_bearing_rendered_condition_requires_semantic_review"
            elif not review and any(a.startswith("unresolved_population_context=") for a in atoms):
                label, reason = None, "multiple_population_context_requires_semantic_review"
            elif raw.get("needs_more_context") and not review:
                label, reason = None, "source_context_flag_requires_review"
        if label is not None:
            candidates.append({
                "source_record_id": uid, "source_row_uid": uid,
                "source_id": _task + "_base", "drug": molecule["parent_smiles"],
                "molecule_identity_key": molecule["parent_inchi_key"],
                "bemis_murcko_scaffold": molecule["bemis_murcko_scaffold"],
                "Y": label, "condition_atoms": sorted(set(atoms)), "condition_group": "+".join(sorted(set(atoms))),
                "condition_text": clean(raw.get("qualifying_conditions")), "pmid": clean(raw["pmid"]),
                "study_id": study, "label_method": _policy.VERSION,
                "reviewer": "codex_record_semantic_review" if review else "source_contract_with_stratified_codex_audit_not_human_expert",
                "raw_value": raw.get("causal_status") or raw.get("carcinogenicity_conclusion"),
                "molecule_name": name, "raw_record": raw,
                "source_payload_sha256": payload_hash(raw),
            })
        decisions.append({"source_row_uid": uid, "candidate_label": label, "reason": reason,
            "rule_candidate_label": initial_label, "rule_reason": initial_reason, "record_reviewed": bool(review)})
    return decisions, candidates


def prepare(task):
    from tools.chembl_tool.common.build_runtime import worker_pool
    global _rows, _identities, _policy, _reviews, _task
    _task = task
    _policy = importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_gold")
    root = ROOT / task
    _identities = {r["input_smiles"]: r for r in pq.read_table(root / "canonical_v1/molecule_identities.parquet").to_pylist()}
    _rows = pq.read_table(root / f"raw_v1/compressed/{task}_base.parquet").to_pylist()
    out = root / GOLD_DIR
    out.mkdir(parents=True, exist_ok=True)
    review_path = out / "record_reviews.jsonl"
    _reviews = {r["source_row_uid"]: r for r in read_jsonl(review_path)} if review_path.exists() else {}
    decisions, candidates = [], []
    with worker_pool(32) as pool:
        for ds, cs in pool.map(_prepare_range, [(i, min(i+2048, len(_rows))) for i in range(0, len(_rows), 2048)]):
            decisions.extend(ds); candidates.extend(cs)
    studies = defaultdict(list)
    for row in candidates:
        studies[row["study_id"], row["molecule_identity_key"], row["condition_group"]].append(row)
    collapsed, conflicts = [], []
    for key, vs in sorted(studies.items()):
        if len({r["Y"] for r in vs}) != 1:
            conflicts.extend({"source_row_uid":r["source_row_uid"],"study_id":key[0],"reason":"within_study_label_conflict"} for r in vs)
            continue
        # Prefer reviewed and primary reporting records over citing articles.
        representative = dict(min(vs, key=lambda r: (r["reviewer"] != "codex_record_semantic_review", r["source_record_id"])))
        representative["study_source_row_uids"] = sorted(r["source_row_uid"] for r in vs)
        representative["study_source_pmids"] = sorted({r["pmid"] for r in vs})
        collapsed.append(representative)
    pq.write_table(pa.Table.from_pylist(decisions), out / "base_record_decisions.parquet", compression="zstd")
    write_jsonl_atomic(out / "candidate_votes.jsonl", candidates)
    write_jsonl_atomic(out / "identity_candidates.jsonl", collapsed)
    write_jsonl_atomic(out / "study_conflicts.jsonl", conflicts)
    summary = {
        "task": task, "contract": _policy.VERSION, "target": _policy.TARGET,
        "base_rows": len(_rows), "candidate_rows": len(candidates), "candidate_label_counts":dict(Counter(r["Y"] for r in candidates)),
        "unambiguous_study_votes": len(collapsed), "study_conflicting_records":len(conflicts),
        "identity_candidate_votes": len(collapsed),
        "identity_candidate_names": len({r['molecule_name'] for r in collapsed}),
        "identity_candidate_parents": len({r['molecule_identity_key'] for r in collapsed}),
        "reason_counts": dict(Counter(r['reason'] for r in decisions)),
        "minimum_studies": 1, "study_identity_policy": "explicit_original_trial_or_author_year_else_reporting_pmid; unresolved_secondary_origin_pending",
        "condition_policy": "normalized known axes with unparsed qualifiers retained; no candidate condition-coverage gate",
        "semantic_review": "stratified Codex source-text audit and hash-bound record overrides; not human-expert/full-paper review",
        "n_record_reviews":len(_reviews),
        "policy_sha256": sha256_file(Path(_policy.__file__)),
        "shared_review_policy_sha256": sha256_file(Path(__file__).with_name("source_gold_review.py")),
        "record_reviews_sha256": sha256_file(review_path) if review_path.exists() else None,
    }
    write_json_atomic(out / "preparation.json", summary)
    print(json.dumps(summary), flush=True)
    return collapsed


def resolve(tasks, client_path):
    """Resolve exact names through cached lookups and parent-key synonyms.

    One cached PubChem parent lookup covers all source spelling variants. Only
    exact case-insensitive official synonyms match; no fuzzy name correction.
    """
    spec = importlib.util.spec_from_file_location("pubchem_skill_client", client_path)
    client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
    rows = [r for task in tasks for r in read_jsonl(ROOT / task / f"{GOLD_DIR}/identity_candidates.jsonl")]
    names = {r["molecule_name"] for r in rows}
    cache = Path("/local/tmp/txagent-new-starling/name_resolution_cache")
    synonyms = Path("/local/tmp/txagent-new-starling/parent_synonym_cache")
    synonyms.mkdir(parents=True, exist_ok=True)
    known = {}
    for path in cache.glob("*.json"):
        if path.name.endswith(".response.json"): continue
        entry = json.loads(path.read_text())
        if entry.get("parent_keys"): known[entry["name"].casefold()] = entry
    ames = ROOT / "ames/identity_review_v1/pubchem_name_responses.jsonl"
    folded_names = {n.casefold() for n in names}
    if ames.exists():
        for entry in read_jsonl(ames):
            if entry["name"].casefold() not in folded_names or not entry.get("ok"): continue
            props = entry.get("response", {}).get("PropertyTable", {}).get("Properties", [])
            keys = sorted({normalize_molecule_identity(p.get("SMILES") or "").parent_inchi_key for p in props} - {""})
            if keys: known.setdefault(entry["name"].casefold(), {"name":entry["name"],"parent_keys":keys,"cids":[p["CID"] for p in props],"ok":True,"url":entry["url"],"properties":props,"cache_source":str(ames)})
    by_parent = defaultdict(set)
    for r in rows:
        if r["molecule_name"].casefold() not in known:
            by_parent[r["molecule_identity_key"]].add(r["molecule_name"])
    def lookup(key):
        target = synonyms / (key + ".json")
        if target.exists(): return json.loads(target.read_text())
        path = "compound/inchikey/" + key + "/synonyms/JSON"
        raw = synonyms / (key + ".response.json")
        result = client.execute({"base_url":"https://pubchem.ncbi.nlm.nih.gov/rest/pug", "path":path,
            "record_path":"InformationList.Information", "max_items":1, "timeout_sec":25,
            "save_raw":True,"raw_output_path":str(raw)})
        info = json.loads(raw.read_text()).get("InformationList",{}).get("Information",[]) if result.get("ok") and raw.exists() else []
        entry = {"parent_key":key,"synonyms":sorted({v for d in info for v in d.get("Synonym",[])}),
            "cids":[d["CID"] for d in info],"ok":bool(info),"url":"https://pubchem.ncbi.nlm.nih.gov/rest/pug/"+path}
        write_json_atomic(target, entry)
        return entry
    entries = []
    pending = [key for key in sorted(by_parent) if not (synonyms/(key+".json")).exists()]
    entries.extend(json.loads((synonyms/(key+".json")).read_text()) for key in sorted(by_parent) if key not in set(pending))
    print(f"Identity: {len(names)} source names; {len(known)} cached names; {len(pending)} unique parent synonym requests",flush=True)
    futures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,key in enumerate(pending,1):
            futures.append(pool.submit(lookup,key)); time.sleep(0.26)
            if i%100==0: print(f"Identity submitted {i}/{len(pending)}; cached {sum(f.done() for f in futures)}",flush=True)
        entries.extend(f.result() for f in futures)
    matches=defaultdict(set); urls=defaultdict(list); cids=defaultdict(set)
    for entry in entries:
        aliases={n.casefold() for n in entry["synonyms"]}
        for name in names:
            explicit_alias = re.fullmatch(r"(.+?)\s+\(([^()]*)\)", name)
            # E.g. full chemical name (DMBA): accept only when BOTH literal
            # components are official synonyms of this exact parent. Never
            # drop stereochemistry, salt qualifiers or arbitrary parentheses.
            paired_alias = bool(explicit_alias and all(part.casefold() in aliases for part in explicit_alias.groups()))
            if name.casefold() in aliases or paired_alias:
                matches[name].add(entry["parent_key"]);urls[name].append(entry["url"]);cids[name].update(entry["cids"])
    results=[]
    for name in sorted(names):
        cached=known.get(name.casefold())
        keys=set(cached["parent_keys"] if cached else [])|matches[name]
        results.append({"name":name,"parent_keys":sorted(keys),"ok":bool(keys),
            "cids":sorted(cids[name]|set(cached.get("cids",[]) if cached else [])),
            "url":cached.get("url") if cached else urls[name],"method":"cached_exact_name_or_official_synonym_including_two_verified_explicit_aliases_for_full_parent_inchikey"})
    for task in tasks:
        relevant={r["molecule_name"] for r in read_jsonl(ROOT/task/f"{GOLD_DIR}/identity_candidates.jsonl")}
        write_jsonl_atomic(ROOT/task/f"{GOLD_DIR}/name_resolutions.jsonl",(r for r in results if r["name"] in relevant))
        relevant_parents = {r["molecule_identity_key"] for r in rows if r["source_id"] == task + "_base"}
        evidence = [{"kind":"cached_name","source_name":name,**known[name.casefold()]} for name in sorted(relevant) if name.casefold() in known]
        evidence += [{"kind":"parent_synonyms",**entry} for entry in entries if entry["parent_key"] in relevant_parents]
        with (ROOT/task/GOLD_DIR/"identity_evidence.jsonl.gz").open("wb") as raw_handle:
            with gzip.GzipFile(filename="",mode="wb",fileobj=raw_handle,mtime=0) as handle:
                for entry in evidence:
                    handle.write((json.dumps(entry,ensure_ascii=False)+"\n").encode())
    print(f"Resolved {len(results)} names; {sum(bool(r['parent_keys']) for r in results)} exact name/parent correspondences",flush=True)


def votes(task):
    root = ROOT / task / GOLD_DIR
    resolutions = {r["name"]: r for r in read_jsonl(root / "name_resolutions.jsonl")}
    accepted, rejected = [], []
    for row in read_jsonl(root / "identity_candidates.jsonl"):
        keys = resolutions.get(row["molecule_name"], {}).get("parent_keys", [])
        if keys != [row["molecule_identity_key"]]:
            rejected.append({"source_row_uid": row["source_row_uid"], "reason": "name_structure_mismatch" if keys else "unresolved_name_identity", "resolved_parent_keys": keys})
            continue
        row["identity_verification"] = "pubchem_name_exact_parent_match"
        accepted.append(row)
    groups = defaultdict(list)
    for row in accepted:
        groups[row["molecule_identity_key"], row["condition_group"]].append(row)
    accepted = [r for rs in groups.values() for r in rs]
    # Standard InChI can identify tautomeric source SMILES as the same parent.
    # Freeze one deterministic query representation; retain each source form.
    representations = {}
    for row in accepted:
        key = row["molecule_identity_key"]
        pair = (row["drug"], row["bemis_murcko_scaffold"])
        representations[key] = min(pair, representations.get(key, pair))
    for row in accepted:
        row["source_parent_smiles"] = row["drug"]
        row["drug"], row["bemis_murcko_scaffold"] = representations[row["molecule_identity_key"]]
    write_jsonl_atomic(root / "source_votes.jsonl", accepted)
    write_jsonl_atomic(root / "identity_rejections.jsonl", rejected)
    accepted_ids = {r["source_record_id"] for r in accepted}
    rejection_reasons = {r["source_row_uid"]: r["reason"] for r in rejected}
    ready_ids = {r["source_record_id"] for r in read_jsonl(root / "identity_candidates.jsonl")}
    claims = {(r["study_id"], r["molecule_identity_key"], r["condition_group"]) for r in accepted}
    dispositions = []
    for row in read_jsonl(root / "candidate_votes.jsonl"):
        uid = row["source_record_id"]
        reason = ("accepted_source_vote" if uid in accepted_ids else rejection_reasons.get(uid)
            or ("duplicate_of_accepted_study_claim" if (row["study_id"], row["molecule_identity_key"], row["condition_group"]) in claims
                else "identity_not_verified" if uid in ready_ids else "duplicate_or_within_study_conflict"))
        dispositions.append({"source_row_uid": uid, "reason": reason})
    pq.write_table(pa.Table.from_pylist(dispositions), root / "candidate_dispositions.parquet", compression="zstd")
    write_json_atomic(root / "condition_definitions.json", {r['condition_group']: [x.replace('%2B', '+').replace('%25', '%') for x in r['condition_atoms']] for r in accepted})
    print(task, "verified votes", len(accepted), "parents", len({r['molecule_identity_key'] for r in accepted}), "labels", Counter(r['Y'] for r in accepted), flush=True)


def finalize(task):
    import pyarrow.compute as pc
    root = ROOT / task
    votes_path = root / GOLD_DIR / "source_votes.jsonl"
    source_votes = read_jsonl(votes_path)
    accepted = pa.array([r["source_record_id"] for r in source_votes])
    canonical = root / "canonical_v1"
    manifest = json.loads((canonical / "manifest.json").read_text())
    count = 0
    identity_counts = Counter()
    resolutions = {r["name"]: r["parent_keys"][0] for r in read_jsonl(root / f"{GOLD_DIR}/name_resolutions.jsonl") if len(r["parent_keys"]) == 1}
    with local_workdir() as staging:
        source = pq.ParquetFile(local_input(canonical / "records.parquet"))
        writer = None
        try:
            for batch in source.iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch])
                is_voter = pc.is_in(table["source_row_uid"], value_set=accepted)
                count += pc.sum(is_voter).as_py()
                table = table.set_column(table.schema.get_field_index("is_gold_voter"), "is_gold_voter", is_voter)
                base = pc.equal(table["source_id"], task + "_base")
                group = pc.if_else(is_voter, "Direct." + task, pc.if_else(base, "NearDirect." + task, table["group_id"]))
                table = table.set_column(table.schema.get_field_index("group_id"), "group_id", group)
                reasons, verified = [], []
                for name, parent, reason in zip(table["molecule_name"].to_pylist(), table["molecule_identity_key"].to_pylist(), table["identity_review_reason"].to_pylist(), strict=True):
                    resolved = resolutions.get(name)
                    stereo_uncertain = bool(resolved and parent != resolved and parent.split("-")[0] == resolved.split("-")[0])
                    if stereo_uncertain and reason == "pubchem_name_structure_mismatch":
                        reason = ""
                    if resolved and parent != resolved and not stereo_uncertain:
                        reason = reason or "pubchem_name_structure_mismatch"
                    reasons.append(reason)
                    verified.append("pubchem_name_exact_parent_match" if resolved == parent else "stereochemistry_or_isotope_unresolved" if stereo_uncertain else "pubchem_name_structure_mismatch" if resolved else "source_structure_only")
                identity_counts.update(reason or "retrieval_eligible" for reason in reasons)
                table = table.set_column(table.schema.get_field_index("retrieval_eligible"), "retrieval_eligible", pa.array([not reason for reason in reasons]))
                table = table.set_column(table.schema.get_field_index("identity_review_reason"), "identity_review_reason", pa.array(reasons))
                table = table.set_column(table.schema.get_field_index("identity_verification"), "identity_verification", pa.array(verified))
                if writer is None:
                    writer = pq.ParquetWriter(staging / "records.parquet", table.schema, compression="zstd", compression_level=3)
                writer.write_table(table)
        finally:
            if writer:
                writer.close()
        if count != len(source_votes):
            raise ValueError("Gold source UID membership mismatch")
        manifest["files"]["records.parquet"] = publish_file(staging / "records.parquet", canonical / "records.parquet")
    manifest.update(n_gold_voters=count, n_retrieval_eligible=identity_counts["retrieval_eligible"], identity_status_counts=dict(identity_counts), gold_status="source_votes_rebuilt_with_stratified_codex_semantic_review", gold_votes_sha256=sha256_file(votes_path),
        membership_builder_sha256=sha256_file(Path(__file__)), family_status="actual_voter_direct_and_base_nonvoter_neardirect; mechanism_source_groups_pending_family_review")
    write_json_atomic(canonical / "manifest.json", manifest)
    from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
    benchmarks = {}
    for scheme in ("scaffold", "random"):
        split_root = task_root(task, scheme)
        benchmarks[scheme] = {"root": str(split_root), "files": {p.name: sha256_file(p) for p in sorted(split_root.glob("*.jsonl"))},
            "splits": {split: {"rows": len(rows), "label_counts": dict(Counter(r["Y"] for r in rows))}
                for split in ("train", "valid", "test") for rows in [read_jsonl(split_root / f"{split}.jsonl")]}}
    write_json_atomic(root / "dataset_manifest.json", {
        "source_commit": manifest["source_commit"], "records": {"path": str(canonical / "records.parquet"), "sha256": manifest["files"]["records.parquet"], "rows": manifest["n_records"]},
        "gold": {"path": str(votes_path), "sha256": sha256_file(votes_path), "source_votes": count},
        "raw_records_deleted": 0, "identity": "PubChem exact name-parent match required for gold; source identity eligibility for retrieval",
        "benchmark": benchmarks, "evaluation_status": "rebuilt_source_derived_cohort_with_stratified_semantic_review; no model evaluations",
        "semantic_review": {"records": len(read_jsonl(root / GOLD_DIR / "record_reviews.jsonl")), "ledger": str(root / GOLD_DIR / "record_reviews.jsonl"), "scope": "Codex source-text reading of accepted and rejected strata; not exhaustive, human-expert, or full-paper verification"},
        "retrieval_status": "records_ready; mechanism_family_review_and_heldout_filtered_indices_pending",
        "gold_artifacts": {p.name: sha256_file(p) for p in sorted((root / GOLD_DIR).iterdir()) if p.is_file()},
        "policy_files": {str(p): sha256_file(p) for p in (Path(__file__), Path(importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_gold").__file__), Path(__file__).with_name("source_gold_review.py"))},
    })
    print(task, "annotated actual voters", count, flush=True)


def benchmark(tasks):
    from tools.chembl_tool.common.starling.fresh_conditioned_benchmark import build_fresh_conditioned_benchmark
    from tools.chembl_tool.common.starling.publish_conditioned_benchmark import publish
    for task in tasks:
        root = ROOT / task / GOLD_DIR
        source_votes = read_jsonl(root / "source_votes.jsonl")
        if {r["Y"] for r in source_votes} != {0, 1}:
            raise ValueError(f"{task}: both classes are required before benchmark publication")
        build_fresh_conditioned_benchmark(task=task, record_votes=source_votes, required_labels=(0, 1),
            source_artifacts=(root / "source_votes.jsonl", root / "name_resolutions.jsonl", root / "preparation.json", root / "record_reviews.jsonl", Path(__file__).with_name("source_gold_review.py"), Path(__file__), Path(importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_gold").__file__)))
        directory = {"dili": "DILI", "carcinogens": "Carcinogens"}[task]
        staged = Path("data/.build/conditioned_benchmark_sources") / directory / "scaffold"
        for source, target in (("accepted_parent_conditions_before_group_gate.jsonl", "parent_condition_labels.jsonl"),
                               ("rejected_parent_conditions.jsonl", "parent_condition_exclusions.jsonl")):
            write_jsonl_atomic(root / target, read_jsonl(staged / source))
        write_json_atomic(root / "benchmark_selection.json", json.loads((staged / "summary.json").read_text()))
    publish(tasks=tuple(tasks))
    # Refresh previously generated random splits using the final class-coverage
    # contract stored on detailed scaffold rows.
    from tools.chembl_tool.common.starling.build_conditioned_random_split import build_all
    build_all(tasks=tuple(tasks), minimum_feasible_eval_size=True)
    manifest_path = Path("data/conditioned_benchmark/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for task in tasks:
        manifest["tasks"][task].update(
            evaluation_status="source_derived_gold_v2_rebuilt; limited_negative_support; retrieval_family_and_index_validation_pending",
            semantic_audit="data/starling_data/new_tasks_gold_audit/REBUILD_V2.md",
            source_gold_root=str(ROOT / task / GOLD_DIR),
        )
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["dili", "carcinogens"], choices=("dili", "carcinogens"))
    parser.add_argument("--phase", choices=("prepare", "resolve", "votes", "benchmark", "finalize"), required=True)
    parser.add_argument("--pubchem-client", type=Path)
    args = parser.parse_args()
    if args.phase == "resolve":
        if not args.pubchem_client:
            parser.error("resolve requires the PubChem skill rest_request.py path")
        resolve(args.tasks, args.pubchem_client)
    elif args.phase == "benchmark":
        benchmark(args.tasks)
    else:
        for task in args.tasks:
            {"prepare": prepare, "votes": votes, "finalize": finalize}[args.phase](task)
