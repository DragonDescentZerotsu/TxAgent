"""Optimize canonical direct cards and joint indirect records over explicit grids."""

from __future__ import annotations
import argparse
import csv
import json
import math
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from data.processing.gold_labels.conditioned_benchmark import (
    split_path,
    tdc_split_path,
    tdc_manifest,
)
from data.processing.paths import REPO_ROOT
from predict.retrieval.selection import attach_uid_selection, indexed_manifests, export_selection
from predict.retrieval.assay_reranking.artifact_bundle import cache_task, resolve_cache_set
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from . import objectives, policies, pool

SCHEMA = "context_selection.v1"


def query_cohort(
    task: str, benchmark: str, subset: str, query_ids_file: Path | None = None
) -> tuple[dict, Path]:
    if benchmark not in {"gold_v1", "tdc"}:
        raise ValueError("benchmark must be gold_v1 or canonical tdc")
    if benchmark == "tdc":
        tdc_manifest(task, subset)
    path = (
        split_path(task, subset)
        if benchmark == "gold_v1"
        else tdc_split_path(task, subset)
    )
    queries = {
        str(row["benchmark_row_id"]): str(row["drug"]) for row in read_jsonl(path)
    }
    if query_ids_file:
        requested = query_ids_file.read_text().splitlines()
        if (
            not requested
            or len(set(requested)) != len(requested)
            or set(requested) - queries.keys()
        ):
            raise ValueError("query IDs must be unique members of the pinned split")
        queries = {query: queries[query] for query in requested}
    return (queries, path)


def _optimize(work):
    query, rows, profiles, kind, budget, diversity, sqrt_gated = work
    prepared = objectives._prepare_candidates(
        rows, direct=kind == "direct", budget=budget
    )
    result = []
    for profile in profiles:
        if kind == "direct":
            chosen, summary = objectives.select_direct_contexts(
                rows,
                profile=profile,
                budget=budget,
                prepared=prepared,
                sqrt_gated_assay=sqrt_gated,
            )
        else:
            chosen, summary = objectives.select_joint_indirect(
                rows,
                profile=profile,
                budget=budget,
                prepared=prepared,
                diversity_function=diversity,
                sqrt_gated_assay=sqrt_gated,
            )
        result.append((profile.name, query, chosen, summary))
    return result


def build(
    output_root: Path,
    tasks: Sequence[str],
    *,
    kind: str,
    benchmark: str = "gold_v1",
    subset: str = "valid_small",
    grid: str | Path | None = None,
    profile_names: Sequence[str] | None = None,
    direct_cache: Path | None = None,
    cache_bundle: Path | None = None,
    score_snapshot: Path | None = None,
    direct_manifest: Path | None = None,
    dili_score_cache: Path | None = None,
    semantic_extension: Path | None = None,
    semantic_snapshot: Path | None = None,
    prediction_filter: Path | None = None,
    query_ids_file: Path | None = None,
    budget: int | None = None,
    workers: int = 1,
    sqrt_gated_assay: bool = False,
    exclude_disjoint_violations: bool = False,
) -> Path:
    if kind not in {"direct", "indirect"} or workers < 1:
        raise ValueError("invalid search policy or worker count")
    if exclude_disjoint_violations and kind != "indirect":
        raise ValueError("Disjoint candidate filtering applies only to indirect pools")
    if output_root.exists():
        raise FileExistsError("search requires a fresh output identity")
    grid_name = grid or ("direct" if kind == "direct" else "sqrt48")
    grid_path = (
        Path(grid_name)
        if Path(grid_name).is_file()
        else policies.GRIDS / f"{grid_name}.json"
    )
    profiles, diversity = policies.load_grid(grid_path, kind=kind)
    if profile_names:
        available = {profile.name: profile for profile in profiles}
        if set(profile_names) - available.keys():
            raise ValueError("profile names are absent from the explicit JSON grid")
        profiles = [available[name] for name in dict.fromkeys(profile_names)]
    budget = (
        budget
        if budget is not None
        else policies.DIRECT_BUDGET
        if kind == "direct"
        else policies.INDIRECT_BUDGET
    )
    if budget < 1:
        raise ValueError("selection budget must be positive")
    cache_subset = "valid" if subset == "valid_small" else subset
    task_inputs = {}
    for task in tasks:
        queries, query_path = query_cohort(task, benchmark, subset, query_ids_file)
        if kind == "indirect":
            for profile in profiles:
                policies.validate_indirect(
                    profile,
                    task=task,
                    diversity=diversity,
                    dili_scores=dili_score_cache,
                    direct_manifest=direct_manifest,
                )
        if cache_bundle is not None or kind == "indirect" or benchmark == "tdc" and direct_cache is None:
            cache_bundle = resolve_cache_set(task, benchmark, cache_bundle)
        spec = cache_task(cache_bundle, task, benchmark) if cache_bundle else None
        index = direct_cache if kind == "direct" else None
        if index is None and spec:
            binding = (spec.get("selection", {}).get("direct_cache") if kind == "direct" else spec["retrieval"]["later"])
            index = (REPO_ROOT / binding).resolve() if binding else None
        if index is None:
            raise ValueError(
                "declare the canonical direct cache or pinned cache bundle"
            )
        scores, score_audit = pool.load_exact_scores(score_snapshot, task=task)
        if kind == "direct":
            if score_audit:
                snapshot = score_audit["manifest"]
                if (
                    snapshot.get("context_identity") != pool.CARD_IDENTITY
                    or snapshot.get("cache_index_sha256") != sha256_file(index)
                    or any(
                        (
                            not snapshot.get(key)
                            for key in (
                                "model",
                                "prompt_version",
                                "scoring_contract_version",
                            )
                        )
                    )
                ):
                    raise ValueError(
                        "canonical card scores require exact cache, model and prompt bindings; parent-grouped scores cannot be reused"
                    )
            rows, audit = pool.load_direct(
                index, task=task, subset=cache_subset, queries=queries, scores=scores
            )
        else:
            rows, audit = pool.load_indirect(
                index,
                task=task,
                subset=cache_subset,
                queries=queries,
                scores=scores if score_audit else None,
                semantic_extension=semantic_extension,
                semantic_snapshot=semantic_snapshot,
                dili_score_cache=dili_score_cache,
                prediction_filter=prediction_filter,
                cache_bundle=cache_bundle,
                exclude_disjoint_violations=exclude_disjoint_violations,
            )
        frozen = None
        if direct_manifest:
            frozen = validate_selection_manifest(direct_manifest)
            if (
                frozen["kind"],
                frozen["benchmark"],
                frozen["task_id"],
                frozen["subset"],
                frozen["benchmark_row_ids"],
            ) != ("direct", benchmark, task, subset, list(queries)):
                raise ValueError("frozen direct panel does not match the search cohort")
        levels = (
            ("L1",)
            if kind == "direct"
            else tuple(json.loads(index.read_text())["splits"][cache_subset]["levels"])
        )
        levels = tuple((level for level in levels if kind == "direct" or level != "L1"))
        manifests = indexed_manifests(index, cache_subset, levels)
        pins = {
            str(grid_path.resolve()): sha256_file(grid_path),
            str(query_path.resolve()): sha256_file(query_path),
            str(index.resolve()): sha256_file(index),
            **{str(p): sha256_file(p) for p in manifests.values()},
            **score_audit.get("pins", {}),
            **audit.get("semantics", {}).get("pinned_inputs", {}),
            **audit.get("prediction_filter", {}),
        }
        if cache_bundle:
            pins[str(cache_bundle.resolve())] = sha256_file(cache_bundle)
        if query_ids_file:
            pins[str(query_ids_file.resolve())] = sha256_file(query_ids_file)
        if direct_manifest:
            pins[str(direct_manifest.resolve())] = sha256_file(direct_manifest)
        task_inputs[task] = (queries, rows, audit, manifests, pins)
    output_root.mkdir(parents=True)
    entries = {profile.name: {} for profile in profiles}
    for task, (queries, rows, audit, manifests, pins) in task_inputs.items():
        work = [
            (query, rows[query], profiles, kind, budget, diversity, sqrt_gated_assay)
            for query in queries
        ]
        grouped = {profile.name: [] for profile in profiles}
        diagnostics = {profile.name: [] for profile in profiles}
        if workers == 1:
            results = map(_optimize, work)
            for group in results:
                for name, query, chosen, summary in group:
                    grouped[name].extend(
                        (
                            dict(
                                row,
                                benchmark_row_id=query,
                                level="L1" if kind == "direct" else row["level"],
                                context_id=row["item_id"] if kind == "direct" else "",
                                source_row_uid=""
                                if kind == "direct"
                                else row["item_id"],
                            )
                            for row in chosen
                        )
                    )
                    diagnostics[name].append(dict(benchmark_row_id=query, **summary))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for group in executor.map(_optimize, work):
                    for name, query, chosen, summary in group:
                        grouped[name].extend(
                            (
                                dict(
                                    row,
                                    benchmark_row_id=query,
                                    level="L1" if kind == "direct" else row["level"],
                                    context_id=row["item_id"]
                                    if kind == "direct"
                                    else "",
                                    source_row_uid=""
                                    if kind == "direct"
                                    else row["item_id"],
                                )
                                for row in chosen
                            )
                        )
                        diagnostics[name].append(
                            dict(benchmark_row_id=query, **summary)
                        )
        for profile in profiles:
            root = output_root / kind / profile.name / task
            root.mkdir(parents=True)
            selected = grouped[profile.name]
            filename = (
                "selected_contexts.tsv" if kind == "direct" else "selected_uids.tsv"
            )
            fields = (
                "benchmark_row_id",
                "selection_rank",
                "level",
                "context_id",
                "source_row_uid",
                "parent_id",
                "gold_label",
                "canonical_smiles",
                "morgan_similarity",
                "assay_transfer_score",
                "score_key",
                "semantic_bucket_id",
                "semantic_weight",
                "semantic_weight_source",
                "dili_tool_score",
            )
            with (root / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(selected)
            write_json_atomic(root / "diagnostics.json", diagnostics[profile.name])
            document = dict(
                schema_version=SCHEMA,
                status="complete",
                kind=kind,
                benchmark=benchmark,
                task_id=task,
                subset=subset,
                benchmark_row_ids=list(queries),
                budget=budget,
                profile=profile.name,
                objective=dict(
                    version="direct_full_molecule_cards.v1"
                    if kind == "direct"
                    else "joint_ledger_records.v1",
                    profile=asdict(profile),
                    diversity=diversity,
                    sqrt_gated_assay=sqrt_gated_assay,
                ),
                score_policy=dict(version="available_scores.v1"),
                inputs=dict(
                    pinned_inputs=pins,
                    pool=audit,
                    score_snapshot=str(score_snapshot.resolve())
                    if score_snapshot
                    else None,
                ),
                records=dict(
                    path=filename,
                    sha256=sha256_file(root / filename),
                    row_count=len(selected),
                ),
            )
            if task == "bbb_martins" and kind == "indirect":
                document["bbb_evidence_policy"] = audit["evidence_eligibility"]["policy"]
            attach_uid_selection(
                document, root, selected, queries=queries, cache_manifests=manifests
            )
            write_json_atomic(root / "manifest.json", document)
            entries[profile.name][task] = dict(
                path=str((root / "manifest.json").relative_to(output_root)),
                sha256=sha256_file(root / "manifest.json"),
            )
    result = output_root / "grid_manifest.json"
    write_json_atomic(
        result,
        dict(
            schema_version="selection_grid.v1",
            status="complete",
            kind=kind,
            benchmark=benchmark,
            subset=subset,
            grid=str(grid or kind),
            profiles=[
                dict(name=profile.name, task_manifests=entries[profile.name])
                for profile in profiles
            ],
            validation_evaluation="external_prediction_harness",
        ),
    )
    return result


def validate_selection_manifest(path: Path) -> dict:
    path = path.resolve()
    document = json.loads(path.read_text())
    if document.get("schema_version") != SCHEMA or document.get("status") != "complete":
        raise ValueError("selection manifest is not a complete canonical search")
    for name, digest in document["inputs"]["pinned_inputs"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("selection input changed")
    if document["inputs"]["pool"].get("dili_tool_scores"):
        source = document["inputs"]["pool"]["dili_tool_scores"]
        _, audit = pool.load_dili_scores(Path(source["manifest"]))
        if audit["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("DILI score provenance changed")
    records = (path.parent / document["records"]["path"]).resolve()
    if (
        not records.is_relative_to(path.parent)
        or sha256_file(records) != document["records"]["sha256"]
    ):
        raise ValueError("selected records changed")
    with records.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != document["records"]["row_count"] or set(
        (row["benchmark_row_id"] for row in rows)
    ) != set(document["benchmark_row_ids"]):
        raise ValueError("selection rows or cohort differ")
    uid = document["uid_selection"]
    child = (path.parent / uid["path"]).resolve()
    if not child.is_relative_to(path.parent) or sha256_file(child) != uid["sha256"]:
        raise ValueError("physical UID selection changed")
    return document


def compose(direct_manifest: Path, indirect_manifest: Path, output: Path,
            cache_bundle: Path | None = None) -> Path:
    if output.exists():
        raise FileExistsError(output)
    if cache_bundle is not None:
        from predict.harnesses.branches.flat import _load_preselected_contexts

        indirect = validate_selection_manifest(indirect_manifest)
        if indirect["kind"] != "indirect" or indirect["benchmark"] != "gold_v1" or output.name != "manifest.json":
            raise ValueError("Frozen direct composition requires Gold indirect selection and manifest.json")
        queries, _ = query_cohort(indirect["task_id"], "gold_v1", indirect["subset"])
        queries = {query: queries[query] for query in indirect["benchmark_row_ids"]}
        contexts, receipt = _load_preselected_contexts(
            direct_manifest, task=indirect["task_id"], queries=queries,
            budget=10, allow_subset=True, benchmark="gold_v1")
        if receipt.get("_selected_scores"):
            raise ValueError("Frozen direct composition requires cache-provided scores")
        spec = cache_task(cache_bundle, indirect["task_id"], "gold_v1")
        subset = "valid" if indirect["subset"] == "valid_small" else indirect["subset"]
        manifests = indexed_manifests(REPO_ROOT / spec["retrieval"]["L1"], subset, ("L1",))
        child = json.loads((indirect_manifest.parent / indirect["uid_selection"]["path"]).read_text())
        manifests.update({level: Path(pin["path"]) for level, pin in child["cache_manifests"].items()})
        rows = [dict(benchmark_row_id=query, level="L1", context_id=context)
                for query, panel in contexts.items() for context in panel]
        with (indirect_manifest.parent / indirect["records"]["path"]).open() as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
        return export_selection(
            output.parent, rows, task=indirect["task_id"], subset=subset,
            queries=queries, cache_manifests=manifests,
            limits={level: indirect["budget"] for level in manifests if level != "L1"},
            provenance=dict(selection_cohort=indirect["subset"], frozen_direct=receipt,
                            indirect=dict(path=str(indirect_manifest.resolve()), sha256=sha256_file(indirect_manifest)),
                            cache_bundle=dict(path=str(cache_bundle.resolve()), sha256=sha256_file(cache_bundle)),
                            score_policy="unchanged_cache_scores"))
    direct = validate_selection_manifest(direct_manifest)
    indirect = validate_selection_manifest(indirect_manifest)
    if (
        direct["kind"] != "direct"
        or indirect["kind"] != "indirect"
        or any(
            (
                direct[field] != indirect[field]
                for field in ("benchmark", "task_id", "subset", "benchmark_row_ids")
            )
        )
    ):
        raise ValueError("direct and indirect panels disagree on cohort or policy")
    queries, _ = query_cohort(direct["task_id"], direct["benchmark"], direct["subset"])
    queries = {query: queries[query] for query in direct["benchmark_row_ids"]}
    rows = []
    manifests = {}
    for path, document in ((direct_manifest, direct), (indirect_manifest, indirect)):
        with (path.parent / document["records"]["path"]).open() as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
        child = json.loads(
            (path.parent / document["uid_selection"]["path"]).read_text()
        )
        manifests.update(
            {
                level: Path(pin["path"])
                for level, pin in child["cache_manifests"].items()
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = dict(
        schema_version="combined_context_selection.v1",
        status="complete",
        benchmark=direct["benchmark"],
        task_id=direct["task_id"],
        subset=direct["subset"],
        benchmark_row_ids=direct["benchmark_row_ids"],
        direct_budget=direct["budget"],
        indirect_budget=indirect["budget"],
        direct=dict(
            path=str(direct_manifest.resolve()), sha256=sha256_file(direct_manifest)
        ),
        indirect=dict(
            path=str(indirect_manifest.resolve()), sha256=sha256_file(indirect_manifest)
        ),
    )
    attach_uid_selection(
        document, output.parent, rows, queries=queries, cache_manifests=manifests
    )
    write_json_atomic(output, document)
    return output


def rank(candidates: Path, output: Path) -> dict:
    """Freeze a validation winner only after every declared batch is complete."""
    from predict.harnesses.branches.manifests import read_record

    if output.exists():
        raise FileExistsError(output)
    declared = json.loads(candidates.read_text())
    expected = declared["benchmark_row_ids"]
    if not expected or len(expected) != len(set(expected)) or len({c["profile"] for c in declared["candidates"]}) != len(declared["candidates"]):
        raise ValueError("Validation cohort and candidate names must be nonempty and unique")
    ranked = []
    for candidate in declared["candidates"]:
        batch = Path(candidate["batch"])
        manifest = read_record(batch / "manifest.json")
        metrics = json.loads((batch / "metrics.json").read_text())
        predictions = list(read_jsonl(batch / "predictions.jsonl"))
        settings = manifest
        query_ids = list(manifest["queries"].values())
        if (settings["evaluation_subset"] != "valid" or settings["model"] != declared["model"]
                or settings["benchmark_release"] != declared["benchmark"]
                or set(query_ids) != set(expected) or len(query_ids) != len(expected)
                or any(metrics.get(key) != len(expected) for key in ("n_total", "n_evaluable", "n_successful"))
                or metrics.get("n_failed_runs") != 0 or len(predictions) != len(expected)
                or len({row["query_index"] for row in predictions}) != len(expected)
                or any(row.get("status") != "ok" or row.get("final_status") != "ok" or row.get("pred_label") not in (0, 1) for row in predictions)
                or {str(row["query_index"]) for row in predictions} != set(manifest["queries"])):
            raise ValueError(f"Incomplete or incompatible validation batch: {batch}")
        if any(not math.isfinite(metrics[key]) for key in ("macro_f1", "accuracy")):
            raise ValueError("Validation metrics must be finite")
        for row in predictions:
            response = json.loads((batch / "runs" / row["run_id"] / "final_reasoning_output.json").read_text())
            if (response.get("status") != "ok" or response["llm"].get("model") != declared["model"]
                    or (declared.get("provider") and response["llm"].get("provider") != declared["provider"])):
                raise ValueError(f"Validation response model/provider differs: {batch}/{row['run_id']}")
        ranked.append(dict(**candidate, macro_f1=metrics["macro_f1"], accuracy=metrics["accuracy"],
                           manifest_sha256=sha256_file(batch / "manifest.json"),
                           metrics_sha256=sha256_file(batch / "metrics.json"),
                           predictions_sha256=sha256_file(batch / "predictions.jsonl")))
    ranked.sort(key=lambda row: (-row["macro_f1"], -row["accuracy"], row["profile"]))
    if not ranked:
        raise ValueError("No validation candidates")
    result = dict(schema_version="validation_selection.v1", status="complete", frozen_at=datetime.now(timezone.utc).isoformat(), benchmark=declared["benchmark"],
                  benchmark_row_ids=expected, model=declared["model"], provider=declared.get("provider"), selection_cohort=declared["selection_cohort"],
                  rule="macro_f1 desc, accuracy desc, profile asc", candidates_sha256=sha256_file(candidates),
                  winner=ranked[0]["profile"], leaderboard=ranked)
    write_json_atomic(output, result)
    return result


def extend_semantics(task: str, cache_bundle: Path, output_root: Path) -> Path:
    """Project missing cache UIDs using existing reviewed memberships and weights."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from optimization.semantic_buckets.inputs import resolve_semantic_bucket_artifacts
    from optimization.semantic_buckets.ledger import project_retrieval_weights

    if output_root.exists():
        raise FileExistsError(output_root)
    semantics, _ = pool._semantic_ledger(task)
    missing = set()
    for benchmark in ("gold_v1", "tdc"):
        spec = cache_task(cache_bundle, task, benchmark)
        index = REPO_ROOT / spec["retrieval"]["later"]
        for subset in ("valid", "test"):
            for level, manifest in indexed_manifests(index, subset, ("L2", "L3", "L4", "L5", "L6")).items():
                with sqlite3.connect(f"file:{manifest.with_name('rankings.sqlite3').resolve()}?mode=ro&immutable=1", uri=True) as db:
                    missing.update((uid, level) for (uid,) in db.execute("SELECT DISTINCT item_id FROM rankings")
                                   if (uid, level) not in semantics)
    output_root.mkdir(parents=True)
    targets = output_root / "missing_retrieval_uids.parquet"
    pq.write_table(pa.table({"source_row_uid": [uid for uid, _ in sorted(missing)],
                             "level": [level for _, level in sorted(missing)]}), targets)
    source = resolve_semantic_bucket_artifacts(task, "CURRENT")
    project_retrieval_weights(source.central_ledger, source.semantic_bucket_weights, targets,
                              output_root / "projection", provisional_mean=False)
    return output_root / "projection" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for kind in ("direct", "indirect"):
        sub = commands.add_parser(kind)
        sub.add_argument("--tasks", nargs="+", required=True)
        sub.add_argument("--output-root", type=Path, required=True)
        sub.add_argument("--benchmark", choices=("gold_v1", "tdc"), default="gold_v1")
        sub.add_argument(
            "--subset", choices=("valid_small", "valid", "test"), default="valid_small"
        )
        sub.add_argument("--grid")
        sub.add_argument("--profiles", nargs="+")
        sub.add_argument("--budget", type=int)
        sub.add_argument("--workers", type=int, default=1)
        for name in (
            "direct-cache",
            "cache-bundle",
            "score-snapshot",
            "direct-manifest",
            "dili-score-cache",
            "semantic-extension",
            "semantic-snapshot",
            "prediction-filter",
            "query-ids-file",
        ):
            sub.add_argument("--" + name, type=Path)
        sub.add_argument("--sqrt-gated-assay", action="store_true")
        sub.add_argument("--exclude-disjoint-violations", action="store_true")
    sub = commands.add_parser("compose")
    sub.add_argument("--direct-manifest", type=Path, required=True)
    sub.add_argument("--indirect-manifest", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--cache-bundle", type=Path)
    sub = commands.add_parser("extend-semantics")
    sub.add_argument("--task", required=True)
    sub.add_argument("--cache-bundle", type=Path, required=True)
    sub.add_argument("--output-root", type=Path, required=True)
    sub = commands.add_parser("rank")
    sub.add_argument("--candidates", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("validate")
    sub.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    if command in ("direct", "indirect"):
        values["profile_names"] = values.pop("profiles")
        result = build(kind=command, **values)
    elif command == "compose":
        result = compose(**values)
    elif command == "rank":
        result = rank(**values)
    elif command == "extend-semantics":
        result = extend_semantics(**values)
    else:
        result = validate_selection_manifest(args.manifest)
    print(result)


def selected_scores(path: Path) -> dict:
    """Load scientific score annotations from a validated canonical panel."""
    document = json.loads(path.read_text())
    if document.get("schema_version") == "combined_context_selection.v1":
        result = {}
        for role in ("direct", "indirect"):
            pin = document[role]
            source = Path(pin["path"])
            if sha256_file(source) != pin["sha256"]:
                raise ValueError("combined panel input changed")
            for query, rows in selected_scores(source).items():
                result.setdefault(query, {}).update(rows)
        return result
    validated = validate_selection_manifest(path)
    scores = {query: {} for query in validated["benchmark_row_ids"]}
    with (path.parent / validated["records"]["path"]).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            score = (
                float(row["assay_transfer_score"])
                if row["assay_transfer_score"]
                else None
            )
            if (score is None) != (not row["score_key"]):
                raise ValueError("selected score and exact key disagree")
            identity = (
                row["level"],
                row["context_id"] if row["level"] == "L1" else row["source_row_uid"],
            )
            scores[row["benchmark_row_id"]][identity] = score
    return scores


if __name__ == "__main__":
    main()
