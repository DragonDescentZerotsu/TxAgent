"""Manual physical-UID inputs for the existing direct-label voting baselines."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from predict.retrieval.assay_reranking.ranked_uid_retrieval import _open
from predict.retrieval.selection import indexed_manifests, read_selection, resolve_selection, selection_query_ids
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


def add_manual_arguments(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uids", nargs="+")
    group.add_argument("--uids-file", type=Path)
    parser.add_argument("--uid-level", default="L1")
    parser.add_argument("--uid-task")
    parser.add_argument("--uid-subset", choices=("valid", "test"))
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--cache-index", type=Path)
    parser.add_argument("--indices", nargs="*")
    parser.add_argument("--uid-record-limit", type=int, default=10)


def run_manual(args) -> Path:
    """Vote once per frozen L1 context, preserving the supplied member UIDs."""
    task = args.uid_task or getattr(args, "task", None)
    if task is None and len(getattr(args, "tasks", [])) == 1:
        task = args.tasks[0]
    subset = args.uid_subset or getattr(args, "subset", None) or "valid"
    if not task or args.input_jsonl is None or args.cache_index is None:
        raise ValueError("manual evidence voting requires --uid-task/--task, --input-jsonl and --cache-index")
    if args.uid_level != "L1":
        raise ValueError("direct-label voting requires labeled L1 context members")
    source = read_jsonl(args.input_jsonl)
    if args.indices is None and args.uids_file:
        args.indices = selection_query_ids(args.uids_file)
    if args.indices is not None:
        source = [row for row in source if str(row["benchmark_row_id"]) in args.indices]
        if {str(row["benchmark_row_id"]) for row in source} != set(args.indices):
            raise ValueError("manual query IDs are absent from the declared input")
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in source}
    targets = {str(row["benchmark_row_id"]): int(row["Y"]) for row in source}
    if not queries or len(queries) != len(source):
        raise ValueError("manual voting requires unique query identities")
    if args.uids_file:
        rows, receipt = read_selection(args.uids_file, task=task, subset=subset, queries=queries)
    else:
        if len(queries) != 1:
            raise ValueError("inline --uids requires exactly one query")
        rows = [dict(benchmark_row_id=next(iter(queries)), level="L1", source_row_uid=uid) for uid in args.uids]
        receipt = dict(uids=list(args.uids))
    if any(row["level"] != "L1" for row in rows):
        raise ValueError("indirect evidence has no frozen direct label to vote")
    manifests = indexed_manifests(args.cache_index, subset, ["L1"])
    k = getattr(args, "k", 10)
    cap = max(k) if isinstance(k, list) else k
    selected = resolve_selection(rows, task=task, subset=subset, queries=queries,
                                 cache_manifests=manifests, molecule_limit=cap, l1_limit=args.uid_record_limit)
    document = json.loads(manifests["L1"].read_text())
    predictions = []
    with _open(manifests["L1"].with_name(document["database"])) as db:
        for query_id in queries:
            members = [row for row in selected if row["benchmark_row_id"] == query_id]
            contexts = list(dict.fromkeys(row["context_id"] for row in members))
            labeled = [(context, int(db.execute("SELECT gold_label FROM contexts WHERE context_id=?", (context,)).fetchone()[0]))
                       for context in contexts]
            if any(vote not in (0, 1) for _, vote in labeled):
                if getattr(args, "ambiguous_card_policy", None) != "abstain":
                    raise ValueError("ambiguous frozen cards require an explicit abstention policy")
                labeled = [(context, vote) for context, vote in labeled if vote in (0, 1)]
            for count in (k if isinstance(k, list) else [k]):
                chosen = labeled[:count]
                if not chosen:
                    raise ValueError(f"no binary-labeled manual evidence for {query_id}")
                fraction = sum(vote for _, vote in chosen) / len(chosen)
                used_contexts = [context for context, _ in chosen]
                predictions.append(dict(task=task, subset=subset, benchmark_row_id=query_id, target=targets[query_id],
                                        k=count, effective_k=len(chosen), prediction=int(fraction >= .5),
                                        positive_vote_fraction=fraction, context_ids=used_contexts,
                                        source_row_uids=[row["source_row_uid"] for row in members if row["context_id"] in used_contexts]))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(output / "predictions.jsonl", predictions)
    metrics = []
    for count in (k if isinstance(k, list) else [k]):
        group = [row for row in predictions if row["k"] == count]
        truth, predicted = [row["target"] for row in group], [row["prediction"] for row in group]
        metrics.append(dict(k=count, n=len(group), accuracy=float(accuracy_score(truth, predicted)),
                            macro_f1=float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0))))
    write_json_atomic(output / "metrics.json", dict(results=metrics) if isinstance(k, list) else metrics[0])
    write_json_atomic(output / "manifest.json", dict(schema_version="manual_uid_direct_votes.v1", status="complete",
                                                     task=task, subset=subset, selection=receipt,
                                                     cache_manifests={level: dict(path=str(path), sha256=sha256_file(path)) for level, path in manifests.items()},
                                                     query_path=str(args.input_jsonl.resolve()), query_sha256=sha256_file(args.input_jsonl),
                                                     cache_index=str(args.cache_index.resolve()), cache_index_sha256=sha256_file(args.cache_index)))
    return output
