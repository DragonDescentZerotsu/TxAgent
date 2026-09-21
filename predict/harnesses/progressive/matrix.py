"""Run a validation matrix through one L1-first, exact-request inference pool.

The progressive runner prepares evidence and yields validated stage-call jobs.
This module shares identical jobs, resumes their dependent chain iterators, and
persists request receipts and matrix metrics. It never selects evidence itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import csv
from dataclasses import replace
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import queue
import shutil
import threading
import time
import traceback

from data.processing.llm_api import DEFAULT_ENV_FILE
from predict.harnesses.branches.matrix import (
    sample_provider_loads,
    top_up_provider_config,
)
from predict.harnesses.progressive import runner
from predict.harnesses.progressive.retrieval_cache import (
    DEFAULT_CACHE_BUNDLE,
    INDIRECT_MORGAN_SEMANTIC_V2_CACHE_BUNDLE,
    INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE,
    load_cache_policy,
    load_candidates,
)
from predict.harnesses.progressive.prompt import (
    prompt_asset_manifest,
    prompt_assets,
    split_prompt_version,
)
from predict.api_client.pool import (
    DEFAULT_PROVIDER_POOL_CONFIG,
    build_provider_pool,
    load_provider_pool_config,
    preflight_provider_models,  # compatibility hook for historical test/launcher patches
    preflight_sglang_tokenized_completion,
    primary_capacity,
    select_healthy_providers,
)
from predict.utils.json import canonical_json_bytes, sha256_file, write_json_atomic
from predict.traces.io import DEFAULT_TRACE_ROOT


DEFAULT_PROVIDER_CONFIG = DEFAULT_PROVIDER_POOL_CONFIG
DEFAULT_RESULTS_ROOT = Path("outputs/paper/assay_transfer_harness/joseph")
METHOD_NAMES = {
    "morgan": "morgan",
    "morgan-contrastive": "morgan",
    "assay-transfer": "assay_transfer",
    "assay-transfer-within-morgan": "assay_transfer",
    "assay-transfer-contrastive": "assay_transfer",
    "joint": "joint",
    "semantic-lap": "lap",
    "semantic-weighted": "weighted",
    "morgan-parent-control": "morgan_parent_control",
    "morgan-parent-semantic": "morgan_parent_semantic",
    "morgan-parent-llm-semantic": "morgan_parent_llm_semantic",
}
PRIOR_DIRECTORIES = {"cached": "with_query_prior", "none": "no_query_prior"}
RESULTS_LEDGER_FIELDS = (
    "study", "method", "query_prior", "run_id", "status", "support_mode",
    "prompt_version", "harness_version", "reranking",
    "morgan_primary_parent_width", "l1_molecules", "l1_min_contrast",
    "molecule_description_mode", "indirect_level", "task", "evaluation_subset", "level",
    "n_total", "n_successful", "n_failed_runs", "macro_f1", "accuracy",
    "path", "run_json_sha256", "metrics_path", "metrics_sha256",
    "diagnostics_manifest_sha256", "finished_at",
)


def _method_name(mode, molecule_description_mode):
    method = METHOD_NAMES[mode]
    return (
        method
        if molecule_description_mode == 'none'
        else f'{method}_quotient_{molecule_description_mode}'
    )


def _molecule_description_identity(mode, prompt_version=None, cache_version='v1'):
    if mode == 'none':
        return {'mode': 'none'}
    artifact = runner.MOLECULE_DESCRIPTION_ARTIFACTS[cache_version]
    identity = {
        'mode': mode,
        'cache_version': cache_version,
        'column': runner.MOLECULE_DESCRIPTION_COLUMNS[mode],
        'path': str(artifact['path'].resolve()),
        'sha256': artifact['sha256'],
    }
    if prompt_version:
        identity['missing_policy'] = prompt_assets(prompt_version)[
            'settings'
        ]['molecule_metadata_contract']['missing_policy']
    return identity


def _study_name(value):
    if not value or not value.replace("_", "").isalnum() or not value[0].isalpha():
        raise argparse.ArgumentTypeError("study must use lowercase letters, digits, and underscores")
    return value.lower()


def _condition_root(
    options, batch_root, key, mode, prior, k, minimum, molecule_description_mode,
    retrieval_width=None, indirect_level=None,
):
    if not options.study:
        return batch_root / "conditions" / key
    method = _method_name(mode, molecule_description_mode)
    storage_root = options.staging_root or options.results_root
    parent = storage_root / options.study / method / PRIOR_DIRECTORIES[prior]
    for receipt in parent.glob("*/run.json"):
        saved = json.loads(receipt.read_text())
        if saved.get("batch_id") == options.batch_id and saved.get("condition_key") == key:
            return receipt.parent
    suffix = options.run_date
    width = f"_w{retrieval_width}" if retrieval_width is not None else ""
    name = (
        f"l{indirect_level}_v1_{suffix}"
        if indirect_level else (
            f"k{k}_m{minimum}{width}_{options.reasoning_phase}_{suffix}"
            if options.harness_version in runner.FULL_FLAT_PROGRESSIVE_HARNESSES
            else f"k{k}_m{minimum}{width}_{suffix}"
        )
    )
    path = parent / name
    serial = 1
    while path.exists():
        serial += 1
        path = parent / f"{name}-{serial:02d}"
    return path


def _update_raw_run(args, status, **fields):
    path = Path(args.output_root) / "run.json"
    if not path.is_file():
        return
    document = json.loads(path.read_text())
    write_json_atomic(path, {**document, **fields, "status": status, "updated_at": runner._now()})
    if document.get("study") and not document.get("staging_root"):
        _refresh_results_catalog(Path(document["results_root"]))


def _copytree_atomic(source: Path, destination: Path, *, ignore=()) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace published artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination.parent / f".{destination.name}.publishing-{os.getpid()}"
    serial = 1
    while candidate.exists():
        serial += 1
        candidate = destination.parent / (
            f".{destination.name}.publishing-{os.getpid()}-{serial:02d}"
        )
    shutil.copytree(source, candidate, ignore=shutil.ignore_patterns(*ignore))
    os.replace(candidate, destination)


def _validate_staged_run(path: Path) -> dict:
    run = json.loads((path / "run.json").read_text())
    experiment = json.loads((path / "experiment_manifest.json").read_text())
    diagnostics = json.loads((path / "diagnostics_manifest.json").read_text())
    if run.get("status") != "complete" or run.get("metric_status") != "complete":
        raise ValueError(f"Staged run is not complete: {path}")
    if diagnostics.get("status") != "complete":
        raise ValueError(f"Staged diagnostics are not complete: {path}")
    required = {
        "visible_evidence.tsv",
        "neighborhood_label_mix.per_query.tsv",
        "neighborhood_label_mix.summary.tsv",
        "reasoning_reference_coverage.per_query.tsv",
        "reasoning_reference_coverage.summary.tsv",
        "diagnostics_manifest.json",
        "report.md",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise ValueError(f"Staged run is missing required diagnostics: {missing}")
    outputs = []
    expected = 0
    output_levels = {}
    for task, indices in experiment["evaluation_indices_by_task"].items():
        output_level = (
            {"bbb_martins": 5, "bioavailability_ma": 6}[task]
            if run.get("reasoning_phase") == "indirect-update"
            else int(run.get("indirect_level") or 1)
        )
        output_levels[task] = output_level
        expected += len(indices)
        outputs.extend((path / task / "queries").glob(
            f"query_idx*/levels/level_{output_level}/output.json"
        ))
    if len(outputs) != expected:
        raise ValueError(
            f"Staged run has {len(outputs)} of {expected} terminal outputs "
            f"at {output_levels}: {path}"
        )
    incomplete = [
        output for output in outputs
        if json.loads(output.read_text()).get("status")
        not in {"ok", "carried_forward", "reused", "reused_none"}
    ]
    if incomplete:
        raise ValueError(
            f"Staged run has incomplete terminal outputs: {incomplete[0]}"
        )
    receipt = {
        "run_json_sha256": sha256_file(path / "run.json"),
        "diagnostics_manifest_sha256": sha256_file(
            path / "diagnostics_manifest.json"
        ),
        "terminal_outputs": len(outputs),
        "output_levels_by_task": output_levels,
    }
    for level in set(output_levels.values()):
        receipt[f"l{level}_outputs"] = sum(
            1 for output in outputs
            if output.parent.name == f"level_{level}"
        )
    return receipt


def _validate_staged_live_run(path: Path) -> dict:
    documents = sorted(path.glob("*/*/run.json"))
    if not documents:
        raise ValueError(f"Staged live run has no task receipts: {path}")
    incomplete = [
        document for document in documents
        if json.loads(document.read_text()).get("status") != "complete"
    ]
    if incomplete:
        raise ValueError(f"Staged live run is not complete: {incomplete[0]}")
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(document)))
    return {"task_runs": len(documents), "run_receipts_sha256": digest.hexdigest()}


def _publish_staged_study(options, batch_root: Path) -> None:
    status = json.loads((batch_root / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError(f"Staged batch is not complete: {batch_root}")
    runs = json.loads((batch_root / "runs.json").read_text())["runs"]
    receipts = []
    live_receipts = []
    for row in runs:
        source = Path(row["staging_path"])
        destination = Path(row["path"])
        receipt = _validate_staged_run(source)
        if not destination.exists():
            _copytree_atomic(source, destination)
        if _validate_staged_run(destination) != receipt:
            raise ValueError(f"Published run failed hash validation: {destination}")
        receipts.append({
            "staging_path": str(source),
            "canonical_path": str(destination),
            **receipt,
        })
        live_source = (
            options.trace_root / row["study"] / row["method"]
            / PRIOR_DIRECTORIES[row["query_prior"]] / row["run_id"]
        )
        live_destination = (
            options.canonical_trace_root / live_source.relative_to(options.trace_root)
        )
        live_receipt = _validate_staged_live_run(live_source)
        if not live_destination.exists():
            _copytree_atomic(live_source, live_destination)
        if _validate_staged_live_run(live_destination) != live_receipt:
            raise ValueError(
                f"Published live run failed hash validation: {live_destination}"
            )
        live_receipts.append({
            "staging_path": str(live_source),
            "canonical_path": str(live_destination),
            **live_receipt,
        })
    canonical_batch = options.results_root / "_batches" / options.batch_id
    if not canonical_batch.exists():
        _copytree_atomic(batch_root, canonical_batch, ignore=("launcher.lock",))
    if (
        sha256_file(canonical_batch / "matrix.json")
        != sha256_file(batch_root / "matrix.json")
        or sha256_file(canonical_batch / "status.json")
        != sha256_file(batch_root / "status.json")
    ):
        raise ValueError(f"Published batch failed hash validation: {canonical_batch}")
    publication = {
        "schema_version": "progressive_staged_publication.v1",
        "status": "complete",
        "staging_root": str(options.staging_root),
        "canonical_results_root": str(options.results_root),
        "staging_batch_root": str(batch_root),
        "canonical_batch_root": str(canonical_batch),
        "published_at": runner._now(),
        "runs": receipts,
        "live_runs": live_receipts,
        "matrix_sha256": sha256_file(canonical_batch / "matrix.json"),
        "status_sha256": sha256_file(canonical_batch / "status.json"),
    }
    write_json_atomic(canonical_batch / "publication.json", publication)
    _refresh_results_catalog(options.results_root)
    from predict.live import _refresh_catalog

    _refresh_catalog(options.canonical_trace_root)


def _refresh_results_catalog(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".results_catalog.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, ledger = [], []
        for path in sorted(root.glob("*/*/*/*/run.json")):
            document = json.loads(path.read_text())
            if not str(document.get("schema_version", "")).startswith(
                ("progressive_study_run.v", "organized_study_run.v")
            ):
                continue
            run_path = path.parent.relative_to(root).as_posix()
            diagnostics_path = path.parent / "diagnostics_manifest.json"
            rows.append({
                key: document.get(key)
                for key in (
                    "study", "method", "run_id", "batch_id", "status",
                    "prompt_version", "harness_version", "reranking",
                    "query_prior", "l1_molecules", "l1_min_contrast",
                    "morgan_primary_parent_width", "molecule_description_mode",
                    "indirect_level", "support_mode", "metric_status", "tasks",
                    "evaluation_subset",
                )
            } | {"path": run_path})
            base = {
                key: document.get(key)
                for key in RESULTS_LEDGER_FIELDS
                if key not in {
                    "task", "level", "n_total", "n_successful", "n_failed_runs",
                    "macro_f1", "accuracy", "path", "run_json_sha256",
                    "metrics_path", "metrics_sha256",
                    "diagnostics_manifest_sha256",
                }
            } | {
                "path": run_path,
                "run_json_sha256": sha256_file(path),
                "diagnostics_manifest_sha256": (
                    sha256_file(diagnostics_path)
                    if diagnostics_path.is_file()
                    else document.get("diagnostics_manifest_sha256")
                ),
            }
            for task in document.get("tasks") or [None]:
                metric_paths = sorted(
                    path.parent.glob(f"{task}/levels/level_*/metrics.json")
                ) if task else []
                if task and not metric_paths:
                    metric_paths = sorted(path.parent.glob(f"{task}/*/metrics.json"))
                if not metric_paths:
                    ledger.append(base | {"task": task})
                    continue
                for metrics_path in metric_paths:
                    metrics = json.loads(metrics_path.read_text())
                    ledger.append(base | {
                        "task": task,
                        "level": metrics.get("level"),
                        "n_total": metrics.get("n_total"),
                        "n_successful": metrics.get("n_successful"),
                        "n_failed_runs": metrics.get("n_failed_runs"),
                        "macro_f1": metrics.get("macro_f1"),
                        "accuracy": metrics.get("accuracy"),
                        "metrics_path": metrics_path.relative_to(path.parent).as_posix(),
                        "metrics_sha256": sha256_file(metrics_path),
                    })
        ledger_path = root / "results_ledger.tsv"
        candidate = root / f".results_ledger.tsv.tmp-{os.getpid()}"
        with candidate.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, RESULTS_LEDGER_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(ledger)
        os.replace(candidate, ledger_path)
        write_json_atomic(root / "catalog.json", {
            "schema_version": "progressive_results_catalog.v3",
            "legacy_included": False,
            "results_ledger": {
                "path": ledger_path.name,
                "rows": len(ledger),
                "sha256": sha256_file(ledger_path),
            },
            "runs": rows,
        })


def endpoint_allocations(parallelism, config):
    priority = min(provider.priority for provider in config.providers)
    providers = [
        provider for provider in config.providers if provider.priority == priority
    ]
    capacity = sum(provider.max_inflight for provider in providers)
    if not 1 <= parallelism <= capacity:
        raise ValueError(f'Provider capacity is {capacity}, not {parallelism}')
    allocations = [0] * len(providers)
    for _ in range(parallelism):
        index = min((i for i, provider in enumerate(providers)
                     if allocations[i] < provider.max_inflight), key=allocations.__getitem__)
        allocations[index] += 1
    return [(provider, allocations[index]) for index, provider in enumerate(providers)
            if allocations[index]]


def visibility_suffix(prompt_version):
    """Return only the model-visibility modifiers needed in live run names."""
    _, modifiers = split_prompt_version(prompt_version)
    visible = [name for name in modifiers
               if name in {'no_scores', 'no_smiles', 'no_query_smiles'}]
    return f'_{"_".join(visible)}' if visible else ''


def request_key(job, settings):
    payload = dict(
        messages=job['messages'], task=job['task'], level=job['level'],
        reasoning_transport=job.get('reasoning_transport'),
        reasoning_grammar=job.get('reasoning_grammar'),
        stage_kind=job.get('stage_kind', 'prediction'),
        settings=settings)
    if job.get('request_namespace'):
        payload['request_namespace'] = job['request_namespace']
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class AttemptClient:
    """Count and persist every wire attempt against its shared stage budget."""

    def __init__(self, pool, max_attempts=10):
        self.pool = pool
        self.max_attempts = max_attempts
        self.local = threading.local()

    def chat_json(self, messages, **transport):
        entry = self.local.entry
        if entry['attempts'] >= entry['attempt_limit']:
            raise RuntimeError('Shared stage attempt budget exhausted')
        entry['attempts'] += 1
        attempt = dict(number=entry['attempts'], started_at=runner._now(),
                       request_sha256=hashlib.sha256(canonical_json_bytes(
                           {'messages': messages, **transport}
                       )).hexdigest())
        entry['history'].append(attempt)
        self.save(entry)
        try:
            response = self.pool.chat_json(messages, **transport)
            attempt.update(status='ok', execution=response.get('execution_provider'),
                           usage=response.get('usage'))
            return response
        except Exception as exc:
            attempt.update(status='error', error=str(exc),
                           providers=getattr(exc, 'attempts', []))
            raise
        finally:
            attempt['finished_at'] = runner._now()
            self.save(entry)

    @staticmethod
    def save(entry):
        write_json_atomic(entry['ledger'], dict(attempts=entry['attempts'], history=entry['history']))


def schedule(chains, client, settings, root, parallelism=2048, retry_delay=5.0,
             reuse_settings=(), reuse_sources=()):
    """Drive chain generators; deduplicate before submitting bounded stage jobs."""
    root = Path(root)
    cache = root / 'responses'
    cache.mkdir(parents=True, exist_ok=True)
    ready, delayed, pending, futures = deque(), [], {}, {}
    results = []
    counts = dict(unique_calls=0, reused_stages=0, peak_inflight=0)
    serial = 0
    last_report = 0.0

    def advance(chain, response=None):
        nonlocal serial
        try:
            job = chain.send(response)
        except StopIteration as done:
            results.append(done.value)
            return
        except Exception as exc:
            results.append(dict(status='error', error=str(exc)))
            return
        key = request_key(job, settings)
        path = cache / key[:2] / f'{key}.json'
        reusable = None
        sources = ((settings, cache), *reuse_sources,
                   *((value, cache) for value in reuse_settings))
        for cached_settings, cached_root in sources:
            cached_key = request_key(job, cached_settings)
            cached_path = cached_root / cached_key[:2] / f'{cached_key}.json'
            if not cached_path.is_file():
                continue
            saved = json.loads(cached_path.read_text())
            usage = saved.get('response', {}).get('usage') or {}
            if (saved.get('key') != cached_key
                    or not runner.structured_response_is_valid(saved.get('response', {}))
                    or (cached_settings != settings
                        and usage.get('completion_tokens', cached_settings['max_tokens'])
                        >= cached_settings['max_tokens'])):
                raise ValueError(f'Invalid shared response: {cached_path}')
            reusable = cached_key, cached_path, saved['response']
            break
        if reusable:
            cached_key, cached_path, response = reusable
            counts['reused_stages'] += 1
            advance(chain, dict(response, inference_reused=True,
                                inference_reuse=dict(key=cached_key, source=str(cached_path))))
        elif key in pending:
            pending[key]['consumers'].append(chain)
        else:
            ledger = path.with_suffix('.attempts.json')
            prior = json.loads(ledger.read_text()) if ledger.is_file() else {}
            prior_attempts = prior.get('attempts', 0)
            entry = dict(job=job, key=key, path=path, ledger=ledger, consumers=[chain],
                         attempts=prior_attempts,
                         attempt_limit=prior_attempts + client.max_attempts,
                         history=prior.get('history', []),
                         sequence=serial)
            serial += 1
            pending[key] = entry
            ready.append(entry)

    def execute(entry):
        client.local.entry = entry
        return entry['job']['execute']()

    streaming = isinstance(chains, queue.Queue)
    producer_done = not streaming
    # A bounded producer queue lets inference start before preparation finishes.
    if not streaming:
        for chain in chains:
            advance(chain)
    ready = deque(sorted(ready, key=lambda e: (e['job']['level'] != 1, e['sequence'])))
    with ThreadPoolExecutor(max_workers=parallelism) as workers:
        while not producer_done or ready or futures or delayed:
            if streaming and not producer_done and len(ready) < parallelism:
                for _ in range(64):
                    try:
                        chain = chains.get_nowait()
                    except queue.Empty:
                        break
                    if chain is None:
                        producer_done = True
                        break
                    advance(chain)
                ready = deque(sorted(ready, key=lambda e: (e['job']['level'] != 1, e['sequence'])))
            now = time.monotonic()
            while delayed and delayed[0][0] <= now:
                _, _, entry = heapq.heappop(delayed)
                ready.appendleft(entry) if entry['job']['level'] == 1 else ready.append(entry)
            while ready and len(futures) < parallelism:
                entry = ready.popleft()
                futures[workers.submit(execute, entry)] = entry
                counts['peak_inflight'] = max(counts['peak_inflight'], len(futures))
            completed, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED) if futures else (set(), set())
            if not futures:
                time.sleep(min(0.5, max(0, delayed[0][0]-time.monotonic())) if delayed else .05)
            for future in completed:
                entry = futures.pop(future)
                error = None
                try:
                    response = future.result()
                    if not runner.structured_response_is_valid(response):
                        raise ValueError('Model response failed progressive validation')
                except Exception as exc:
                    error = str(exc)
                if error is not None:
                    if 0 < entry['attempts'] < entry['attempt_limit']:
                        delay = min(300, retry_delay * 2**min(entry['attempts'], 6))
                        heapq.heappush(delayed, (time.monotonic()+delay*random.uniform(.8,1.2),
                                                entry['sequence'], entry))
                        continue
                    for chain in entry['consumers']:
                        chain.close()
                        results.append(dict(status='error', key=entry['key'], error=error))
                    del pending[entry['key']]
                    continue
                write_json_atomic(entry['path'], dict(key=entry['key'], settings=settings,
                    response=response, completed_at=runner._now()))
                counts['unique_calls'] += 1
                del pending[entry['key']]
                for index, chain in enumerate(entry['consumers']):
                    counts['reused_stages'] += int(index > 0)
                    advance(chain, dict(response, inference_reused=index > 0,
                        inference_reuse=dict(key=entry['key'], source=str(entry['path']))))
            if time.monotonic()-last_report > 30 or (producer_done and not (ready or futures or delayed)):
                last_report = time.monotonic()
                snapshot = dict(counts, finished_chains=len(results),
                    preparation_finished=producer_done,
                    failed_chains=sum(r.get('status') != 'ok' for r in results),
                    inflight=len(futures), ready=len(ready), delayed=len(delayed),
                    providers=client.pool.snapshot(), updated_at=runner._now())
                write_json_atomic(root/'progress.json', snapshot)
                print(json.dumps(snapshot), flush=True)
    write_json_atomic(root/'completion.json', dict(counts, chains=results,
        status='complete' if all(r.get('status') == 'ok' for r in results) else 'incomplete'))
    return results


def _reuse_inference_sources(paths, current_settings):
    """Load exact-request response caches from completed matrix batches."""
    sources, receipts = [], []
    for source in paths:
        matrix_path = source / 'matrix.json' if source.is_dir() else source
        document = json.loads(matrix_path.read_text())
        settings = document.get('settings')
        if not isinstance(settings, dict):
            raise ValueError(f'Reuse matrix has no inference settings: {matrix_path}')
        for key in ('model', 'temperature', 'request_extra_body', 'response_format'):
            if settings.get(key) != current_settings.get(key):
                raise ValueError(f'Incompatible reuse setting {key}: {matrix_path}')
        response_root = matrix_path.parent / 'inference' / 'responses'
        if not response_root.is_dir():
            raise ValueError(f'Reuse response cache is missing: {response_root}')
        sources.append((settings, response_root))
        receipts.append({
            'matrix': str(matrix_path.resolve()),
            'matrix_sha256': sha256_file(matrix_path),
            'responses': str(response_root.resolve()),
        })
    return tuple(sources), receipts


def provider_client(parallelism, max_tokens, timeout_s, provider_config):
    specs = tuple(
        replace(spec, max_inflight=slots)
        for spec, slots in endpoint_allocations(parallelism, provider_config)
    )
    config = replace(provider_config, providers=specs)

    pool = build_provider_pool(
        config,
        env_file=DEFAULT_ENV_FILE,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=0.0,
        tool_service_url='http://127.0.0.1:8765',
        enable_group_tools=False,
        max_tool_rounds=0,
        reasoning_effort='',
        enable_thinking=False,
        transport_max_retries=0,
    )
    return AttemptClient(pool)


def main(argv=None):
    raw_argv = list(os.sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        help='Legacy explicit matrix root; omit when using --study.')
    parser.add_argument('--results-root', type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        '--staging-root', type=Path,
        help=(
            'Node-local work root for organized runs; publish only a fully '
            'validated batch to --results-root.'
        ),
    )
    parser.add_argument('--study', type=_study_name,
                        help='Write study/method/query-prior/kK_mM_date leaves under --results-root.')
    parser.add_argument('--batch-id', default='',
                        help='Shared matrix/inference identity; generated when omitted.')
    parser.add_argument('--tasks', nargs='+', choices=('bbb_martins','bioavailability_ma'),
                        default=['bbb_martins','bioavailability_ma'])
    parser.add_argument('--reranking-modes', nargs='+', choices=(
        'morgan','morgan-contrastive','assay-transfer','joint',
        'assay-transfer-within-morgan','assay-transfer-contrastive',
        'semantic-lap','semantic-weighted',
        'morgan-parent-control','morgan-parent-semantic',
        'morgan-parent-llm-semantic'),
                        default=['morgan','assay-transfer'])
    parser.add_argument('--harness-version',
                        choices=('reranked-progressive-v2','reranked-progressive-v3',
                                 'reranked-progressive-v4',
                                 'reranked-progressive-l1-context-v1',
                                 'reranked-progressive-l1-context-l2-v1',
                                 'reranked-progressive-l1-context-l2-weighted-v1',
                                 'reranked-progressive-l1-context-l2-morgan-bucket-v1',
                                 'reranked-progressive-indirect-only-v1',
                                 'reranked-progressive-indirect-only-v2',
                                 *runner.FULL_FLAT_PROGRESSIVE_HARNESSES),
                        default='reranked-progressive-v2')
    parser.add_argument('--record-pools', nargs='+', choices=('all','assay-transfer-trained'),
                        default=['all'])
    parser.add_argument('--records-per-level', nargs='+', type=int, default=[10,25,50])
    parser.add_argument('--l1-molecules', nargs='+', type=int, default=[10])
    parser.add_argument('--l1-min-contrasts', nargs='+', type=int, default=[3])
    parser.add_argument('--l2-molecules', type=int, default=10)
    parser.add_argument('--l2-records-per-molecule', type=int, default=10)
    parser.add_argument('--parallelism', type=int, default=None,
                        help='Required global outstanding-request budget for inference.')
    parser.add_argument('--provider-pool-config', type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument('--trace-root', type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument('--execution-mode', choices=('live', 'throughput'), default='throughput')
    parser.add_argument('--publish-review-traces', action='store_true',
                        help='Publish the first three trace samples while throughput continues.')
    parser.add_argument('--prompt-version', default='reranked_progressive_v8')
    parser.add_argument(
        '--variant', action='append', nargs=3, metavar=('MODE', 'CACHE', 'PROMPT'),
        help='Repeat MODE CACHE PROMPT triples to run an explicit non-Cartesian matrix.',
    )
    parser.add_argument(
        '--molecule-description-modes', nargs='+',
        choices=('none', *runner.MOLECULE_DESCRIPTION_COLUMNS), default=['none'],
        help='Factorial axis for hash-pinned cached Quotient molecule metadata.',
    )
    parser.add_argument(
        '--molecule-description-cache-version',
        choices=tuple(runner.MOLECULE_DESCRIPTION_ARTIFACTS),
        default='v1',
        help='Select the immutable Quotient description artifact.',
    )
    parser.add_argument('--query-prior-modes', nargs='+', choices=('cached', 'none'),
                        default=['cached'])
    parser.add_argument(
        '--gold-label-version', choices=('current', 'v1'), default='current',
    )
    parser.add_argument('--evaluation-subset', choices=('valid', 'test'), default='valid')
    parser.add_argument(
        '--allow-test-inference', action='store_true',
        help='Explicitly authorize non-prepare-only inference on the formal test split.',
    )
    parser.add_argument('--replicates', type=int, default=1,
                        help='Independent samples of every selected condition in one queue.')
    parser.add_argument(
        '--exclude-task-condition', action='append', nargs=3, default=[],
        metavar=('TASK', 'QUERY_PRIOR', 'DESCRIPTION_MODE'),
        help='Omit one task/prior/description cell; repeat for multiple cells.',
    )
    parser.add_argument(
        '--max-inflight-per-endpoint', type=int,
        help='Cap each selected provider before allocating the global request pool.',
    )
    parser.add_argument('--target-total-load-per-endpoint', type=int)
    parser.add_argument('--load-samples', type=int, default=6)
    parser.add_argument('--load-sample-interval-s', type=float, default=1.0)
    parser.add_argument('--prior-root', type=Path, default=runner.DEFAULT_SINGLE_CACHE_ROOT)
    parser.add_argument('--max-level', type=int, default=0)
    parser.add_argument('--reasoning-phase', choices=('l1', 'indirect-update'))
    parser.add_argument('--l1-prior-run', type=Path)
    parser.add_argument(
        '--require-complete-l1-prior', action='store_true',
        help='Disable default partial L1 reuse for a full-flat indirect run.',
    )
    indirect_group = parser.add_mutually_exclusive_group()
    indirect_group.add_argument('--indirect-level', type=int, choices=(2, 3, 4))
    indirect_group.add_argument('--indirect-levels', nargs='+', type=int, choices=(2, 3, 4))
    parser.add_argument('--max-tokens', type=int, default=524_288)
    parser.add_argument('--request-timeout-s', type=int, default=3600)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument('--assay-transfer-cache', type=Path)
    cache_group.add_argument('--assay-transfer-caches', nargs='+', type=Path)
    parser.add_argument('--reuse-settings-from', action='append', type=Path, default=[])
    parser.add_argument(
        '--reuse-inference-from', action='append', type=Path, default=[],
        help='Reuse only exact rendered requests found in a completed matrix batch.',
    )
    parser.add_argument('--prepared-from', type=Path)
    parser.add_argument('--reprepare-from-cache', action='store_true')
    parser.add_argument('--pilot-queries', type=int, default=3)
    parser.add_argument('--continue-after-pilot', action='store_true')
    parser.add_argument('--live-run-id', default='', help=argparse.SUPPRESS)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--rolling-from', type=Path,
                        help='Resume prepared conditions with a bounded producer, without a pilot')
    options = parser.parse_args(raw_argv)
    if options.target_total_load_per_endpoint is not None and options.parallelism is not None:
        parser.error('top-up allocation derives parallelism; do not also pass --parallelism')
    if options.target_total_load_per_endpoint is not None and options.prepare_only:
        parser.error('top-up allocation is an immediate pre-launch operation')
    if options.load_samples < 1 or options.load_sample_interval_s < 0:
        parser.error('load samples must be positive and sample interval non-negative')
    if (
        options.evaluation_subset == 'test'
        and not options.prepare_only
        and not options.allow_test_inference
    ):
        parser.error('Formal-test inference requires --allow-test-inference')
    if options.variant:
        conflicting = {
            '--reranking-modes', '--assay-transfer-cache',
            '--assay-transfer-caches', '--prompt-version',
        }
        if any(value.split('=', 1)[0] in conflicting for value in raw_argv):
            parser.error('--variant replaces reranking, cache, and prompt matrix axes')
        if (not options.study
                or options.harness_version != 'reranked-progressive-l1-context-v1'):
            parser.error('--variant requires an organized L1-context study')
        allowed_modes = {
            'morgan', 'morgan-contrastive', 'assay-transfer',
            'assay-transfer-within-morgan', 'assay-transfer-contrastive',
        }
        if any(mode not in allowed_modes for mode, _, _ in options.variant):
            parser.error('--variant contains an unsupported L1-context mode')
        options.variant = [
            {
                'mode': mode,
                'cache': Path(cache).resolve(),
                'prompt_version': prompt,
            }
            for mode, cache, prompt in options.variant
        ]
        identities = [(row['mode'], row['cache']) for row in options.variant]
        if len(identities) != len(set(identities)):
            parser.error('--variant mode/cache pairs must be unique')
        options.reranking_modes = list(dict.fromkeys(
            row['mode'] for row in options.variant
        ))
    if bool(options.output_root) == bool(options.study):
        parser.error('choose exactly one of --output-root or --study')
    options.run_date = time.strftime('%Y%m%d_%H%M')
    options.batch_id = options.batch_id or f'{options.study}_{options.run_date}'
    options.results_root = options.results_root.resolve()
    options.staging_root = (
        options.staging_root.resolve() if options.staging_root else None
    )
    if options.staging_root and not options.study:
        parser.error('--staging-root requires --study')
    if options.staging_root == options.results_root:
        parser.error('--staging-root must differ from --results-root')
    molecule_l2_v1 = options.harness_version == 'reranked-progressive-l1-context-l2-morgan-bucket-v1'
    molecule_l2 = molecule_l2_v1
    indirect_only = options.harness_version in runner.INDIRECT_HARNESSES
    full_flat = options.harness_version in runner.FULL_FLAT_PROGRESSIVE_HARNESSES
    full_flat_contrasts = (
        [1] if options.harness_version == runner.FULL_FLAT_PROGRESSIVE_HARNESS
        else [1]
    )
    if full_flat:
        options.prompt_version = runner.FULL_FLAT_PROGRESSIVE_HARNESSES[
            options.harness_version
        ]
        if options.reasoning_phase is None:
            parser.error('full-flat-progressive-v1 requires --reasoning-phase')
        if options.reasoning_phase == 'l1' and options.l1_prior_run is not None:
            parser.error('L1 does not accept --l1-prior-run')
        if options.reasoning_phase == 'l1' and options.require_complete_l1_prior:
            parser.error('L1 does not accept --require-complete-l1-prior')
        if options.require_complete_l1_prior and options.l1_prior_run is None:
            parser.error('--require-complete-l1-prior requires --l1-prior-run')
        options.l1_prior_run = (
            options.l1_prior_run.resolve() if options.l1_prior_run else None
        )
    elif options.require_complete_l1_prior:
        parser.error('--require-complete-l1-prior requires full-flat-progressive')
    if molecule_l2_v1:
        options.records_per_level = [options.l2_molecules * options.l2_records_per_molecule]
    if options.study and (
            len(options.record_pools) != 1
            or len(options.records_per_level) != 1):
        parser.error('organized study runs require one pool and record cap per batch')
    options.provider_pool_config = options.provider_pool_config.resolve()
    options.canonical_trace_root = options.trace_root.resolve()
    options.trace_root = (
        options.staging_root / 'live'
        if options.staging_root else options.canonical_trace_root
    )
    options.prior_root = options.prior_root.resolve()
    options.provider_config = load_provider_pool_config(options.provider_pool_config)
    if options.max_inflight_per_endpoint is not None:
        if options.max_inflight_per_endpoint < 1:
            parser.error('--max-inflight-per-endpoint must be positive')
        options.provider_config = replace(
            options.provider_config,
            providers=tuple(
                replace(
                    provider,
                    max_inflight=min(
                        provider.max_inflight, options.max_inflight_per_endpoint
                    ),
                )
                for provider in options.provider_config.providers
            ),
        )
    options.exclude_task_conditions = [
        tuple(row) for row in options.exclude_task_condition
    ]
    if len(options.exclude_task_conditions) != len(set(options.exclude_task_conditions)):
        parser.error('--exclude-task-condition entries must be unique')
    for task, prior, description in options.exclude_task_conditions:
        if (
            task not in options.tasks
            or prior not in options.query_prior_modes
            or description not in options.molecule_description_modes
        ):
            parser.error(
                '--exclude-task-condition must reference selected task, prior, '
                'and description axes'
            )
    models = {provider.model for provider in options.provider_config.providers}
    if len(models) != 1:
        parser.error('--provider-pool-config must use one exact model across endpoints')
    options.model = next(iter(models))
    if not options.prepare_only and any(
        (provider.request_extra_body or {}).get('chat_template_kwargs', {}).get(
            'reasoning_effort'
        ) != 'high'
        for provider in options.provider_config.providers
    ):
        parser.error('Inference requires reasoning_effort=high on every provider')
    options.requested_parallelism = options.parallelism
    options.endpoint_selection = None
    options.load_receipt = []
    options.top_up_allocations = []
    if options.prepare_only:
        options.parallelism = options.parallelism or 1
    elif options.target_total_load_per_endpoint is not None:
        options.parallelism = primary_capacity(options.provider_config)
    else:
        if options.parallelism is None:
            parser.error('full-batch inference requires explicit --parallelism')
        options.endpoint_selection = select_healthy_providers(
            options.provider_config, options.parallelism
        )
        options.provider_config = options.endpoint_selection.config
        options.parallelism = options.endpoint_selection.effective_parallelism
    options.live_command = [os.sys.executable, '-m', 'predict.harnesses.progressive.matrix', *raw_argv]
    if options.study and not any(
            value == '--batch-id' or value.startswith('--batch-id=') for value in raw_argv):
        options.live_command.extend(['--batch-id', options.batch_id])
    if (not 1 <= options.parallelism <= primary_capacity(options.provider_config)
            or min(options.records_per_level) < 1 or min(options.l1_molecules) < 1
            or len(options.l1_molecules) != len(set(options.l1_molecules))
            or min(options.l1_min_contrasts) < 0
            or len(options.l1_min_contrasts) != len(set(options.l1_min_contrasts))
            or len(options.molecule_description_modes)
            != len(set(options.molecule_description_modes))
            or options.replicates < 1
            or options.pilot_queries < 1
            or min(options.max_tokens, options.request_timeout_s) < 1):
        parser.error(f'Require 1–{primary_capacity(options.provider_config)} slots, unique positive L1 molecule counts, and positive record caps, timeout, and pilot count')
    if options.rolling_from and options.execution_mode != 'throughput':
        parser.error('--rolling-from is an uninterrupted resume; use --execution-mode throughput')
    if options.rolling_from and (
        options.replicates != 1 or options.exclude_task_conditions
    ):
        parser.error('rolling resume does not support replicate or exclusion axes')
    if options.rolling_from and options.molecule_description_modes != ['none']:
        parser.error('rolling resume requires --molecule-description-modes none')
    if options.harness_version == 'reranked-progressive-v4' and (
            options.reranking_modes != ['joint'] or options.l1_molecules != [10]):
        parser.error('Reranked Progressive v4 requires only joint mode with 10 L1 panel slots')
    if options.harness_version == 'reranked-progressive-l1-context-v1' and (
            not options.reranking_modes
            or any(mode not in {
                'morgan', 'morgan-contrastive', 'assay-transfer-within-morgan',
                'assay-transfer-contrastive'
            } for mode in options.reranking_modes)
            or options.max_level not in {0, 1}):
        parser.error(
            'L1 context matrix requires supported context modes and max level 1'
        )
    if full_flat and (
        options.reranking_modes != ['assay-transfer-contrastive']
        or options.record_pools != ['all']
        or options.records_per_level != [10]
        or options.l1_molecules != [10]
        or options.l1_min_contrasts != full_flat_contrasts
        or options.query_prior_modes != ['cached']
        or options.molecule_description_modes != ['none']
        or options.max_tokens != 131_072
        or options.execution_mode != 'throughput'
        or options.continue_after_pilot
    ):
        parser.error(
            'Full-flat progressive requires assay-transfer-contrastive, all/10, '
            f'K=10, M={full_flat_contrasts}, cached prior, no descriptions, '
            '131072 tokens, and throughput'
        )
    if options.harness_version == 'reranked-progressive-l1-context-l2-v1' and (
            set(options.reranking_modes) != {'morgan', 'semantic-lap'}
            or options.record_pools != ['all']
            or options.records_per_level != [12]
            or options.l1_molecules != [10]
            or options.max_level not in {0, 2}):
        parser.error(
            'Context-L2 matrix requires Morgan + semantic-lap, all records, K=10, and 12 L2 records'
        )
    if options.harness_version == 'reranked-progressive-l1-context-l2-weighted-v1' and (
            set(options.reranking_modes) != {'morgan', 'semantic-weighted'}
            or options.record_pools != ['all']
            or options.records_per_level != [12]
            or options.l1_molecules != [10]
            or options.max_level not in {0, 2}):
        parser.error(
            'Weighted context-L2 matrix requires Morgan + semantic-weighted, all records, K=10, and 12 L2 records'
        )
    if molecule_l2 and (
            set(options.reranking_modes) != {
                'morgan-parent-control', 'morgan-parent-semantic'
            }
            or options.record_pools != ['all']
            or options.l1_molecules != [10]
            or (options.l2_molecules, options.l2_records_per_molecule)
            != (10, 10)
            or options.max_level not in {0, 2}):
        parser.error(
            'Molecule-first L2 requires both matched parent modes, all records, '
            'L1 K=10, the configured parent ceiling, and 10 records per parent'
        )
    requested_indirect_levels = (
        options.indirect_levels
        or ([options.indirect_level] if options.indirect_level is not None else [])
    )
    options.indirect_levels = requested_indirect_levels
    indirect_modes = (
        {'morgan-parent-semantic', 'morgan-parent-llm-semantic'}
        if options.harness_version == runner.INDIRECT_FILTER_HARNESS
        else {'morgan-parent-control', 'morgan-parent-semantic'}
    )
    if indirect_only:
        if (
            not requested_indirect_levels
            or len(requested_indirect_levels) != len(set(requested_indirect_levels))
            or set(options.reranking_modes) != indirect_modes
            or options.record_pools != ['all']
            or options.records_per_level != [25]
            or options.query_prior_modes != ['none']
            or options.molecule_description_modes != ['none']
            or options.l1_min_contrasts != [0]
            or options.prompt_version != 'reranked_progressive_l1_context_v4'
            or options.max_tokens != 65_536
            or options.execution_mode != 'throughput'
        ):
            parser.error(
                'Indirect-only matrix requires L2/L3/L4 and its matched Morgan-parent arms, '
                'all/25 records, no prior/descriptions, V4, M=0, high providers, '
                '64K output, and throughput mode'
            )
        options.max_level = max(requested_indirect_levels)
    elif requested_indirect_levels:
        parser.error('--indirect-level(s) requires the indirect-only harness')
    if (options.harness_version not in {
            'reranked-progressive-l1-context-v1',
            'reranked-progressive-l1-context-l2-v1',
            'reranked-progressive-l1-context-l2-weighted-v1',
            'reranked-progressive-l1-context-l2-morgan-bucket-v1',
            *runner.INDIRECT_HARNESSES}
            and options.l1_min_contrasts != (
                full_flat_contrasts if full_flat else [3]
            )):
        parser.error('--l1-min-contrasts applies only to the L1 context matrix')
    nonempty_description_modes = [
        mode for mode in options.molecule_description_modes if mode != 'none'
    ]
    if nonempty_description_modes:
        allowed_prompts = runner.MOLECULE_DESCRIPTION_PROMPTS.get(options.harness_version)
        explicit_prompt = any(
            value == '--prompt-version' or value.startswith('--prompt-version=')
            for value in raw_argv
        )
        if allowed_prompts is None:
            parser.error(
                '--molecule-description-modes is unsupported by this harness'
            )
        if (
            not explicit_prompt
            or split_prompt_version(options.prompt_version)[0] not in allowed_prompts
        ):
            parser.error(
                'non-none molecule descriptions require an explicit supported prompt version'
            )
    if options.rolling_from and (
        options.prompt_version != 'reranked_progressive_v8'
        or options.query_prior_modes != ['cached']
        or options.l1_molecules != [10]
        or options.max_level
    ):
        parser.error('rolling resume uses its frozen prompt, prior, and level settings')
    options.prompt_versions = {
        prior: (
            options.prompt_version
            if prior == 'cached'
            else f'{options.prompt_version}_no_query_prior'
        )
        for prior in options.query_prior_modes
    }
    for row in options.variant or []:
        row['prompt_versions'] = {
            prior: (
                row['prompt_version']
                if prior == 'cached'
                else f"{row['prompt_version']}_no_query_prior"
            )
            for prior in options.query_prior_modes
        }
    prompt_versions = (
        [
            (prior, version)
            for row in options.variant
            for prior, version in row['prompt_versions'].items()
        ]
        if options.variant else list(options.prompt_versions.items())
    )
    for prior, version in prompt_versions:
        settings = prompt_assets(version)['settings']
        if options.study and not settings.get('reasoning_reference_contract'):
            parser.error(
                f'organized study prompt {version} lacks a reasoning-reference contract'
            )
        expected_prior = prior == 'cached'
        if (
            expected_prior and settings.get('query_prior') is False
            or not expected_prior and settings.get('query_prior') is not False
        ):
            parser.error(f'{version} is incompatible with query prior mode {prior}')
        if (not indirect_only and settings.get('max_level')
                and options.max_level != settings['max_level']):
            parser.error(f'{version} requires --max-level {settings["max_level"]}')
        metadata_contract = settings.get('molecule_metadata_contract')
        if any(
            not metadata_contract or mode not in metadata_contract.get('modes', ())
            for mode in nonempty_description_modes
        ):
            parser.error(f'{version} does not support the requested molecule descriptions')
    root = (options.output_root.resolve() if options.output_root else
            (options.staging_root or options.results_root)
            / '_batches' / options.batch_id)
    options.assay_transfer_caches = list(dict.fromkeys(
        row['cache'] for row in options.variant
    )) if options.variant else [
        path.resolve() for path in (
            options.assay_transfer_caches
            or [options.assay_transfer_cache or (
                INDIRECT_MORGAN_SEMANTIC_V3_CACHE_BUNDLE
                if options.harness_version == runner.INDIRECT_FILTER_HARNESS
                else DEFAULT_CACHE_BUNDLE
            )]
        )
    ]
    if len(options.assay_transfer_caches) != len(set(options.assay_transfer_caches)):
        parser.error('retrieval cache bundles must be unique')
    options.assay_transfer_cache = options.assay_transfer_caches[0]
    if options.rolling_from and len(options.assay_transfer_caches) != 1:
        parser.error('--rolling-from accepts exactly one retrieval cache bundle')
    options.reuse_settings_from = [path.resolve() for path in options.reuse_settings_from]
    options.reuse_inference_from = [path.resolve() for path in options.reuse_inference_from]
    if options.prepared_from:
        options.prepared_from = options.prepared_from.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.rolling_from:
            return run_rolling(options, root)
        result = run_matrix(options, root)
    if result == 0 and options.staging_root and not options.prepare_only:
        _publish_staged_study(options, root)
    return result


def run_rolling(options, root):
    """Stream saved BBB chains, then bounded fresh task batches into one pool.

    Reuse valid completed responses from the frozen, smaller output ceiling.
    """
    source = options.rolling_from.resolve()
    frozen = json.loads((source/'matrix.json').read_text())
    frozen_settings = frozen['settings']
    settings = dict(frozen_settings, max_tokens=options.max_tokens,
        runner_code_sha256=sha256_file(Path(runner.__file__)),
        matrix_code_sha256=sha256_file(Path(__file__)),
        client_code_sha256=sha256_file(
            Path(__file__).resolve().parents[3] / 'predict/api_client/client.py'
        ))
    current_prompt = prompt_asset_manifest('reranked_progressive_v8')
    old_rendering, new_rendering = deepcopy(settings['prompt']), deepcopy(current_prompt)
    # Prepared retrieval is frozen, while prompt prose, cards and state assembly
    # remain byte-identical and are still checked below.
    for asset in (old_rendering,new_rendering):
        for name in ('cache_matched.py', 'level_selection.py', 'prompt.py'):
            asset['assembly_files_sha256'].pop(name,None)
    rendering_changed = old_rendering != new_rendering
    if rendering_changed and not (options.reprepare_from_cache or options.prepared_from):
        raise ValueError('Frozen prompt rendering contract changed')
    if options.reprepare_from_cache or options.prepared_from:
        settings['prompt'] = current_prompt
    for key, actual in dict(validation_code_sha256=sha256_file(
            Path(runner.__file__).resolve().parents[2]/'llm_io/response.py')).items():
        if settings[key] != actual:
            raise ValueError(f'Frozen inference contract changed: {key}')
    if (frozen['modes'] != options.reranking_modes or frozen['pools'] != options.record_pools
            or not set(options.records_per_level).issubset(frozen['caps'])
            or frozen['limit'] != options.limit):
        raise ValueError('Rolling resume shape differs from prepared matrix')
    allocations = endpoint_allocations(options.parallelism, options.provider_config)
    compatible_settings = [frozen_settings]
    reuse_receipts = []
    for path in options.reuse_settings_from:
        document = json.loads(path.read_text())
        candidate = document.get('settings')
        if not isinstance(candidate, dict):
            raise ValueError(f'Reuse receipt has no settings: {path}')
        for key in ('model', 'temperature', 'request_extra_body', 'response_format'):
            if candidate.get(key) != settings.get(key):
                raise ValueError(f'Incompatible reuse setting {key}: {path}')
        if canonical_json_bytes(candidate) not in {
                canonical_json_bytes(value) for value in compatible_settings}:
            compatible_settings.append(candidate)
        reuse_receipts.append(dict(path=str(path), sha256=sha256_file(path)))
    client = provider_client(
        options.parallelism,
        settings['max_tokens'],
        options.request_timeout_s,
        options.provider_config,
    )
    incoming = queue.Queue(maxsize=128)
    conditions, errors, live_runs = [], [], []
    receipt = dict(source=str(source), settings=settings, tasks=options.tasks,
        prepared_from=str(options.prepared_from) if options.prepared_from else None,
        assay_transfer_cache=str(options.assay_transfer_cache),
        reuse_settings_receipts=reuse_receipts,
        request_timeout_s=options.request_timeout_s,
        prompt_rendering_changed=rendering_changed,
        current_prompt_assets=current_prompt,
        parallelism=options.parallelism,
        requested_parallelism=options.requested_parallelism,
        endpoint_allocations={spec.name: slots for spec, slots in allocations},
        provider_pool_config=str(options.provider_pool_config),
        endpoint_selection=options.endpoint_selection.public_dict(),
        pilot='skipped_by_throughput_mode',preparation_batch_queries=16,
        descriptor_queue_limit=128, scheduler_sha256=sha256_file(Path(__file__)),
        request_namespace='bounded_output_with_frozen_cache_reuse',
        compatible_reuse_settings=compatible_settings, created_at=runner._now())
    write_json_atomic(root/'rolling.json', receipt)

    def produce():
        try:
            for task in options.tasks:
                saved = bool(options.prepared_from) or (
                    task in frozen['final_levels'] and not options.reprepare_from_cache)
                variants = []
                for mode in options.reranking_modes:
                    for pool in options.record_pools:
                        for cap in options.records_per_level:
                            name = f'{mode}_{pool}_records{cap}'
                            if options.prepared_from:
                                path = options.prepared_from/'conditions'/task/name
                            else:
                                path = source/'conditions'/name if saved else root/'conditions'/task/name
                            args = runner.parse_args(['--tasks',task,'--reranking',mode,
                                '--record_pool',pool,'--records-per-level',str(cap),
                                '--assay-transfer-cache',str(options.assay_transfer_cache),
                                '--l1-molecules','10','--l1-records-per-molecule','10',
                                '--query-prior','cached','--allow-frozen-l1-vote-scores',
                                '--evaluation-subset', options.evaluation_subset,
                                *(['--allow-test-inference'] if options.allow_test_inference else []),
                                '--prepare-only','--output-root',str(path),'--limit',str(options.limit),
                                '--max-tokens',str(options.max_tokens),
                                '--base-url',options.provider_config.providers[0].base_url,
                                '--model',options.provider_config.providers[0].model,
                                '--provider-pool-config',str(options.provider_pool_config),
                                '--execution-mode',options.execution_mode])
                            args.trace_root = str(options.trace_root)
                            from predict.live import PILOT_SIZE, create_run, update_run
                            method = f'progressive_{mode}_{pool}_records{cap}'
                            run_dir = create_run(
                                root=options.trace_root,
                                dataset=task,
                                method=method,
                                command=options.live_command,
                                requested_id=options.live_run_id,
                                execution_mode='throughput',
                                metadata={
                                    'harness': 'progressive',
                                    'harness_version': args.harness_version,
                                    'prompt_version': args.assay_transfer_prompt_version,
                                    'retrieval_cache_bundle': str(options.assay_transfer_cache),
                                    'evaluation_subset': options.evaluation_subset,
                                    'pilot_size': PILOT_SIZE,
                                    'matrix_output_root': str(root),
                                },
                            )
                            args.live_method = method
                            args.live_run_dirs = {task: run_dir}
                            args.live_run_ids = {task: run_dir.name}
                            live_runs.append(run_dir)
                            update_run(
                                run_dir,
                                resume_command=[
                                    *options.live_command,
                                    '--live-run-id',
                                    run_dir.name,
                                ],
                            )
                            records = runner._validate_inputs(args)
                            indices = list(range(len(records[task])))
                            if options.limit:
                                indices = indices[:options.limit]
                            if saved and not options.prepared_from:
                                manifest = json.loads((path/'experiment_manifest.json').read_text())
                                if (manifest['evaluation_indices_by_task'][task] != indices
                                        or manifest['inputs'][task]['input_sha256'] != sha256_file(Path(manifest['inputs'][task]['input_jsonl']))
                                        or manifest['reranking'] != mode or manifest['record_pool'] != pool):
                                    raise ValueError('Prepared condition input/selection mismatch')
                            conditions.append((args, records, {task:indices}))
                            variants.append((args, indices))
                for start in range(0,len(indices),16):
                    for args, indices in variants:
                        batch = indices[start:start+16]
                        if saved:
                            queries = []
                            for index in batch:
                                directory = runner._query_dir(Path(args.output_root),task,index)
                                manifest = json.loads((directory/'prepared_manifest.json').read_text())
                                if (manifest['status'] != 'ok' or manifest.get('task') != task
                                        or manifest.get('query_index') != index
                                        or not manifest['tool_prefetch_complete']
                                        or manifest['prompt_version'] != 'reranked_progressive_v8'
                                        or manifest['context_limit'] != 10
                                        or manifest['record_limit_per_context_level'] != 10
                                        or manifest['indirect_record_limit_per_level'] != args.indirect_record_limit_per_level):
                                    raise ValueError(f'Invalid prepared query: {directory}')
                                queries.append(runner.PreparedQuery(task,index,directory))
                            for q in queries:
                                incoming.put(runner.query_steps(args,q,client))
                        else:
                            batch_args = deepcopy(args)
                            batch_args.indices = batch
                            batch_args.output_root = str(Path(args.output_root)/'batches'/f'{start:05d}')
                            def collect(queries, records, selected):
                                for q in queries:
                                    link = runner._query_dir(Path(args.output_root),task,q.index)
                                    link.parent.mkdir(parents=True,exist_ok=True)
                                    if not link.exists():
                                        link.symlink_to(q.query_dir.resolve(),target_is_directory=True)
                                    elif link.resolve() != q.query_dir.resolve():
                                        raise ValueError('Batch query output link changed')
                                    incoming.put(runner.query_steps(batch_args,q,client))
                            runner.run(batch_args,prepared_callback=collect)
                    write_json_atomic(root/'preparation.json',dict(task=task,prepared_through=start+len(batch),
                        total=len(indices),updated_at=runner._now()))
        except BaseException as exc:
            errors.append(str(exc))
            write_json_atomic(root/'preparation_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        finally:
            incoming.put(None)

    # Also lock the reused output/cache root: two launchers must never own it.
    with (source/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        producer = threading.Thread(target=produce,daemon=True)
        producer.start()
        results = schedule(incoming,client,settings,source/'inference',parallelism=options.parallelism,
                           reuse_settings=tuple(compatible_settings))
        producer.join()
    if errors:
        from predict.live import update_run

        for run_dir in live_runs:
            update_run(run_dir, status='failed', finished_at=runner._now())
        raise RuntimeError('; '.join(errors))
    summarize(conditions,root)
    failed = sum(r.get('status') != 'ok' for r in results)
    write_json_atomic(root/'status.json',dict(status='complete' if not failed else 'incomplete',
        logical_chains=len(results),failed_chains=failed,finished_at=runner._now()))
    from predict.live import update_run

    for run_dir in live_runs:
        update_run(
            run_dir,
            status='failed' if failed else 'complete',
            finished_at=runner._now(),
        )
    return int(bool(failed))


def run_matrix(options, root):
    molecule_l2_v1 = options.harness_version == (
        'reranked-progressive-l1-context-l2-morgan-bucket-v1')
    molecule_l2 = molecule_l2_v1
    full_flat = options.harness_version in runner.FULL_FLAT_PROGRESSIVE_HARNESSES
    all_prompt_versions = list(dict.fromkeys(
        version
        for row in options.variant or [{'prompt_versions': options.prompt_versions}]
        for version in row['prompt_versions'].values()
    ))
    prompt_manifests = {
        version: prompt_asset_manifest(version) for version in all_prompt_versions
    }
    transports = {
        prompt_assets(version)['settings'].get('reasoning_transport')
        for version in all_prompt_versions
    }
    if len(transports) != 1:
        raise ValueError('Matrix prompt variants must use one reasoning transport')
    reasoning_transport = next(iter(transports))
    if reasoning_transport not in {None, 'sglang_tokenized_completion.v1'}:
        raise ValueError(f'Unsupported reasoning transport: {reasoning_transport!r}')
    settings = dict(model=options.model, temperature=0.0, max_tokens=options.max_tokens,
                    request_extra_body=dict(options.provider_config.providers[0].request_extra_body or {}),
                    response_format=(None if reasoning_transport else {'type':'json_object'}),
                    reasoning_transport=reasoning_transport,
                    prompt=(next(iter(prompt_manifests.values())) if len(prompt_manifests) == 1
                            else {'by_prompt_version': prompt_manifests}),
                    runner_code_sha256=sha256_file(Path(runner.__file__)),
                    validation_code_sha256=sha256_file(Path(runner.__file__).resolve().parents[2]/'llm_io/response.py'),
                    matrix_code_sha256=sha256_file(Path(__file__)),
                    client_code_sha256=sha256_file(
                        Path(__file__).resolve().parents[3]
                        / 'predict/api_client/client.py'
                    ))
    reuse_sources, reuse_receipts = _reuse_inference_sources(
        options.reuse_inference_from, settings
    )
    endpoint_selection = (
        options.endpoint_selection.public_dict()
        if options.endpoint_selection is not None else None
    )
    endpoint_preflight = {
        'models': (
            endpoint_selection
        ),
        'tokenized_reasoning': (
            preflight_sglang_tokenized_completion(options.provider_config)
            if reasoning_transport and options.endpoint_selection is not None
            else None
        ),
    }
    cache_variants = []
    for cache_path in options.assay_transfer_caches:
        cache_mode = next((
            row['mode'] for row in options.variant or []
            if row['cache'] == cache_path
        ), options.reranking_modes[0])
        policy = load_cache_policy(
            cache_path,
            options.tasks[0],
            'valid',
            cache_mode,
            options.max_level,
        )
        cache_document_path = Path(
            policy.get('cache_manifest') or policy['cache_index']
        )
        cache_version = json.loads(cache_document_path.read_text())
        retrieval_width = (
            25
            if full_flat
            else (
                cache_version.get('morgan_primary_parent_width')
                or cache_version.get('capacity')
            )
        )
        if len(options.assay_transfer_caches) > 1 and not isinstance(retrieval_width, int):
            raise ValueError(
                f'Multi-cache matrix requires morgan_primary_parent_width: {cache_path}'
            )
        cache_variants.append((cache_path, retrieval_width))
    if len({width for _, width in cache_variants}) != len(cache_variants):
        raise ValueError('Multi-cache matrix retrieval widths must be unique')
    variant_lookup = {
        (row['cache'], row['mode']): row for row in options.variant or []
    }
    variant_mode_counts = Counter(row['mode'] for row in options.variant or [])
    invariant = dict(settings=settings, harness_version=options.harness_version,
        study=options.study, batch_id=options.batch_id,
        results_root=str(options.results_root) if options.study else None,
        staging_root=str(options.staging_root) if options.staging_root else None,
        modes=options.reranking_modes, pools=options.record_pools,
        molecule_description_modes=options.molecule_description_modes,
        molecule_description_artifact=(
            {
                'cache_version': options.molecule_description_cache_version,
                'path': str(runner.MOLECULE_DESCRIPTION_ARTIFACTS[
                    options.molecule_description_cache_version
                ]['path'].resolve()),
                'sha256': runner.MOLECULE_DESCRIPTION_ARTIFACTS[
                    options.molecule_description_cache_version
                ]['sha256'],
                'columns_by_mode': runner.MOLECULE_DESCRIPTION_COLUMNS,
            }
            if any(mode != 'none' for mode in options.molecule_description_modes)
            else None
        ),
        query_prior_modes=options.query_prior_modes,
        replicates=options.replicates,
        excluded_task_conditions=[
            {
                'task': task,
                'query_prior': prior,
                'molecule_description_mode': description,
            }
            for task, prior, description in options.exclude_task_conditions
        ],
        prompt_versions=(None if options.variant else options.prompt_versions),
        variants=[
            {
                'mode': row['mode'],
                'cache': str(row['cache']),
                'prompt_versions': row['prompt_versions'],
            }
            for row in options.variant or []
        ],
        max_level=options.max_level,
        reasoning_phase=options.reasoning_phase,
        l1_prior_run=str(options.l1_prior_run) if options.l1_prior_run else None,
        gold_label_version=options.gold_label_version,
        prior_root=str(options.prior_root),
        assay_transfer_cache=(
            str(options.assay_transfer_cache)
            if len(options.assay_transfer_caches) == 1 else None
        ),
        assay_transfer_caches=[
            {
                'bundle': str(cache_path),
                'morgan_primary_parent_width': retrieval_width,
            }
            for cache_path, retrieval_width in cache_variants
        ],
        request_timeout_s=options.request_timeout_s,
        caps=options.records_per_level, limit=options.limit,
        provider_pool=options.provider_config.public_dict(),
        provider_candidate_inventory={
            'path': str(options.provider_pool_config),
            'sha256': sha256_file(options.provider_pool_config),
        },
        parallelism=options.parallelism,
        requested_parallelism=options.requested_parallelism,
        max_inflight_per_endpoint=options.max_inflight_per_endpoint,
        target_total_load_per_endpoint=options.target_total_load_per_endpoint,
        load_samples=options.load_receipt,
        top_up_allocations=options.top_up_allocations,
        execution_mode=options.execution_mode,
        publish_review_traces=options.publish_review_traces,
        l1_molecules=options.l1_molecules,
        l1_min_contrasts=options.l1_min_contrasts,
        l1_records_per_molecule=10,
        l2_molecules=options.l2_molecules,
        l2_records_per_molecule=options.l2_records_per_molecule,
        indirect_level=(
            options.indirect_levels[0] if len(options.indirect_levels) == 1 else None
        ),
        indirect_levels=options.indirect_levels,
        endpoint_preflight=endpoint_preflight,
        inference_reuse_sources=reuse_receipts,
        subset=options.evaluation_subset, final_levels={task: options.max_level or {'bbb_martins':5,'bioavailability_ma':6}[task]
                                     for task in options.tasks})
    manifest = root/'matrix.json'
    if manifest.exists() and canonical_json_bytes(json.loads(manifest.read_text())) != canonical_json_bytes(invariant):
        raise ValueError('Matrix resume inputs changed')
    write_json_atomic(manifest, invariant)
    prepared, conditions, selections, run_receipts = [], [], {}, []
    streaming = (
        options.execution_mode == 'throughput'
        and not options.prepare_only
        and options.target_total_load_per_endpoint is None
    )
    incoming = queue.Queue(maxsize=128) if streaming else None
    streamed_results, streaming_errors = [], []
    client = None
    consumer = None
    if streaming:
        client = provider_client(
            options.parallelism,
            settings['max_tokens'],
            options.request_timeout_s,
            options.provider_config,
        )

        def consume():
            try:
                streamed_results.extend(schedule(
                    incoming, client, settings, root / 'inference',
                    parallelism=options.parallelism,
                    reuse_sources=reuse_sources,
                ))
            except BaseException as exc:
                streaming_errors.append(exc)

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()

    def candidates(queries, **kwargs):
        # Record-level mapping assigns each record to one level, so cap prefixes
        # can share one validated maximum-cap retrieval without cross-level moves.
        cap = kwargs.pop('later_limit')
        if molecule_l2_v1:
            return load_candidates(
                queries, later_limit=options.l2_records_per_molecule, **kwargs
            )
        key = hashlib.sha256(canonical_json_bytes(dict(queries=queries,
            arguments=json.loads(json.dumps(kwargs, default=str))))).hexdigest()
        if key not in selections:
            contract = kwargs.get('policy', {}).get('selection_contract')
            retrieval_cap = (
                cap if contract in {
                    'indirect_morgan_semantic_l2_l4.v2',
                    'indirect_morgan_semantic_l2_l4.v3',
                }
                else max(options.records_per_level)
            )
            selections[key] = load_candidates(
                queries, later_limit=retrieval_cap, **kwargs
            )
        molecules, full, original = selections[key]
        later, audit = deepcopy(full), deepcopy(original)
        audit['contract']['later_limit'] = cap
        for qid, levels in later.items():
            for level, entry in levels.items():
                level_cap = cap[level] if isinstance(cap, dict) else cap
                entry['records'] = entry['records'][:level_cap]
                entry.update(selected_records=len(entry['records']),
                    selected_molecules=len({r['reference_molecule_id'] for r in entry['records']}),
                    shortfall=max(0,level_cap-len(entry['records'])))
                audit['query_audits'][qid][level].update({k:entry[k] for k in
                    ('selected_records','selected_molecules','shortfall')})
        return molecules, later, audit

    for (cache_path, retrieval_width), prior, replicate_index in (
        (cache_variant, prior, replicate_index)
        for cache_variant in cache_variants
        for prior in options.query_prior_modes
        for replicate_index in range(1, options.replicates + 1)
    ):
        prior_label = 'query_prior' if prior == 'cached' else 'no_query_prior'
        prior_suffix = (
            f'_{prior_label}'
            if len(options.query_prior_modes) > 1
            or options.variant
            or options.prompt_version != 'reranked_progressive_v8'
            else ''
        )
        for mode, molecule_description_mode, indirect_level in (
            (mode, description, level)
            for level in options.indirect_levels or [None]
            for mode in options.reranking_modes
            for description in options.molecule_description_modes
        ):
            condition_tasks = [
                task for task in options.tasks
                if (task, prior, molecule_description_mode)
                not in options.exclude_task_conditions
            ]
            if not condition_tasks:
                continue
            variant = variant_lookup.get((cache_path, mode))
            if options.variant and variant is None:
                continue
            prompt_version = (
                variant['prompt_versions'][prior]
                if variant else options.prompt_versions[prior]
            )
            for pool in options.record_pools:
                for l1_molecules in options.l1_molecules:
                    k_suffix = (
                        f'_k{l1_molecules}'
                        if options.harness_version in {
                            'reranked-progressive-l1-context-v1',
                            'reranked-progressive-l1-context-l2-v1',
                            'reranked-progressive-l1-context-l2-weighted-v1',
                            'reranked-progressive-l1-context-l2-morgan-bucket-v1',
                        }
                        or options.l1_molecules != [10]
                        else ''
                    )
                    contrast_values = (
                        options.l1_min_contrasts
                        if mode in {'morgan-contrastive', 'assay-transfer-contrastive'}
                        or not options.study else [0]
                    )
                    for min_contrast in contrast_values:
                        contrast_suffix = (
                            f'_m{min_contrast}'
                            if mode in {
                                'morgan-contrastive', 'assay-transfer-contrastive'
                            }
                            else ''
                        )
                        for cap in options.records_per_level:
                            description_suffix = (
                                f'_description_{molecule_description_mode}'
                                if options.molecule_description_modes != ['none']
                                else ''
                            )
                            record_axis = (
                                f'molecules{options.l2_molecules}_records_per_molecule'
                                f'{options.l2_records_per_molecule}'
                                if molecule_l2_v1 else f'records{cap}'
                            )
                            named_width = (
                                retrieval_width
                                if len(cache_variants) > 1
                                and (not options.variant
                                     or variant_mode_counts[mode] > 1)
                                else None
                            )
                            width_suffix = f'_w{named_width}' if named_width else ''
                            indirect_suffix = (
                                f'_l{indirect_level}'
                                if indirect_level else ''
                            )
                            name = (f'{mode}_{pool}_{record_axis}{k_suffix}{contrast_suffix}'
                                    f'{prior_suffix}{description_suffix}{width_suffix}'
                                    f'{indirect_suffix}')
                            if options.replicates > 1:
                                name = f'{name}_replicate{replicate_index}'
                            path = _condition_root(
                                options, root, name, mode, prior, l1_molecules,
                                min_contrast, molecule_description_mode,
                                named_width,
                                indirect_level,
                            )
                            args = runner.parse_args([
                                '--tasks', *condition_tasks,
                                '--harness-version', options.harness_version,
                                '--reranking', mode, '--record_pool', pool,
                                '--records-per-level', str(cap),
                                '--assay-transfer-cache', str(cache_path),
                                '--l1-molecules', str(l1_molecules),
                                '--l1-records-per-molecule', '10',
                                '--l2-molecules', str(options.l2_molecules),
                                '--l2-records-per-molecule', str(options.l2_records_per_molecule),
                                '--l1-min-contrast', str(min_contrast),
                                '--query-prior', prior, '--prior-root', str(options.prior_root),
                                '--gold-label-version', options.gold_label_version,
                                '--evaluation-subset', options.evaluation_subset,
                                *(['--allow-test-inference'] if options.allow_test_inference else []),
                                '--prompt-version', prompt_version,
                                '--molecule-description-mode', molecule_description_mode,
                                '--molecule-description-cache-version',
                                options.molecule_description_cache_version,
                                '--max-level', str(indirect_level or options.max_level),
                                *(
                                    ['--reasoning-phase', options.reasoning_phase]
                                    if full_flat else []
                                ),
                                *(
                                    ['--l1-prior-run', str(options.l1_prior_run)]
                                    if options.l1_prior_run else []
                                ),
                                *(
                                    ['--require-complete-l1-prior']
                                    if options.require_complete_l1_prior else []
                                ),
                                *(
                                    ['--indirect-level', str(indirect_level)]
                                    if indirect_level else []
                                ),
                                '--allow-frozen-l1-vote-scores',
                                '--max-tokens', str(options.max_tokens),
                                '--prepare-only', '--output-root', str(path),
                                '--limit', str(options.limit),
                                '--base-url', options.provider_config.providers[0].base_url,
                                '--model', options.provider_config.providers[0].model,
                                '--provider-pool-config', str(options.provider_pool_config),
                                '--execution-mode', options.execution_mode,
                                *(
                                    ['--skip-tool-prefetch']
                                    if prompt_assets(prompt_version)['settings'].get('tools') is False
                                    else []
                                ),
                            ])
                            args.morgan_primary_parent_width = retrieval_width
                            args.request_namespace = (
                                f'replicate:{replicate_index}'
                                if options.replicates > 1 else ''
                            )
                            args.trace_root = str(options.trace_root)
                            args.live_method = (_method_name(mode, molecule_description_mode) if options.study else
                                f'progressive_{mode}_{pool}_records{cap}{k_suffix}{contrast_suffix}'
                                f'{visibility_suffix(options.prompt_version)}_{prior_label}'
                                f'{description_suffix}'
                            )
                            run_document = {
                                'schema_version': 'progressive_study_run.v2',
                                'study': options.study,
                                'method': _method_name(mode, molecule_description_mode),
                                'run_id': path.name,
                                'batch_id': options.batch_id,
                                'condition_key': name,
                                'replicate_index': replicate_index,
                                'status': 'preparing',
                                'created_at': runner._now(),
                                'harness_version': options.harness_version,
                                'prompt_version': prompt_version,
                                'reranking': mode,
                                'record_pool': pool,
                                'records_per_level': cap,
                                'l1_molecules': l1_molecules,
                                'l1_min_contrast': min_contrast,
                                'indirect_level': indirect_level,
                                'reasoning_phase': options.reasoning_phase,
                                'l1_prior_run': (
                                    str(options.l1_prior_run)
                                    if options.l1_prior_run else None
                                ),
                                'query_prior': prior,
                                'molecule_description_mode': molecule_description_mode,
                                'molecule_description': _molecule_description_identity(
                                    molecule_description_mode, prompt_version,
                                    options.molecule_description_cache_version,
                                ),
                                'morgan_primary_parent_width': retrieval_width,
                                'query_prior_directory': PRIOR_DIRECTORIES[prior],
                                'results_root': str(options.results_root),
                                'staging_root': (
                                    str(options.staging_root)
                                    if options.staging_root else None
                                ),
                                'support_mode': 'runnable',
                                'metric_status': 'pending',
                                'tasks': condition_tasks,
                                'evaluation_subset': options.evaluation_subset,
                                'gold_label_version': options.gold_label_version,
                                'retrieval_cache_bundle': str(cache_path),
                                'batch_root': str(
                                    options.results_root / '_batches' / options.batch_id
                                ),
                                'staging_batch_root': (
                                    str(root) if options.staging_root else None
                                ),
                            }
                            if options.study:
                                write_json_atomic(path / 'run.json', run_document)
                                run_receipts.append({
                                    'study': options.study,
                                    'method': _method_name(mode, molecule_description_mode),
                                    'query_prior': prior,
                                    'molecule_description_mode': molecule_description_mode,
                                    'morgan_primary_parent_width': retrieval_width,
                                    'indirect_level': indirect_level,
                                    'run_id': path.name,
                                    'condition_key': name,
                                    'replicate_index': replicate_index,
                                    'tasks': condition_tasks,
                                    'path': str(
                                        options.results_root
                                        / path.relative_to(options.staging_root)
                                        if options.staging_root else path
                                    ),
                                    'staging_path': (
                                        str(path) if options.staging_root else None
                                    ),
                                })
                            args.live_run_dirs, args.live_run_ids = {}, {}
                            from predict.live import PILOT_SIZE, create_run, promote_run, update_run
                            for task in condition_tasks:
                                run_dir = create_run(
                                    root=args.trace_root,
                                    dataset=task,
                                    method=args.live_method,
                                    study=options.study or '',
                                    run_group=path.name if options.study else '',
                                    condition=name if options.study else '',
                                    query_prior=prior if options.study else '',
                                    command=options.live_command,
                                    requested_id=options.live_run_id,
                                    execution_mode=options.execution_mode,
                                    metadata={
                                        'harness': 'progressive',
                                        'harness_version': args.harness_version,
                                        'prompt_version': args.assay_transfer_prompt_version,
                                        'retrieval_cache_bundle': str(cache_path),
                                        'morgan_primary_parent_width': retrieval_width,
                                        'evaluation_subset': options.evaluation_subset,
                                        'l1_molecules': l1_molecules,
                                        'l1_min_contrast': min_contrast,
                                        'indirect_level': indirect_level,
                                        'pilot_size': PILOT_SIZE,
                                        'matrix_output_root': str(
                                            options.results_root / '_batches' / options.batch_id
                                        ),
                                        'raw_run_root': str(
                                            options.results_root
                                            / path.relative_to(options.staging_root)
                                            if options.staging_root else path
                                        ),
                                        'staging_raw_run_root': (
                                            str(path) if options.staging_root else None
                                        ),
                                        'study': options.study,
                                        'method_family': _method_name(mode, molecule_description_mode),
                                        'query_prior': prior,
                                        'molecule_description_mode': molecule_description_mode,
                                        'molecule_description': _molecule_description_identity(
                                            molecule_description_mode, prompt_version,
                                            options.molecule_description_cache_version,
                                        ),
                                        'replicate_index': replicate_index,
                                    },
                                )
                                args.live_run_dirs[task] = run_dir
                                args.live_run_ids[task] = run_dir.name
                                if options.publish_review_traces:
                                    promote_run(run_dir)
                                update_run(run_dir, resume_command=[
                                    *options.live_command, '--live-run-id', run_dir.name,
                                ])
                            write_json_atomic(path/'matrix_execution.json', dict(
                                matrix=str(
                                    options.results_root / '_batches'
                                    / options.batch_id / 'matrix.json'
                                    if options.staging_root else manifest
                                ),
                                staging_matrix=(
                                    str(manifest) if options.staging_root else None
                                ), settings=settings,
                                parallelism=options.parallelism,
                                max_inflight_per_endpoint=options.max_inflight_per_endpoint,
                                endpoint_allocations={
                                    spec.name: slots
                                    for spec, slots in endpoint_allocations(
                                        options.parallelism, options.provider_config
                                    )
                                } if not options.prepare_only else {},
                                response_read_timeout=options.request_timeout_s,
                                max_attempts=10,
                                note='runner manifest describes preparation; this receipt owns inference execution'))

                            def collect(queries, records, indices):
                                queries = list(queries)
                                prepared.extend((args, q) for q in queries)
                                conditions.append((args, records, indices))

                            def stream_query(query):
                                if incoming is not None:
                                    incoming.put(runner.query_steps(args, query, client))

                            print(f'Preparing {path.name}', flush=True)
                            run_kwargs = dict(
                                prepared_callback=collect,
                                candidate_loader=candidates,
                            )
                            if incoming is not None:
                                run_kwargs['prepared_query_callback'] = stream_query
                            runner.run(args, **run_kwargs)
                if replicate_index == options.replicates:
                    selections.clear()
    if options.study:
        write_json_atomic(root / 'runs.json', {
            'schema_version': 'progressive_study_batch_runs.v1',
            'batch_id': options.batch_id,
            'runs': run_receipts,
        })
        _refresh_results_catalog(options.results_root)
    if options.prepare_only:
        for args, _, _ in conditions:
            _update_raw_run(args, 'prepared')
        return 0
    if options.target_total_load_per_endpoint is not None:
        candidate_failovers = options.provider_config.max_failovers
        selection = select_healthy_providers(
            options.provider_config, primary_capacity(options.provider_config),
        )
        options.provider_config = replace(
            selection.config, max_failovers=candidate_failovers,
        )
        model_checks = list(selection.checks)
        options.load_receipt = sample_provider_loads(
            options.provider_config,
            samples=options.load_samples,
            interval_s=options.load_sample_interval_s,
        )
        options.provider_config, options.top_up_allocations = top_up_provider_config(
            options.provider_config,
            options.load_receipt,
            target_total=options.target_total_load_per_endpoint,
        )
        options.parallelism = primary_capacity(options.provider_config)
        options.requested_parallelism = options.parallelism
        options.endpoint_selection = {
            'checks': model_checks,
            'mode': 'top_up_to_total_running_plus_waiting',
            'load_samples': options.load_receipt,
            'allocations': options.top_up_allocations,
        }
        endpoint_preflight['models'] = options.endpoint_selection
        endpoint_preflight['tokenized_reasoning'] = (
            preflight_sglang_tokenized_completion(options.provider_config)
            if reasoning_transport else None
        )
        invariant.update(
            provider_pool=options.provider_config.public_dict(),
            parallelism=options.parallelism,
            requested_parallelism=options.requested_parallelism,
            endpoint_preflight=endpoint_preflight,
            load_samples=options.load_receipt,
            top_up_allocations=options.top_up_allocations,
        )
        write_json_atomic(manifest, invariant)
        for args, _, _ in conditions:
            receipt_path = Path(args.output_root) / 'matrix_execution.json'
            receipt = json.loads(receipt_path.read_text())
            receipt.update(
                parallelism=options.parallelism,
                target_total_load_per_endpoint=(
                    options.target_total_load_per_endpoint
                ),
                endpoint_allocations={
                    provider.name: slots
                    for provider, slots in endpoint_allocations(
                        options.parallelism, options.provider_config
                    )
                },
                load_samples=options.load_receipt,
                top_up_allocations=options.top_up_allocations,
                endpoint_preflight=endpoint_preflight,
            )
            write_json_atomic(receipt_path, receipt)
    if streaming:
        incoming.put(None)
        consumer.join()
        if streaming_errors:
            raise streaming_errors[0]
        result = streamed_results
    else:
        client = provider_client(
            options.parallelism,
            settings['max_tokens'],
            options.request_timeout_s,
            options.provider_config,
        )
    pilot, pilot_counts = [], {}
    for args, query in prepared:
        key = (id(args), query.task)
        if pilot_counts.get(key, 0) >= options.pilot_queries:
            continue
        pilot.append((args, query))
        pilot_counts[key] = pilot_counts.get(key, 0) + 1
    if not streaming and options.execution_mode == 'live':
        result = schedule((runner.query_steps(args,q,client) for args,q in pilot), client, settings,
                          root/'inference', parallelism=min(24, options.parallelism),
                          reuse_sources=reuse_sources)
        if any(r.get('status') != 'ok' for r in result):
            raise RuntimeError('Pilot failed; full matrix was not launched')
        write_json_atomic(root/'pilot.json', dict(status='passed', logical_chains=len(pilot),
            finished_at=runner._now()))
    if (options.execution_mode == 'live' and not options.continue_after_pilot
            and len(pilot) < len(prepared)):
        from predict.live import update_run

        for run_dir in {run_dir for args, _ in prepared for run_dir in args.live_run_dirs.values()}:
            update_run(run_dir, status='awaiting_review',
                       pilot_completed_at=runner._now())
        write_json_atomic(root/'status.json', dict(status='awaiting_review',
            logical_chains=len(pilot), total_chains=len(prepared),
            updated_at=runner._now()))
        for args, _, _ in conditions:
            _update_raw_run(args, 'awaiting_review')
        return 0
    if not streaming:
        result = schedule((runner.query_steps(args,q,client) for args,q in prepared), client, settings,
                          root/'inference', parallelism=options.parallelism,
                          reuse_sources=reuse_sources)
    if len(result) != len(prepared):
        raise RuntimeError('Matrix completion count does not match prepared chains')
    failed = sum(r.get('status') != 'ok' for r in result)
    summarize(conditions, root)
    diagnostic_errors = []
    for condition in conditions:
        args = condition[0]
        condition_root = Path(args.output_root)
        condition_error = None
        if not failed:
            try:
                summarize([condition], condition_root)
                from predict.harnesses.progressive.diagnostics import build_run_diagnostics

                build_run_diagnostics(condition_root)
            except Exception as exc:
                condition_error = str(exc)
                diagnostic_errors.append({'run': str(condition_root), 'error': str(exc)})
        _update_raw_run(
            args,
            'complete' if not failed and condition_error is None else 'incomplete',
            finished_at=runner._now(),
            metric_status='complete' if not failed and condition_error is None else 'incomplete',
            diagnostics='diagnostics_manifest.json' if not failed and condition_error is None else None,
            diagnostic_error=condition_error,
        )
    complete = not failed and not diagnostic_errors
    write_json_atomic(root/'status.json', dict(status='complete' if complete else 'incomplete',
        logical_chains=len(prepared), failed_chains=failed,
        diagnostic_errors=diagnostic_errors, finished_at=runner._now()))
    from predict.live import update_run

    for run_dir in {run_dir for args, _ in prepared for run_dir in args.live_run_dirs.values()}:
        update_run(run_dir, status='failed' if not complete else 'complete',
                   finished_at=runner._now())
    return int(not complete)


def summarize(conditions, root):
    rows = []
    coverage = []
    for args, records, indices in conditions:
        for task in args.tasks:
            runner._summarize_task(task=task, records=records[task], indices=indices[task],
                output_root=Path(args.output_root), max_level=args.max_level,
                query_prior_mode=args.query_prior,
                profile='context_records', context_record_l3_l5=True,
                prompt_version=args.assay_transfer_prompt_version,
                levels_override=runner._run_levels(args, task))
            row = dict(task=task, mode=args.reranking, pool=args.record_pool,
                       records_per_level=args.indirect_record_limit_per_level,
                       l1_molecules=args.context_limit,
                       l1_min_contrast=args.l1_min_contrast,
                       morgan_primary_parent_width=getattr(
                           args, 'morgan_primary_parent_width', None
                       ),
                       molecule_description_mode=args.molecule_description_mode,
                       query_prior=args.query_prior,
                       prompt_version=args.assay_transfer_prompt_version)
            for level in range(1,7):
                path = Path(args.output_root)/task/'levels'/f'level_{level}'/'metrics.json'
                row[f'L{level}'] = json.loads(path.read_text()) if path.exists() else None
                if row[f'L{level}'] is None:
                    continue
                counts = dict(molecules=0, records=0, new_records=0, reused=0, carried=0, calls=0)
                for index in indices[task]:
                    query_dir = runner._query_dir(Path(args.output_root), task, index)
                    stage = query_dir/'levels'/f'level_{level}'
                    if (stage/'prepared.json').exists():
                        payload = json.loads((stage/'prepared.json').read_text())
                        evidence = payload['active_evidence']
                        counts['molecules'] += len(evidence)
                        counts['records'] += sum(len(m.get('cards', {})) for m in evidence.values())
                        counts['new_records'] += len(payload['new_card_ids'])
                    if (stage/'output.json').exists():
                        output = json.loads((stage/'output.json').read_text())
                        counts['calls'] += int(output.get('model_called') is True)
                        counts['reused'] += int(bool(output.get('inference_reuse')) and not output.get('model_called'))
                        counts['carried'] += int(output.get('status') in {'carried_forward','reused_none'})
                n = len(indices[task])
                coverage.append(dict(task=task,mode=args.reranking,pool=args.record_pool,
                    records_per_level=args.indirect_record_limit_per_level,
                    l1_molecules=args.context_limit,
                    l1_min_contrast=args.l1_min_contrast,
                    morgan_primary_parent_width=getattr(
                        args, 'morgan_primary_parent_width', None
                    ),
                    molecule_description_mode=args.molecule_description_mode,
                    level=level,n=n,
                    failed=row[f'L{level}']['n_failed_runs'],**{k:(v/n if k in
                        {'molecules','records','new_records'} else v) for k,v in counts.items()}))
            rows.append(row)
    write_json_atomic(root/'metrics.json', rows)
    for metric in ('macro_f1','accuracy'):
        with (root/f'{metric}.tsv').open('w') as handle:
            fields = ['task','mode','pool','records_per_level','l1_molecules',
                      'l1_min_contrast','morgan_primary_parent_width',
                      'molecule_description_mode','query_prior','prompt_version']+[f'L{i}' for i in range(1,7)]
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (row[k].get(metric) if row[k] else 'NA')
                    if k.startswith('L') else row[k] for k in fields})
    if coverage:
        with (root/'completion_reuse_evidence.tsv').open('w') as handle:
            writer = csv.DictWriter(handle, list(coverage[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(coverage)


if __name__ == '__main__':
    raise SystemExit(main())
