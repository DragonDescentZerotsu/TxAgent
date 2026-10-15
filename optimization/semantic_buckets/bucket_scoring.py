"""Canonical batched-first-pass and locked-anchor sliding semantic utility scoring."""

from __future__ import annotations
from . import storage
from . import execution
import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
import sqlite3
import statistics
from typing import Any
import hashlib
from pathlib import Path
import threading
import argparse
import gzip
from typing import Any, Mapping, Sequence
from jinja2 import Environment, StrictUndefined
import pandas as pd
from predict.api_client.transport import DEFAULT_ENV_FILE
from . import bucket_cards as card_rendering
from predict.utils.json import write_json_atomic
from .inputs import REPOSITORY_ROOT
from .bucket_cards import _card, _anchor_card

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_TOKENS = 20480
TASK = "skin_reaction"
TARGET = "skin sensitization or allergic contact dermatitis"
PROMPT_ROOT = Path(__file__).with_name("prompts") / "bucket_scoring"
SCORING_VERSION = "semantic_bucket_sliding.v8"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _render(
    candidates: Sequence[str],
    anchors: Sequence[str],
    level: str,
    payloads: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    previous_weights: Mapping[str, float] | None = None,
    target: str | None = None,
    source_labels: Mapping[str, str] | None = None,
    prompt_root: Path | None = None,
    rubric: str = "",
) -> str:
    candidate_cards = "\n\n".join(
        (
            _card(
                f"Candidate {index}" + (
                    f"\n  Previous-pass weight (revisable): {previous_weights[bucket]:.2f}"
                    if previous_weights is not None else ""
                ),
                payloads[bucket], source_labels=source_labels,
            )
            for index, bucket in enumerate(candidates, 1)
        )
    )
    anchor_cards = "\n\n".join(
        (
            _anchor_card(
                index, payloads[bucket], scores[bucket], source_labels=source_labels
            )
            for index, bucket in enumerate(anchors, 1)
        )
    )
    name = "anchored.jinja" if anchors else "initial.jinja"
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        ((prompt_root or PROMPT_ROOT) / name).read_text(encoding="utf-8")
    )
    rendered = template.render(
        target=target or TARGET,
        level=level,
        candidate_cards=candidate_cards,
        anchor_cards=anchor_cards,
        previous_weights=previous_weights,
        task_rules=rubric,
    ).strip()
    return rendered + ("\n\nTask-specific scoring rubric:\n" + rubric if rubric else "")


def _preview(anchors: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        bucket: {
            "weight": "<locked prior score>",
            "rationale": "<locked prior rationale>",
        }
        for bucket in anchors
    }


def _queue(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    prompt: str,
    endpoint: Mapping[str, str],
    *,
    model: str = MODEL,
    reasoning_effort: str = "high",
    max_tokens: int = MAX_TOKENS,
) -> str:
    candidates = json.loads(row["candidate_bucket_ids_json"])
    anchors = json.loads(row["anchor_bucket_ids_json"])
    task_prefix = f"{row['task']}/" if "task" in row else ""
    phase = (
        f"{row['stage']}/{task_prefix}{row['level']}/w{int(row['wave']):04d}/c{int(row['chain']):02d}"
        if "wave" in row
        else f"{row['stage']}/{task_prefix}{row['level']}/{int(row['batch']):04d}"
    )
    request_id = storage._request_id("weight_assignment", phase, prompt)
    validation = {
        "candidate_aliases": [f"Candidate {i}" for i in range(1, len(candidates) + 1)],
        "candidate_bucket_ids": candidates,
        "anchor_bucket_ids": anchors,
        "requested_model": model,
        "selected_endpoint": endpoint["name"],
    }
    if "task" in row:
        validation.update(task=row["task"], level=row["level"])
    if "execution_backend" in row:
        validation["execution_backend"] = row["execution_backend"]
    elif reasoning_effort == "medium" and model == "openai/gpt-6-luna":
        validation["execution_backend"] = "luna_standard"
    return storage._queue_request(
        connection,
        request_id=request_id,
        kind="weight_assignment",
        phase=phase,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        validation=validation,
    )


async def _execute_one(
    connection: sqlite3.Connection,
    request_id: str,
    endpoint: Mapping[str, str],
    client: Any,
    profile_hash: str,
    semaphore: asyncio.Semaphore,
    *,
    fanout: int,
    required: int,
    model: str = MODEL,
) -> None:
    for attempt in range(3):
        row = storage._pending_rows(connection, [request_id])
        if not row:
            return
        async with semaphore:
            result = await execution.execute_request(
                client,
                row[0],
                model=model,
                initial_fanout=fanout,
                hedge_seconds=3600,
                maximum=fanout,
                required=required,
            )
        storage.persist_result(
            connection,
            result,
            base_url=endpoint["base_url"],
            model=model,
            benchmark_sha256=profile_hash,
        )
        if result["status"] == "complete":
            return
        invalid_output = any(
            (
                str(receipt.get("error", "")).startswith(
                    ("ValueError:", "JSONDecodeError:")
                )
                for receipt in result["receipts"]
            )
        )
        if not invalid_output or attempt == 2:
            raise RuntimeError(f"request failed: {request_id}: {result['error']}")


def _scores(connection: sqlite3.Connection, stage: str) -> pd.DataFrame:
    aggregates = storage.aggregate_lookup(connection)
    rows = []
    query = connection.execute(
        "SELECT * FROM requests WHERE status='complete' AND phase LIKE ? ORDER BY phase",
        (f"{stage}/%",),
    )
    for request in query:
        validation, response = (
            json.loads(request["validation_json"]),
            json.loads(request["response_json"]),
        )
        buckets = dict(
            zip(
                validation["candidate_aliases"],
                validation["candidate_bucket_ids"],
                strict=True,
            )
        )
        aggregate = aggregates[request["request_id"]]
        for score in response["scores"]:
            component = aggregate["components"][score["candidate"]]
            rows.append(
                {
                    "task": validation.get("task", TASK),
                    "level": validation.get("level", request["phase"].split("/")[1]),
                    "semantic_bucket_id": buckets[score["candidate"]],
                    "weight": float(score["weight"]),
                    "rationale": score["rationale"],
                    "request_id": request["request_id"],
                    "anchor_bucket_ids_json": storage._canonical_json(
                        validation["anchor_bucket_ids"]
                    ),
                    "endpoint": validation["selected_endpoint"],
                    "execution_backend": validation.get(
                        "execution_backend", "shared_provider_pool"
                        if validation["selected_endpoint"] == "shared_provider_pool" else "local_dgx"
                    ),
                    "weight_stddev": component["sample_stddev"],
                    "weight_min": component["min"],
                    "weight_max": component["max"],
                    "component_weights_json": storage._canonical_json(
                        component["component_weights"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _pool_adapter(config: Any, reasoning_effort: str = "high") -> Any:
    from predict.api_client.pool import async_completion_pool

    return async_completion_pool(
        config,
        reasoning_effort=reasoning_effort,
        max_tokens=MAX_TOKENS,
        env_file=DEFAULT_ENV_FILE,
    )


def _pass1_schedule(
    world: pd.DataFrame | None = None, batch_size: int = 12
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if world is None:
        raise ValueError("an explicit frozen bucket world is required")
    if world.duplicated(["level", "semantic_bucket_id"]).any():
        raise ValueError("ranking world contains repeated bucket identities")
    rows = []
    for level, group in world.groupby("level", sort=True):
        buckets = sorted(group.semantic_bucket_id.astype(str))
        for batch, start in enumerate(range(0, len(buckets), batch_size)):
            rows.append(
                dict(
                    stage="pass1",
                    level=str(level),
                    batch=batch,
                    candidate_bucket_ids_json=storage._canonical_json(
                        buckets[start : start + batch_size]
                    ),
                    anchor_bucket_ids_json="[]",
                )
            )
    return pd.DataFrame(rows)


def _pass2_schedule(
    pass1: pd.DataFrame,
    *,
    chain_mode: str = "auto",
    chains: int = 1,
    chain_counts: Mapping[str, int] | None = None,
    max_chains: int = 8,
    batch_size: int = 12,
) -> pd.DataFrame:
    """Balance all requested task-levels against the same longest chain."""
    if chain_mode not in {"auto", "manual"} or min(chains, max_chains, batch_size) < 1:
        raise ValueError("invalid chain mode, count or batch size")
    if chain_mode == "auto" and chain_counts:
        raise ValueError(
            "manual chain overrides cannot be combined with automatic mode"
        )
    frame = pass1.copy()
    if "task" not in frame:
        frame["task"] = TASK
    groups = list(frame.groupby(["task", "level"], sort=True))
    if not groups or frame.duplicated(["task", "level", "semantic_bucket_id"]).any():
        raise ValueError("Pass 1 must cover each task-level bucket exactly once")
    keys = {f"{task}:{level}" for (task, level), _ in groups}
    if set(chain_counts or {}) - keys or any(
        (int(value) < 1 for value in (chain_counts or {}).values())
    ):
        raise ValueError(
            "chain overrides must name requested task-levels with positive counts"
        )
    allowed = [
        2**power for power in range(max_chains.bit_length()) if 2**power <= max_chains
    ]
    reference = max((len(group) for _, group in groups)) / max_chains
    rows = []
    for (task, level), group in groups:
        buckets = group.sort_values(
            ["weight", "semantic_bucket_id"], ascending=[False, True]
        ).semantic_bucket_id.tolist()
        width = (
            min(
                allowed,
                key=lambda count: (abs(len(buckets) / count - reference), count),
            )
            if chain_mode == "auto"
            else int((chain_counts or {}).get(f"{task}:{level}", chains))
        )
        batches = [
            buckets[start : start + batch_size]
            for start in range(0, len(buckets), batch_size)
        ]
        if batch_size > 2 and len(batches) > 1 and (len(batches[-1]) == 1):
            batches[-1].insert(0, batches[-2].pop())
        width = min(width, len(batches))
        for batch, candidates in enumerate(batches):
            rows.append(
                dict(
                    stage="pass2",
                    task=str(task),
                    level=str(level),
                    batch=batch,
                    wave=batch // width,
                    chain=batch % width,
                    candidate_bucket_ids_json=storage._canonical_json(candidates),
                    anchor_bucket_ids_json="[]",
                )
            )
    return pd.DataFrame(rows)


def _wave_decision(
    connection: sqlite3.Connection,
    level: str,
    wave: int,
    rows: list[dict[str, Any]],
    *,
    advance_after: int = 2,
    overlap: int = 2,
    stage: str = "pass2",
) -> list[str] | None:
    decision_key = level if stage == "pass2" else f"{stage}:{level}"
    saved = connection.execute(
        "SELECT anchor_bucket_ids_json FROM wave_anchor_decisions WHERE level=? AND wave=?",
        (decision_key, wave),
    ).fetchone()
    if saved:
        return json.loads(saved[0])
    completed = []
    for row in rows:
        phase = f"{stage}/{level}/w{wave:04d}/c{int(row['chain']):02d}"
        receipt = connection.execute(
            "SELECT a.created_at,r.request_id FROM requests r JOIN speculative_aggregates a ON a.request_id=r.request_id WHERE r.phase=? AND r.status='complete'",
            (phase,),
        ).fetchone()
        if receipt:
            completed.append((receipt[0], int(row["chain"]), receipt[1], row))
    quorum = min(advance_after, len(rows))
    if len(completed) < quorum:
        return None
    chosen = sorted(completed, key=lambda item: (item[0], item[1]))[:quorum]
    anchors = [
        bucket
        for _, _, _, row in chosen
        for bucket in (
            json.loads(row["candidate_bucket_ids_json"])[-overlap:] if overlap else []
        )
    ]
    sources = [
        dict(chain=chain, request_id=request_id) for _, chain, request_id, _ in chosen
    ]
    connection.execute(
        "INSERT INTO wave_anchor_decisions VALUES (?,?,?,?,?)",
        (
            decision_key,
            wave,
            storage._canonical_json(anchors),
            storage._canonical_json(sources),
            _now(),
        ),
    )
    connection.commit()
    return anchors


def prepare(
    inputs: Mapping[str, Mapping[str, Any]], root: Path, settings: Mapping[str, Any]
) -> dict:
    """Freeze scientific cards, source bindings, settings and provider configs offline."""
    if root.exists():
        raise FileExistsError(root)
    settings = dict(settings)
    settings.setdefault("passes", 2)
    settings.setdefault("reasoning_effort", "high")
    settings.setdefault("max_tokens", MAX_TOKENS)
    settings.setdefault("prompt_dir", str(PROMPT_ROOT.resolve()))
    if settings["max_tokens"] < 1 or settings["reasoning_effort"] not in {
        "high",
        "medium",
    }:
        raise ValueError("invalid scoring token budget or reasoning effort")
    if not isinstance(settings["passes"], int) or settings["passes"] < 1:
        raise ValueError("passes must be a positive integer")
    from predict.api_client.pool import load_provider_pool_config

    profiles = {}
    for stage in ("pass1", "pass2"):
        path = Path(settings[f"{stage}_provider_config"]).resolve()
        config = load_provider_pool_config(path)
        fixed = json.loads(path.read_text()).get("fixed_replica_counts", {})
        if fixed and (
            set(fixed) - {provider.name for provider in config.providers}
            or any(
                (not isinstance(count, int) or count < 1 for count in fixed.values())
            )
            or sum(fixed.values()) != settings[f"{stage}_fanout"]
        ):
            raise ValueError(
                "fixed replica counts must name declared providers and match fanout"
            )
        profiles[stage] = dict(
            path=str(path),
            sha256=_sha256(path),
            config=config.public_dict(),
            fixed_replica_counts=fixed,
        )
        if not 1 <= settings[f"{stage}_average"] <= settings[f"{stage}_fanout"]:
            raise ValueError(
                "each averaging count must be positive and no greater than fanout"
            )
    if (
        settings["chain_mode"] not in {"auto", "manual"}
        or settings["overlap"] < 0
        or settings["advance_after"] < 1
        or (settings["pass1_batch_size"] < 1)
        or (settings["pass2_batch_size"] < 1)
        or (settings["chains"] < 1)
        or (settings["max_chains"] < 1)
    ):
        raise ValueError("invalid window or chain settings")
    schedules, payloads, bindings, targets, labels = ([], {}, {}, {}, {})
    rubrics = {}
    for task, entry in inputs.items():
        world_path, cards_path = (
            Path(entry["world"]).resolve(),
            Path(entry["payloads"]).resolve(),
        )
        world = pd.read_parquet(world_path, columns=["level", "semantic_bucket_id"])
        world = world.astype({"level": str, "semantic_bucket_id": str})
        if (
            world.duplicated(["level", "semantic_bucket_id"]).any()
            or not world.level.str.fullmatch("L[1-9][0-9]*").all()
            or world.semantic_bucket_id.str.strip().eq("").any()
        ):
            raise ValueError(
                f"invalid or duplicated task-level bucket identities: {task}"
            )
        if entry.get("levels"):
            if set(entry["levels"]) - set(world.level):
                raise ValueError(f"requested levels are absent for {task}")
            world = world[world.level.isin(entry["levels"])]
        opener = gzip.open if cards_path.suffix == ".gz" else open
        with opener(cards_path, "rt", encoding="utf-8") as handle:
            cards = json.load(handle)
        wanted = set(world.semantic_bucket_id.astype(str))
        if world.empty or not wanted <= cards.keys() or (not entry.get("target")):
            raise ValueError(f"missing scientific cards, target or task-levels: {task}")
        payloads[task] = {bucket: cards[bucket] for bucket in sorted(wanted)}
        for card in payloads[task].values():
            _card("Candidate", card, source_labels=entry.get("source_labels", {}))
        targets[task], labels[task] = (
            entry["target"],
            dict(entry.get("source_labels", {})),
        )
        rubrics[task] = ""
        if entry.get("rubric"):
            rubric_path = Path(entry["rubric"]).resolve()
            rubrics[task] = (
                Environment(undefined=StrictUndefined)
                .from_string(rubric_path.read_text())
                .render(task=task, target=entry["target"])
                .strip()
            )
        schedule = _pass1_schedule(world, settings["pass1_batch_size"])
        schedule["task"] = task
        schedules.append(schedule)
        bindings[task] = {
            name: dict(path=str(path), sha256=_sha256(path))
            for name, path in (("world", world_path), ("payloads", cards_path))
        }
        if entry.get("input_manifest"):
            path = Path(entry["input_manifest"]).resolve()
            bindings[task]["input_manifest"] = dict(path=str(path), sha256=_sha256(path))
        if entry.get("rubric"):
            bindings[task]["rubric"] = dict(
                path=str(rubric_path), sha256=_sha256(rubric_path)
            )
    if not schedules:
        raise ValueError("at least one task-level is required")
    requested_keys = {
        f"{task}:{row.level}"
        for task, schedule in zip(inputs, schedules)
        for row in schedule.itertuples()
    }
    if set(settings["chain_counts"]) - requested_keys or any(
        (count < 1 for count in settings["chain_counts"].values())
    ):
        raise ValueError(
            "manual chain overrides name unavailable task-levels or invalid counts"
        )
    if settings["chain_mode"] == "auto" and (
        settings["chain_counts"] or settings["chains"] != 1
    ):
        raise ValueError("automatic chaining cannot accept manual chain counts")
    schedule = pd.concat(schedules, ignore_index=True)
    seed = None
    seed_schedule = None
    if settings.get("seed_weights"):
        if settings["passes"] < 2:
            raise ValueError("seeded scoring requires at least one new pass (--passes >= 2)")
        seed_path = Path(settings["seed_weights"]).resolve()
        seed = pd.read_parquet(seed_path).copy()
        if "task" not in seed and len(inputs) == 1:
            seed["task"] = next(iter(inputs))
        keys = ["task", "level", "semantic_bucket_id"]
        if not set(keys + ["weight"]) <= set(seed):
            raise ValueError("seed weights require task, level, bucket and weight")
        expected = {(r.task, r.level, b) for r in schedule.itertuples()
                    for b in json.loads(r.candidate_bucket_ids_json)}
        if seed.duplicated(keys).any() or set(seed[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError("seed weights must cover the exact task-level bucket world once")
        seed["weight"] = pd.to_numeric(seed.weight, errors="raise")
        if seed.weight.dtype.kind == "b" or not seed.weight.between(0, 1).all():
            raise ValueError("seed weights must be finite numbers in [0, 1]")
        bindings["baseline"] = {"seed_weights": dict(path=str(seed_path), sha256=_sha256(seed_path))}
        seed_schedule = _pass2_schedule(seed, chain_mode=settings["chain_mode"],
            chains=settings["chains"], chain_counts=settings["chain_counts"],
            max_chains=settings["max_chains"], batch_size=settings["pass2_batch_size"])
    root.mkdir(parents=True)
    _atomic_frame(schedule, root / "pass1_schedule.parquet")
    with gzip.open(
        root / "bucket_prompt_payloads.json.gz", "wt", encoding="utf-8"
    ) as handle:
        json.dump(payloads, handle, sort_keys=True, ensure_ascii=False)
    review = root / "prompt_review"
    review.mkdir()
    review_schedule = schedule if seed_schedule is None else seed_schedule
    for (task, level), group in review_schedule.groupby(["task", "level"]):
        candidates = json.loads(group.iloc[0].candidate_bucket_ids_json)
        previous_weights = None if seed is None else seed[
            seed.task.eq(task) & seed.level.eq(level)
        ].set_index("semantic_bucket_id").weight.to_dict()
        prompt = _render(
            candidates,
            [],
            level,
            payloads[task],
            {},
            previous_weights=previous_weights,
            target=targets[task],
            source_labels=labels[task],
            prompt_root=Path(settings["prompt_dir"]),
            rubric=rubrics[task],
        )
        (review / f"{task}_{level}_initial.txt").write_text(prompt + "\n")
        anchors = (
            candidates[-min(settings["overlap"], len(candidates)) :]
            if settings["overlap"]
            else []
        )
        next_group = group.iloc[1] if len(group) > 1 else group.iloc[0]
        preview_candidates = [
            bucket
            for bucket in json.loads(next_group.candidate_bucket_ids_json)
            if bucket not in anchors
        ]
        if preview_candidates:
            anchored = _render(
                preview_candidates,
                anchors,
                level,
                payloads[task],
                _preview(anchors),
                previous_weights=previous_weights,
                target=targets[task],
                source_labels=labels[task],
                prompt_root=Path(settings["prompt_dir"]),
                rubric=rubrics[task],
            )
            (review / f"{task}_{level}_anchored_preview.txt").write_text(
                anchored + "\n"
            )
    manifest = dict(
        version=SCORING_VERSION,
        status="prepared",
        created_at=_now(),
        targets=targets,
        source_labels=labels,
        rubrics=rubrics,
        settings=dict(settings),
        completed_passes=0,
        source_bindings=bindings,
        provider_configs=profiles,
        completion_requests_made=0,
        engine_sha256=_sha256(Path(__file__)),
        execution_code={
            str(path): _sha256(path)
            for path in (
                Path(execution.__file__),
                Path(card_rendering.__file__),
                Path(storage.__file__),
                REPOSITORY_ROOT / "predict/api_client/pool.py",
                REPOSITORY_ROOT / "predict/api_client/transport.py",
            )
        },
        prompt_assets={
            name: dict(
                path=str(Path(settings["prompt_dir"]) / name),
                sha256=_sha256(Path(settings["prompt_dir"]) / name),
            )
            for name in ("initial.jinja", "anchored.jinja")
        },
        files={
            name: _sha256(root / name)
            for name in ("pass1_schedule.parquet", "bucket_prompt_payloads.json.gz")
        },
        expected_bucket_count=sum(
            (
                len(json.loads(row.candidate_bucket_ids_json))
                for row in schedule.itertuples()
            )
        ),
    )
    if seed is not None:
        _atomic_frame(seed, root / "pass1_weights.parquet")
        next_schedule = seed_schedule
        _atomic_frame(next_schedule, root / "pass2_schedule.parquet")
        manifest.update(status="pass1_complete", completed_passes=1, baseline_source="published_weights")
        manifest["files"].update({name: _sha256(root / name)
            for name in ("pass1_weights.parquet", "pass2_schedule.parquet")})
        manifest["chain_counts"] = {f"{task}:{level}": int(group.chain.max()) + 1
            for (task, level), group in next_schedule.groupby(["task", "level"])}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _load_scoring_run(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["version"] != SCORING_VERSION or manifest["engine_sha256"] != _sha256(
        Path(__file__)
    ):
        raise ValueError("scoring implementation changed; use a fresh run identity")
    for name, digest in manifest["execution_code"].items():
        if _sha256(Path(name)) != digest:
            raise ValueError(
                "scoring transport or aggregation changed; use a fresh run identity"
            )
    for name, digest in manifest["files"].items():
        if _sha256(root / name) != digest:
            raise ValueError(f"frozen scoring artifact changed: {name}")
    for pins in manifest["source_bindings"].values():
        for pin in pins.values():
            if _sha256(Path(pin["path"])) != pin["sha256"]:
                raise ValueError("scoring source changed; use a fresh run identity")
    for pin in [
        *manifest["provider_configs"].values(),
        *manifest["prompt_assets"].values(),
    ]:
        if _sha256(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("provider route changed; use a fresh run identity")
    with gzip.open(
        root / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8"
    ) as handle:
        return (manifest, json.load(handle))


async def _execute_fixed_one(
    connection: sqlite3.Connection,
    request_id: str,
    targets: Sequence[tuple[Any, str]],
    profile_hash: str,
    drains: list[asyncio.Task],
    required: int,
) -> None:
    row = connection.execute(
        "SELECT * FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if row["status"] == "complete":
        return
    for attempt in range(3):
        result, drain = await execution.execute_fixed_request(
            targets, row, required=required
        )
        storage.persist_result(
            connection,
            result,
            base_url="shared-provider-pool",
            model=targets[0][1],
            benchmark_sha256=profile_hash,
        )
        if drain is not None:

            async def store_late(result=result, drain=drain):
                storage.persist_late_receipts(connection, result, await drain)

            drains.append(asyncio.create_task(store_late()))
        if result["status"] == "complete":
            return
        if attempt == 2:
            raise RuntimeError(
                f"scoring request failed: {request_id}: {result['error']}"
            )


async def _run_scoring_stage(
    root: Path, stage: str, manifest: dict, payloads: dict
) -> None:
    from predict.api_client.pool import (
        async_completion_pool,
        load_provider_pool_config,
        preflight_provider_models,
    )
    from predict.api_client.transport import DEFAULT_ENV_FILE

    settings = dict(manifest["settings"])
    previous_weights = None
    if stage != "pass1":
        previous = pd.read_parquet(root / f"pass{int(stage.removeprefix('pass')) - 1}_weights.parquet")
        previous_weights = previous.set_index(["task", "level", "semantic_bucket_id"]).weight.to_dict()
    for field in ("fanout", "average", "batch_size"):
        settings[f"{stage}_{field}"] = settings[
            f"{('pass1' if stage == 'pass1' else 'pass2')}_{field}"
        ]
    profile = manifest["provider_configs"]["pass1" if stage == "pass1" else "pass2"]
    config = load_provider_pool_config(Path(profile["path"]))
    checks = await asyncio.to_thread(preflight_provider_models, config)
    model = config.providers[0].model
    clients, targets = ([], [])
    fixed = profile.get("fixed_replica_counts")
    if fixed:
        from dataclasses import replace

        for provider in config.providers:
            if provider.name in fixed:
                client = async_completion_pool(
                    replace(config, providers=(provider,), max_failovers=0),
                    env_file=DEFAULT_ENV_FILE,
                    max_tokens=settings["max_tokens"],
                    reasoning_effort=settings["reasoning_effort"],
                )
                clients.append(client)
                targets.extend([(client, provider.model)] * fixed[provider.name])
    else:
        client = async_completion_pool(
            config,
            env_file=DEFAULT_ENV_FILE,
            max_tokens=settings["max_tokens"],
            reasoning_effort=settings["reasoning_effort"],
        )
        clients.append(client)
        targets = [(client, model)] * settings[f"{stage}_fanout"]
    manifest.setdefault("provider_preflight", {})[stage] = checks
    write_json_atomic(root / "manifest.json", manifest)
    connection = storage._request_database(
        root / "requests.sqlite3", journal_mode="DELETE"
    )
    storage.ensure_tables(connection)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS wave_anchor_decisions (level TEXT NOT NULL,wave INTEGER NOT NULL,anchor_bucket_ids_json TEXT NOT NULL,source_requests_json TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(level,wave))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS scoring_weights (stage TEXT,task TEXT,level TEXT,semantic_bucket_id TEXT,weight REAL,rationale TEXT,request_id TEXT,PRIMARY KEY(stage,task,level,semantic_bucket_id))"
    )
    connection.commit()
    drains, outstanding = ([], [])
    profile_hash = hashlib.sha256(
        storage._canonical_json(dict(settings=settings, provider=profile)).encode()
    ).hexdigest()
    schedule = pd.read_parquet(root / f"{stage}_schedule.parquet")

    def queue(row, anchors):
        task, level = (row["task"], row["level"])
        score_map = {}
        if anchors:
            placeholders = ",".join(("?" for _ in anchors))
            score_map = {
                row["semantic_bucket_id"]: dict(row)
                for row in connection.execute(
                    f"SELECT * FROM scoring_weights WHERE stage=? AND task=? AND level=? AND semantic_bucket_id IN ({placeholders})",
                    (stage, task, level, *anchors),
                )
            }
        if not set(anchors) <= score_map.keys():
            raise ValueError(f"missing locked anchors: {task}/{level}")
        row["anchor_bucket_ids_json"] = storage._canonical_json(anchors)
        candidates = json.loads(row["candidate_bucket_ids_json"])
        prompt = _render(
            candidates,
            anchors,
            level,
            payloads[task],
            score_map,
            previous_weights=None if previous_weights is None else {
                bucket: previous_weights[task, level, bucket] for bucket in candidates
            },
            target=manifest["targets"][task],
            source_labels=manifest["source_labels"][task],
            prompt_root=Path(settings["prompt_dir"]),
            rubric=manifest.get("rubrics", {}).get(task, ""),
        )
        return _queue(
            connection,
            row,
            prompt,
            {"name": "shared_provider_pool"},
            model=model,
            reasoning_effort=settings["reasoning_effort"],
            max_tokens=settings["max_tokens"],
        )

    async def execute(row, anchors):
        request_id = queue(row, anchors)
        await _execute_fixed_one(
            connection,
            request_id,
            targets,
            profile_hash,
            drains,
            settings[f"{stage}_average"],
        )
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()[0]
        )
        candidates = json.loads(row["candidate_bucket_ids_json"])
        by_alias = {score["candidate"]: score for score in response["scores"]}
        connection.executemany(
            "INSERT OR IGNORE INTO scoring_weights VALUES (?,?,?,?,?,?,?)",
            [
                (
                    stage,
                    row["task"],
                    row["level"],
                    bucket,
                    by_alias[f"Candidate {i}"]["weight"],
                    by_alias[f"Candidate {i}"]["rationale"],
                    request_id,
                )
                for i, bucket in enumerate(candidates, 1)
            ],
        )
        connection.commit()

    async def chain(task, level, frame):
        pending = []
        try:
            for wave, group in frame.groupby("wave", sort=True):
                qualified = f"{task}/{level}"
                decision_key = qualified if stage == "pass2" else f"{stage}:{qualified}"
                prior = (
                    []
                    if wave == 0
                    else connection.execute(
                        "SELECT anchor_bucket_ids_json FROM wave_anchor_decisions WHERE level=? AND wave=?",
                        (decision_key, int(wave) - 1),
                    ).fetchone()
                )
                if wave and prior is None:
                    raise ValueError("missing frozen prior-wave anchor decision")
                anchors = [] if wave == 0 else json.loads(prior[0])
                rows = group.sort_values("chain").to_dict("records")
                current = [asyncio.create_task(execute(row, anchors)) for row in rows]
                pending.extend(current)
                while (
                    _wave_decision(
                        connection,
                        qualified,
                        int(wave),
                        rows,
                        advance_after=settings["advance_after"],
                        overlap=settings["overlap"],
                        stage=stage,
                    )
                    is None
                ):
                    running = [job for job in current if not job.done()]
                    if not running:
                        raise RuntimeError(
                            f"wave cannot reach anchor quorum: {qualified}/{wave}"
                        )
                    await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        finally:
            results = await asyncio.gather(*pending, return_exceptions=True)
            if any((isinstance(result, BaseException) for result in results)):
                raise RuntimeError(f"required scoring window failed: {task}/{level}")

    try:
        if stage == "pass1":
            outstanding = [
                asyncio.create_task(execute(row, []))
                for row in schedule.to_dict("records")
            ]
        else:
            outstanding = [
                asyncio.create_task(chain(task, level, frame))
                for (task, level), frame in schedule.groupby(
                    ["task", "level"], sort=True
                )
            ]
        results = await asyncio.gather(*outstanding, return_exceptions=True)
        if any((isinstance(result, BaseException) for result in results)):
            raise RuntimeError(f"{stage} contains failed scoring requests")
        await asyncio.gather(*drains, return_exceptions=False)
        scores = _scores(connection, stage)
        expected = {
            (row.task, row.level, bucket)
            for row in schedule.itertuples()
            for bucket in json.loads(row.candidate_bucket_ids_json)
        }
        actual = (
            set(
                scores[["task", "level", "semantic_bucket_id"]].itertuples(
                    index=False, name=None
                )
            )
            if len(scores)
            else set()
        )
        if actual != expected or len(scores) != len(expected):
            raise ValueError(
                f"{stage} did not score every requested bucket exactly once"
            )
        _atomic_frame(scores, root / f"{stage}_weights.parquet")
        _atomic_frame(scores, root / f"{stage}_weights.tsv")
        manifest["files"][f"{stage}_weights.parquet"] = _sha256(
            root / f"{stage}_weights.parquet"
        )
        completed = int(stage.removeprefix("pass"))
        manifest["completed_passes"] = completed
        if completed < settings["passes"]:
            next_stage = f"pass{completed + 1}"
            next_schedule = _pass2_schedule(
                scores,
                chain_mode=settings["chain_mode"],
                chains=settings["chains"],
                chain_counts=settings["chain_counts"],
                max_chains=settings["max_chains"],
                batch_size=settings["pass2_batch_size"],
            )
            next_schedule["stage"] = next_stage
            schedule_name = f"{next_stage}_schedule.parquet"
            _atomic_frame(next_schedule, root / schedule_name)
            manifest["files"][schedule_name] = _sha256(root / schedule_name)
            manifest["chain_counts"] = {
                f"{task}:{level}": int(group.chain.max()) + 1
                for (task, level), group in next_schedule.groupby(["task", "level"])
            }
            manifest["status"] = f"{stage}_complete"
        else:
            first = pd.read_parquet(root / "pass1_weights.parquet").sort_values(
                ["task", "level", "weight", "semantic_bucket_id"],
                ascending=[True, True, False, True],
            )
            first["pass1_order_rank"] = first.groupby(["task", "level"]).cumcount() + 1
            final = scores.merge(
                first[["task", "level", "semantic_bucket_id", "pass1_order_rank"]],
                on=["task", "level", "semantic_bucket_id"],
                validate="one_to_one",
            )
            final = final.sort_values(
                ["task", "level", "weight", "semantic_bucket_id"],
                ascending=[True, True, False, True],
            )
            final["final_rank"] = final.groupby(["task", "level"]).cumcount() + 1
            _atomic_frame(final, root / "weights.parquet")
            _atomic_frame(final, root / "weights.tsv")
            _atomic_frame(
                final[
                    [
                        "task",
                        "level",
                        "semantic_bucket_id",
                        "final_rank",
                        "weight",
                        "rationale",
                        "pass1_order_rank",
                    ]
                ],
                root / "semantic_bucket_rankings.parquet",
            )
            manifest["files"]["weights.parquet"] = _sha256(root / "weights.parquet")
            manifest["files"]["semantic_bucket_rankings.parquet"] = _sha256(
                root / "semantic_bucket_rankings.parquet"
            )
            manifest.update(
                status="complete_unselected_candidate",
                completed_at=_now(),
                active_policy_changed=False,
            )
        manifest["completion_requests_made"] = connection.execute(
            "SELECT count(*) FROM speculative_replica_receipts"
        ).fetchone()[0]
        write_json_atomic(root / "manifest.json", manifest)
    finally:
        await asyncio.gather(*drains, return_exceptions=True)
        connection.close()
        await asyncio.gather(*(client.close() for client in clients))


def run(root: Path, *, through_pass: int | None = None) -> dict:
    import fcntl

    with (root / "writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, payloads = _load_scoring_run(root)
        total = manifest["settings"]["passes"]
        stop = total if through_pass is None else through_pass
        if not 1 <= stop <= total:
            raise ValueError("requested pass is outside the prepared scoring run")
        for number in range(manifest.get("completed_passes", 0) + 1, stop + 1):
            asyncio.run(_run_scoring_stage(root, f"pass{number}", manifest, payloads))
        return json.loads((root / "manifest.json").read_text())


def collect(root: Path) -> dict:
    manifest, _ = _load_scoring_run(root)
    if manifest["status"] != "complete_unselected_candidate":
        raise ValueError("all prepared scoring passes must complete before collection")
    final = pd.read_parquet(root / "weights.parquet")
    target = root / "semantic_bucket_weights.parquet"
    if not target.exists():
        _atomic_frame(final, target)
        manifest["files"][target.name] = _sha256(target)
        write_json_atomic(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    from predict.api_client.pool import DEFAULT_PROVIDER_POOL_CONFIG

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Task bindings: target, world Parquet, cached bucket payloads, optional levels/source_labels.",
    )
    parser.add_argument("--chain-mode", choices=("auto", "manual"), default="auto")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--chain-count", action="append", default=[], metavar="TASK:LEVEL=N"
    )
    parser.add_argument("--max-chains", type=int, default=8)
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Locked tail anchors per advancing chain.",
    )
    parser.add_argument("--advance-after", type=int, default=2)
    for stage, fanout in (("pass1", 1), ("pass2", 3)):
        parser.add_argument(f"--{stage}-batch-size", type=int, default=12)
        parser.add_argument(f"--{stage}-fanout", type=int, default=fanout)
        parser.add_argument(f"--{stage}-average", type=int, default=fanout)
        parser.add_argument(
            f"--{stage}-provider-config",
            type=Path,
            default=DEFAULT_PROVIDER_POOL_CONFIG,
        )
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed-weights", type=Path,
                        help="Use published weights as the completed initial pass; subsequent passes make model calls.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort", choices=("high", "medium"), default="high"
    )
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_ROOT)
    parser.add_argument("--through-pass", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.input_manifest is None:
            parser.error(
                "prepare requires --input-manifest; scientific inputs are explicit"
            )
        settings = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key
            not in {"command", "root", "input_manifest", "chain_count", "through_pass"}
        }
        overrides = {}
        for value in args.chain_count:
            key, separator, count = value.partition("=")
            if not separator or key in overrides:
                parser.error("chain-count requires unique TASK:LEVEL=N entries")
            try:
                overrides[key] = int(count)
            except ValueError:
                parser.error("chain counts must be integers")
        settings["chain_counts"] = overrides
        if args.chain_mode == "auto" and (overrides or args.chains != 1):
            parser.error("automatic chaining does not accept manual chain counts")
        document = json.loads(args.input_manifest.read_text())
        inputs = document["tasks"]
        for entry in inputs.values():
            entry["input_manifest"] = str(args.input_manifest.resolve())
            for key in ("world", "payloads", "rubric"):
                if not entry.get(key):
                    continue
                entry[key] = str((args.input_manifest.parent / entry[key]).resolve())
        result = prepare(inputs, args.root.resolve(), settings)
    else:
        result = (
            run(args.root.resolve(), through_pass=args.through_pass)
            if args.command == "run"
            else collect(args.root.resolve())
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _validate_weight_response(
    result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"scores"} or not isinstance(result["scores"], list):
        raise ValueError("weight response must contain only a scores list")
    expected = list(validation["candidate_aliases"])
    found = []
    for index, score in enumerate(result["scores"]):
        if not isinstance(score, Mapping) or set(score) != {
            "candidate",
            "weight",
            "rationale",
        }:
            raise ValueError(f"scores[{index}] has unexpected fields")
        candidate, weight, rationale = (
            score["candidate"],
            score["weight"],
            score["rationale"],
        )
        if candidate not in expected or candidate in found:
            raise ValueError(f"scores[{index}] has an unknown or repeated candidate")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"scores[{index}] weight must be numeric")
        if (
            not 0 <= float(weight) <= 1
            or abs(float(weight) * 100 - round(float(weight) * 100)) > 1e-08
        ):
            raise ValueError(f"scores[{index}] weight must use hundredth increments")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"scores[{index}] requires a rationale")
        found.append(candidate)
    if found != expected:
        raise ValueError("weight response must cover candidates in displayed order")
    return result


_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()


def aggregation_method(required: int) -> str:
    return f"first_{required}_valid_mean_v1"


REQUIRED_REPLICAS = 8


def aggregate_responses(
    chosen: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    *,
    required: int = REQUIRED_REPLICAS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(chosen) != required:
        raise ValueError(f"aggregation requires exactly {required} replicas")
    vectors = []
    responses = []
    for receipt in chosen:
        response = receipt["response"]
        by_alias = {score["candidate"]: score for score in response["scores"]}
        vectors.append([float(by_alias[alias]["weight"]) for alias in aliases])
        responses.append(by_alias)
    decimal_means = [
        sum((Decimal(str(value)) for value in column), Decimal()) / Decimal(len(chosen))
        for column in zip(*vectors, strict=True)
    ]
    means = [float(value) for value in decimal_means]
    distances = [
        sum(((value - mean) ** 2 for value, mean in zip(vector, means, strict=True)))
        for vector in vectors
    ]
    medoid = min(range(len(chosen)), key=lambda index: (distances[index], index))
    scores, components = ([], {})
    for index, alias in enumerate(aliases):
        values = [vector[index] for vector in vectors]
        rounded = float(
            decimal_means[index].quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
        scores.append(
            {
                "candidate": alias,
                "weight": rounded,
                "rationale": responses[medoid][alias]["rationale"],
            }
        )
        components[alias] = {
            "mean": means[index],
            "rounded_weight": rounded,
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "component_weights": values,
        }
    metadata = {
        "aggregation_method": aggregation_method(required),
        "replicate_count": len(chosen),
        "medoid_replica_id": chosen[medoid]["replica_id"],
        "valid_replica_ids": [row["replica_id"] for row in chosen],
        "components": components,
    }
    return ({"scores": scores}, metadata)


VERSION = "semantic_weight_speculative_first8.v1"
AGGREGATION_METHOD = "first_8_valid_mean_v1"
DEFAULT_BASE_URL = "http://dgx008:50001/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REQUIRED_REPLICAS = 8
MAX_FANOUT = 128
MAX_ROUND_REQUESTS = 9
if __name__ == "__main__":
    main()
