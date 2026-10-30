"""Audit whether progressive DeepSeek traces analyze and adopt new evidence.

The audit deliberately separates evidence visibility, free-text analysis,
structured citation, decision-basis adoption, and prediction changes.  It reads
only frozen run artifacts and writes deterministic TSV/JSONL summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.paper_experiments.paired_binary_predictions import (
    paired_binary_summary,
)


DEFAULT_TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
TASKS = (*DEFAULT_TASKS, "ames", "dili", "carcinogens")
PREDICTION_FIELDS = {
    "bbb_martins": "bbb_prediction",
    "bioavailability_ma": "bioavailability_prediction",
    "skin_reaction": "skin_reaction_prediction",
    "ames": "ames_prediction",
    "dili": "dili_prediction",
    "carcinogens": "carcinogenicity_prediction",
}

TASK_DISPLAY = {
    "bbb_martins": "BBB",
    "bioavailability_ma": "Bioavailability",
    "skin_reaction": "Skin",
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _flatten_card_ids(state: dict[str, Any]) -> set[str]:
    cited: set[str] = set()
    for field in ("supportive_card_ids", "contradictory_card_ids", "prediction_basis_card_ids"):
        cited.update(state.get(field) or [])
    for item in state.get("claims") or []:
        cited.update(item.get("card_ids") or [])
    for item in state.get("new_evidence_assessment") or []:
        cited.update(item.get("card_ids") or [])
    return cited


def _card_catalog(prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for analog in (prepared.get("active_evidence") or {}).values():
        for card_id, card in (analog.get("cards") or {}).items():
            cards[card_id] = card
    return cards


def _confidence_delta(previous: str | None, current: str | None) -> int:
    order = {"low": 0, "moderate": 1, "high": 2}
    if previous not in order or current not in order:
        return 0
    return order[current] - order[previous]


def _rows_for_task(run_root: Path, task: str) -> list[dict[str, Any]]:
    task_root = run_root / task
    prediction_field = PREDICTION_FIELDS[task]
    labels_by_level: dict[int, dict[int, dict[str, Any]]] = {}
    for path in sorted((task_root / "levels").glob("level_*/predictions.jsonl")):
        level = int(path.parent.name.split("_")[-1])
        labels_by_level[level] = {int(row["index"]): row for row in _read_jsonl(path)}

    rows: list[dict[str, Any]] = []
    for query_dir in sorted((task_root / "queries").glob("query_idx*")):
        query_index = int(query_dir.name.removeprefix("query_idx"))
        previous_state: dict[str, Any] | None = None
        previous_correct: bool | None = None
        for level_dir in sorted(
            (query_dir / "levels").glob("level_*"),
            key=lambda path: int(path.name.split("_")[-1]),
        ):
            level = int(level_dir.name.split("_")[-1])
            prepared = _read_json(level_dir / "prepared.json")
            output = _read_json(level_dir / "output.json")
            state = output.get("state") or {}
            prediction_row = labels_by_level[level][query_index]
            label = int(prediction_row["label"])
            correct = bool(prediction_row["correct"])
            model_called = bool(output.get("model_called"))
            new_ids = set(prepared.get("new_card_ids") or [])
            catalog = _card_catalog(prepared)
            level_definition = prepared.get("level_definition") or {}
            current_family = level_definition.get("endpoint_group") or level_definition.get("family_id", "")

            reasoning = ""
            aliases_for_new: list[str] = []
            if model_called:
                request = _read_json(level_dir / "request.json")
                alias_map = request.get("card_alias_map") or {}
                aliases_for_new = sorted(alias for alias, stable in alias_map.items() if stable in new_ids)
                reasoning = str((output.get("llm") or {}).get("reasoning_content") or "")
            reasoning_mentions_new = [alias for alias in aliases_for_new if alias in reasoning]

            structured_ids = _flatten_card_ids(state)
            basis_ids = set(state.get("prediction_basis_card_ids") or [])
            assessment_ids = {
                card_id
                for item in state.get("new_evidence_assessment") or []
                for card_id in (item.get("card_ids") or [])
            }
            decision_effects = sorted(
                {
                    str(item.get("decision_effect"))
                    for item in state.get("new_evidence_assessment") or []
                    if item.get("decision_effect")
                }
            )
            indirect_basis_ids = {
                card_id
                for card_id in basis_ids
                if int((catalog.get(card_id) or {}).get("first_seen_level") or 1) >= 2
            }
            current_prediction = state.get(prediction_field)
            previous_prediction = previous_state.get(prediction_field) if previous_state else None
            label_flip = previous_prediction is not None and current_prediction != previous_prediction
            if not label_flip:
                flip_outcome = "no_flip"
            elif previous_correct is False and correct is True:
                flip_outcome = "beneficial_flip"
            elif previous_correct is True and correct is False:
                flip_outcome = "harmful_flip"
            else:
                flip_outcome = "wrong_to_wrong_flip"

            rows.append(
                {
                    "task": task,
                    "query_index": query_index,
                    "level": level,
                    "family": current_family,
                    "label": label,
                    "prediction": current_prediction,
                    "correct": correct,
                    "confidence": state.get("confidence"),
                    "status": output.get("status"),
                    "model_called": model_called,
                    "n_active_molecules": int(prepared.get("n_active_molecules") or 0),
                    "n_active_cards": int(prepared.get("n_active_cards") or 0),
                    "n_new_cards": len(new_ids),
                    "n_new_molecules": len(
                        {
                            analog_id
                            for analog_id, analog in (prepared.get("active_evidence") or {}).items()
                            if new_ids.intersection((analog.get("cards") or {}).keys())
                        }
                    ),
                    "reasoning_chars": len(reasoning),
                    "n_new_aliases_mentioned_in_reasoning": len(reasoning_mentions_new),
                    "reasoning_mentions_any_new": bool(reasoning_mentions_new),
                    "n_new_cards_structured_cited": len(new_ids & structured_ids),
                    "structured_cites_any_new": bool(new_ids & structured_ids),
                    "n_new_cards_assessed": len(new_ids & assessment_ids),
                    "assessment_cites_any_new": bool(new_ids & assessment_ids),
                    "n_new_cards_in_basis": len(new_ids & basis_ids),
                    "new_card_in_basis": bool(new_ids & basis_ids),
                    "n_indirect_cards_in_basis": len(indirect_basis_ids),
                    "any_indirect_card_in_basis": bool(indirect_basis_ids),
                    "decision_effects": ",".join(decision_effects),
                    "revision_action": state.get("revision_action"),
                    "previous_prediction": previous_prediction,
                    "previous_correct": previous_correct,
                    "label_flip": label_flip,
                    "flip_outcome": flip_outcome,
                    "confidence_delta": _confidence_delta(
                        previous_state.get("confidence") if previous_state else None,
                        state.get("confidence"),
                    ),
                    "query_dir": str(query_dir.resolve()),
                }
            )
            previous_state = state
            previous_correct = correct
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["level"] >= 2:
            grouped[(row["task"], row["level"], row["family"])].append(row)
    result: list[dict[str, Any]] = []
    for (task, level, family), group in sorted(grouped.items()):
        calls = [row for row in group if row["model_called"]]
        with_new = [row for row in calls if row["n_new_cards"] > 0]
        outcome_counts = Counter(row["flip_outcome"] for row in group)
        result.append(
            {
                "task": task,
                "level": level,
                "family": family,
                "queries": len(group),
                "model_calls": len(calls),
                "carry_forward": len(group) - len(calls),
                "calls_with_new_cards": len(with_new),
                "reasoning_mentions_new": sum(row["reasoning_mentions_any_new"] for row in with_new),
                "reasoning_mention_rate": _rate(
                    sum(row["reasoning_mentions_any_new"] for row in with_new), len(with_new)
                ),
                "structured_cites_new": sum(row["structured_cites_any_new"] for row in with_new),
                "structured_citation_rate": _rate(
                    sum(row["structured_cites_any_new"] for row in with_new), len(with_new)
                ),
                "assessment_cites_new": sum(row["assessment_cites_any_new"] for row in with_new),
                "assessment_rate": _rate(
                    sum(row["assessment_cites_any_new"] for row in with_new), len(with_new)
                ),
                "new_card_in_basis": sum(row["new_card_in_basis"] for row in with_new),
                "new_basis_adoption_rate": _rate(
                    sum(row["new_card_in_basis"] for row in with_new), len(with_new)
                ),
                "any_indirect_in_basis": sum(row["any_indirect_card_in_basis"] for row in group),
                "indirect_basis_rate_all_queries": _rate(
                    sum(row["any_indirect_card_in_basis"] for row in group), len(group)
                ),
                "label_flips": sum(row["label_flip"] for row in group),
                "beneficial_flips": outcome_counts["beneficial_flip"],
                "harmful_flips": outcome_counts["harmful_flip"],
                "wrong_to_wrong_flips": outcome_counts["wrong_to_wrong_flip"],
                "correct": sum(row["correct"] for row in group),
                "accuracy": _rate(sum(row["correct"] for row in group), len(group)),
            }
        )
    return result


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prediction_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row.get("index", row.get("query_index")))
        result[index] = row
    return result


def _stage_predictions(
    run_root: Path,
    task: str,
    none_root: Path | None,
) -> list[tuple[str, dict[int, dict[str, Any]]]]:
    stages: list[tuple[str, dict[int, dict[str, Any]]]] = []
    if none_root is not None:
        stages.append(("None", _prediction_rows(none_root / task / "none" / "predictions.jsonl")))
    for path in sorted(
        (run_root / task / "levels").glob("level_*/predictions.jsonl"),
        key=lambda value: int(value.parent.name.split("_")[-1]),
    ):
        level = int(path.parent.name.split("_")[-1])
        stages.append((f"L{level}", _prediction_rows(path)))
    return stages


def _prediction_flip_audit(
    run_roots: dict[str, Path],
    none_root: Path | None,
    tasks: tuple[str, ...] = TASKS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    transitions: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for task in tasks:
        stages = _stage_predictions(run_roots[task], task, none_root)
        if len(stages) < 2:
            continue
        stage_maps = dict(stages)
        common_indices = sorted(set.intersection(*(set(rows) for _, rows in stages)))
        comparisons = [
            (left_name, right_name, "sequential")
            for (left_name, _), (right_name, _) in zip(stages, stages[1:], strict=False)
        ]
        if len(stages) > 2:
            comparisons.append(("L1", stages[-1][0], "indirect_total"))
        if stages[0][0] == "None" and len(stages) > 2:
            comparisons.append(("None", stages[-1][0], "cumulative_total"))

        for left_name, right_name, comparison_type in comparisons:
            left_rows = stage_maps[left_name]
            right_rows = stage_maps[right_name]
            labels = [int(left_rows[index]["label"]) for index in common_indices]
            left = [int(left_rows[index]["pred_label"]) for index in common_indices]
            right = [int(right_rows[index]["pred_label"]) for index in common_indices]
            paired = paired_binary_summary(labels, left, right)
            directions = Counter(
                f"{left_prediction}_to_{right_prediction}"
                for left_prediction, right_prediction in zip(left, right, strict=True)
                if left_prediction != right_prediction
            )
            transitions.append(
                {
                    "task": TASK_DISPLAY[task],
                    "task_id": task,
                    "comparison_type": comparison_type,
                    "from_stage": left_name,
                    "to_stage": right_name,
                    "n": paired["n"],
                    "from_macro_f1": paired["left_macro_f1"],
                    "to_macro_f1": paired["right_macro_f1"],
                    "delta_macro_f1": paired["delta_macro_f1"],
                    "delta_macro_f1_ci_low": paired["delta_macro_f1_bootstrap_95ci"][0],
                    "delta_macro_f1_ci_high": paired["delta_macro_f1_bootstrap_95ci"][1],
                    "prediction_flips": paired["prediction_flips"],
                    "beneficial_flips": paired["right_only_correct"],
                    "harmful_flips": paired["left_only_correct"],
                    "net_correct": paired["right_only_correct"] - paired["left_only_correct"],
                    "zero_to_one": directions["0_to_1"],
                    "one_to_zero": directions["1_to_0"],
                    "mcnemar_exact_p": paired["mcnemar_exact_p"],
                }
            )
            for index, label, left_prediction, right_prediction in zip(
                common_indices, labels, left, right, strict=True
            ):
                if left_prediction == right_prediction:
                    continue
                left_correct = left_prediction == label
                right_correct = right_prediction == label
                details.append(
                    {
                        "task": TASK_DISPLAY[task],
                        "task_id": task,
                        "comparison_type": comparison_type,
                        "from_stage": left_name,
                        "to_stage": right_name,
                        "query_index": index,
                        "label": label,
                        "from_prediction": left_prediction,
                        "to_prediction": right_prediction,
                        "flip_outcome": (
                            "beneficial" if right_correct else "harmful" if left_correct else "wrong_to_wrong"
                        ),
                    }
                )

        none_prediction_by_index = stage_maps.get("None")
        for index in common_indices:
            sequence = [int(rows[index]["pred_label"]) for _, rows in stages]
            n_flips = sum(left != right for left, right in zip(sequence, sequence[1:], strict=False))
            trajectories.append(
                {
                    "task": TASK_DISPLAY[task],
                    "task_id": task,
                    "query_index": index,
                    "label": int(stages[0][1][index]["label"]),
                    "stage_predictions": ">".join(map(str, sequence)),
                    "n_flips": n_flips,
                    "any_flip": n_flips > 0,
                    "multiple_flips": n_flips > 1,
                    "final_reverted_to_none": bool(
                        none_prediction_by_index is not None
                        and n_flips > 0
                        and sequence[-1] == sequence[0]
                    ),
                }
            )
    return transitions, details, trajectories


def _trajectory_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    result: list[dict[str, Any]] = []
    for task, group in grouped.items():
        flip_counts = Counter(int(row["n_flips"]) for row in group)
        result.append(
            {
                "task": task,
                "n": len(group),
                "queries_with_any_flip": sum(row["any_flip"] for row in group),
                "queries_with_multiple_flips": sum(row["multiple_flips"] for row in group),
                "queries_reverted_to_none": sum(row["final_reverted_to_none"] for row in group),
                "zero_flips": flip_counts[0],
                "one_flip": flip_counts[1],
                "two_flips": flip_counts[2],
                "three_or_more_flips": sum(count for flips, count in flip_counts.items() if flips >= 3),
            }
        )
    return result


def _parse_task_roots(
    values: list[str],
    default: Path | None,
    tasks: tuple[str, ...] = TASKS,
) -> dict[str, Path]:
    roots = {task: default for task in tasks if default is not None}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or task not in TASKS:
            raise ValueError(f"Expected TASK=PATH with TASK in {TASKS}: {value}")
        roots[task] = Path(path)
    missing = [task for task in tasks if task not in roots]
    if missing:
        raise ValueError(f"Missing run roots for: {', '.join(missing)}")
    return {task: roots[task] for task in tasks}


def citation_role_metrics(content, cards):
    """Distinct-record citations grouped by saved retrieval parent, not polarity labels."""
    roles = {}
    for role in ('supportive', 'contradictory'):
        refs = content.get(role + '_card_ids')
        if refs is None:
            refs = [ref for claim in content.get('claims', [])
                    if claim.get('evidence_role') == role for ref in claim.get('card_ids', [])]
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError('citation roles must contain card-ID arrays')
        roles[role] = set(refs)
        if roles[role] - cards.keys():
            raise ValueError('citation refers to a card absent from the visible context')
    roles['combined'] = roles['supportive'] | roles['contradictory']
    metrics = {}
    for role, ids in roles.items():
        metrics[role] = {}
        for kind in ('all', 'direct', 'indirect'):
            visible = {key for key, card in cards.items() if kind == 'all' or card['kind'] == kind}
            counts = Counter(cards[key]['parent'] for key in ids & visible)
            n = sum(counts.values())
            metrics[role][kind] = {
                'records': n, 'parents': len(counts),
                'effective_parents': n * n / sum(c*c for c in counts.values()) if n else None,
                'visible_records': len(visible),
                'record_fraction': n / len(visible) if visible else None,
            }
    return {
        'roles': metrics,
        'citations': {role: sorted(ids) for role, ids in roles.items()},
        'overlap_records': len(roles['supportive'] & roles['contradictory']),
        'overlap_parents': len({cards[k]['parent'] for k in roles['supportive']}
                               & {cards[k]['parent'] for k in roles['contradictory']}),
    }


def _budgeted_citation_row(job):
    import hashlib
    import ijson
    from tools.chembl_tool.common.budgeted_evidence import content_hash

    task, setting, index, root, manifest_hash = job
    query = root/'runs'/task/f'query_{index:05d}'
    prepared_bytes = (query/'prepared.json').read_bytes()
    output_bytes = (query/'output.json').read_bytes()
    prepared, output = json.loads(prepared_bytes), json.loads(output_bytes)
    if (output.get('status') != 'ok' or prepared['manifest_hash'] != manifest_hash
            or output['request_hash'] != prepared['request_hash']
            or content_hash(prepared['messages']) != prepared['request_hash']
            or output['card_alias_map'] != prepared['card_alias_map']):
        raise ValueError(f'unverified prediction/request binding: {query}')
    # Stop before the much larger unselected candidate ledger. No new retrieval.
    selected = {}
    with (query/'selection.json').open('rb') as handle:
        for key, value in ijson.kvitems(handle, ''):
            if key in ('direct', 'selected'):
                selected[key] = value
            if len(selected) == 2:
                break
    if set(selected) != {'direct', 'selected'}:
        raise ValueError(f'missing selected evidence: {query}')
    if (sorted(r['id'] for r in selected['direct']) != sorted(prepared['direct_ids'])
            or sorted(r['id'] for r in selected['selected']) != sorted(prepared['indirect_ids'])):
        raise ValueError(f'selection differs from prepared record IDs: {query}')
    records = {r['id']: {'parent': r['molecule_id'], 'kind': r['kind']}
               for group in selected.values() for r in group}
    aliases = prepared['card_alias_map']
    if set(records) != set(aliases.values()) or len(set(aliases.values())) != len(aliases):
        raise ValueError(f'incomplete or duplicate card mapping: {query}')
    cards = {alias: records[key] for alias, key in aliases.items()}
    if any(not r['parent'] or r['kind'] not in ('direct', 'indirect') for r in cards.values()):
        raise ValueError(f'invalid selected parent/kind: {query}')
    from tools.chembl_tool.common.budgeted_reasoning import prepared_payload
    payload = prepared_payload(prepared)
    visible = {r['card_id']: r['evidence_kind'] for n in payload['neighbors'] for r in n['records']}
    if visible != {alias: r['kind'] for alias, r in cards.items()}:
        raise ValueError(f'card mapping differs from actual prompt: {query}')
    return {'task': task, 'setting': setting, 'query_index': index,
            **citation_role_metrics(output['llm']['content'], cards),
            'card_parents': cards, 'selected_parent_map_sha256': content_hash(cards),
            'prepared_sha256': hashlib.sha256(prepared_bytes).hexdigest(),
            'output_sha256': hashlib.sha256(output_bytes).hexdigest()}


def analyze_budgeted_citations(suite_root, output_dir, tasks=None):
    """Role-separated final-answer citations; no label transitions or cost analysis."""
    from statistics import mean
    from tools.chembl_tool.common.budgeted_evidence import content_hash
    from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic

    verification_path = suite_root/'verification.json'
    verified = _read_json(verification_path)
    if verified['status'] != 'complete_verified':
        raise ValueError('citation analysis requires a completely verified suite')
    rows, sources = [], []
    cells = [c for c in verified['cells'] if tasks is None or c['task'] in tasks]
    if not cells or (tasks and set(tasks) - {c['task'] for c in cells}):
        raise ValueError('requested tasks are missing from the verified suite')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for cell in cells:
            task, setting = cell['task'], cell['setting']
            root = suite_root/'runs'/setting/task
            manifest = _read_json(root/'experiment_manifest.json')
            digest = content_hash(manifest)
            sources.append({'task': task, 'setting': setting, 'manifest_hash': digest})
            rows.extend(pool.map(_budgeted_citation_row,
                                [(task, setting, i, root, digest) for i in range(cell['n'])]))
            print(f'[citations] {task}/{setting}: {cell["n"]} verified', flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['task'], row['setting']].append(row)
    summary = []
    for (task, setting), group in groups.items():
        for role in ('supportive', 'contradictory', 'combined'):
            for kind in ('all', 'direct', 'indirect'):
                values = [r['roles'][role][kind] for r in group]
                effective = [v['effective_parents'] for v in values if v['effective_parents'] is not None]
                fractions = [v['record_fraction'] for v in values if v['record_fraction'] is not None]
                summary.append({'task': task, 'setting': setting, 'role': role, 'kind': kind,
                    'queries': len(group), 'queries_with_citations': len(effective),
                    'queries_with_visible_records': len(fractions),
                    'mean_records': mean(v['records'] for v in values),
                    'mean_parents': mean(v['parents'] for v in values),
                    'mean_effective_parents_when_cited': mean(effective) if effective else None,
                    'mean_record_fraction_when_visible': mean(fractions) if fractions else None,
                    'mean_role_overlap_records': mean(r['overlap_records'] for r in group),
                    'mean_role_overlap_parents': mean(r['overlap_parents'] for r in group)})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir/'citations.per_query.jsonl', rows)
    _write_tsv(output_dir/'citations.summary.tsv', summary)
    write_json_atomic(output_dir/'manifest.json', {
        'version': 'budgeted_citation_roles.v1', 'status': 'complete', 'suite_root': str(suite_root),
        'verification_sha256': sha256_file(verification_path), 'cells': sources, 'queries': len(rows),
        'analysis_code_sha256': sha256_file(Path(__file__)),
        'semantics': 'Model-assigned final-answer roles, not gold labels or reasoning mentions. '
                     'Deduplicate record IDs per role; combined is the union, with overlaps retained. '
                     'Parents use saved retrieval molecule_id. Effective parents = N^2/sum(parent_count^2); '
                     'undefined for zero citations. Means are per query within each task/setting; '
                     'conditional denominators are explicit. Direct/indirect metrics use their own visible pools.',
        'outputs': {name: sha256_file(output_dir/name)
                    for name in ('citations.per_query.jsonl', 'citations.summary.tsv')}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument('--budgeted-suite-root', type=Path,
                        help='Analyze supportive/contradictory citations in a verified simple-baseline suite.')
    parser.add_argument(
        "--task-run-root",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override the artifact root for one task; repeat for mixed-lineage audits.",
    )
    parser.add_argument(
        "--none-root",
        type=Path,
        help="Optional cumulative-family root containing task/none/predictions.jsonl.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Restrict the audit to one or more tasks; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.budgeted_suite_root:
        if args.run_root or args.task_run_root or args.none_root:
            parser.error('--budgeted-suite-root cannot be combined with progressive run roots')
        analyze_budgeted_citations(args.budgeted_suite_root, args.output_dir, args.task)
        return

    selected_tasks = tuple(dict.fromkeys(args.task or DEFAULT_TASKS))
    run_roots = _parse_task_roots(args.task_run_root, args.run_root, selected_tasks)
    rows = [
        row
        for task in selected_tasks
        for row in _rows_for_task(run_roots[task], task)
    ]
    aggregate = _aggregate(rows)
    prediction_transitions, prediction_flip_details, trajectories = _prediction_flip_audit(
        run_roots, args.none_root, selected_tasks
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "query_level_audit.jsonl", rows)
    _write_tsv(args.output_dir / "transition_summary.tsv", aggregate)
    _write_tsv(args.output_dir / "prediction_transition_summary.tsv", prediction_transitions)
    _write_jsonl(args.output_dir / "prediction_flip_details.jsonl", prediction_flip_details)
    _write_jsonl(args.output_dir / "prediction_trajectories.jsonl", trajectories)
    _write_tsv(args.output_dir / "prediction_trajectory_summary.tsv", _trajectory_summary(trajectories))

    candidates = {
        category: sorted(
            [
                row
                for row in rows
                if row["level"] >= 2
                and (
                    row["flip_outcome"] == category
                    if category in {"beneficial_flip", "harmful_flip", "wrong_to_wrong_flip"}
                    else row["model_called"]
                    and row["n_new_cards"] > 0
                    and row["structured_cites_any_new"]
                    and not row["label_flip"]
                )
            ],
            key=lambda row: (
                -row["n_new_cards_in_basis"],
                -row["n_new_cards_structured_cited"],
                -row["reasoning_chars"],
                row["task"],
                row["query_index"],
                row["level"],
            ),
        )[:50]
        for category in ("beneficial_flip", "harmful_flip", "wrong_to_wrong_flip", "analyzed_no_flip")
    }
    (args.output_dir / "candidate_cases.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "trace_rows": len(rows),
                "adoption_transitions": len(aggregate),
                "prediction_transitions": len(prediction_transitions),
                "prediction_flip_details": len(prediction_flip_details),
                "output": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
