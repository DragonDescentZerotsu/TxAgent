"""Choose IID/OOD regimes, evaluation queries, and their reference rankings."""
from collections import Counter,defaultdict,deque
from itertools import islice,zip_longest
from statistics import median
from .storage import PACKAGE,read_json
from .targets import stable_sha256
from .sampling import edge
from .sampling.safety import _fingerprints
from rdkit import DataStructs

def support_reason(members, train, ood):
    evidence = members if ood else train
    if len({row['parent_id'] for row in evidence}) < (4 if ood else 6):
        return 'insufficient_parent_support'
    if not ood and len(train) < 12:
        return 'fewer_than_12_train_records'
    return None


def support_diagnostics(members, train):
    return {'bucket_total_record_count': len(members), 'bucket_train_record_count': len(train),
            'bucket_total_parent_count': len({row['parent_id'] for row in members}),
            'bucket_train_parent_count': len({row['parent_id'] for row in train})}


def select_record_evaluation(rows, gate, accepted, policy):
    if not policy['evaluation_levels']:
        return [], {}
    train_parents = defaultdict(set)
    for row in rows:
        if row['dataset_split'] == 'train':
            train_parents[row['level']].add(row['parent_id'])
    test = [row for row in test_records(rows, gate, accepted, policy['ood_bucket_records']) if row['level'] in policy['evaluation_levels']]
    if policy['task'] == 'BBB_Martins':
        return test, {'validation':select_panel(rows, train_parents),
                      'test':select_panel(test, train_parents, 'test')}
    panels = {}
    for split, values in (('validation',rows),('test',test)):
        for level in policy['evaluation_levels']:
            candidates = [r for r in values if r['level'] == level and r['dataset_split'] in ('validation','ood','test')
                          and (split == 'test' or r['validation_parent_eligible'])]
            if policy['task'] == 'Bioavailability_Ma':
                panels[f'{level}/{split}'] = oral_panel([r for r in values if r['level']==level],split)
            else:
                panels[f'{level}/{split}'] = skin_panel(candidates, f'query_{split}')
    return test, panels

def select_panel(rows, train_parents, split="validation"):
    strata = defaultdict(lambda: defaultdict(list))
    buckets = defaultdict(list)
    for row in rows:
        row[f"query_{split}"] = False
        allowed = row["dataset_split"] == "test" if split == "test" else (
            row["dataset_split"] in {"validation", "ood"} and
            row["validation_parent_eligible"])
        if allowed and row["parent_id"] not in train_parents[row["level"]]:
            buckets[row["pair_bucket_key"]].append(row)
    for bucket, candidates in buckets.items():
        first = candidates[0]
        family = "binary" if first["measurement_kind"] == "binary" else "continuous"
        strata[(first["evaluation_regime"], family)][first["level"]].append((bucket, candidates))
    report = {}
    for (regime, family), levels in sorted(strata.items()):
        sizes = sorted(len(items) for items in levels.values())
        cap = sizes[min(1, len(sizes) - 1)]
        for level, items in sorted(levels.items()):
            expanded = split == "validation" and level != "L4"
            support = "bucket_train_record_count" if regime == "id" else "bucket_total_record_count"
            items.sort(key=lambda item: (-item[1][0][support], stable_sha256(f"v24:bucket:{item[0]}")))
            selected, count = items if expanded else items[:cap], 0
            for _, candidates in selected:
                candidates.sort(key=lambda r: (stable_sha256(f"v24:query:{r['record_id']}"), r["record_id"]))
                if expanded:
                    by_parent = {}
                    for row in candidates:
                        by_parent.setdefault(row["parent_id"], row)
                    candidates = list(by_parent.values())
                else:
                    candidates = candidates[:5 if level in {"L3", "L4"} else 3]
                for row in candidates:
                    row[f"query_{split}"] = True
                count += len(candidates)
            report[f"{level}/{regime}/{family}"] = {"queries": count, "selected_buckets": len(selected),
                "available_buckets": len(items), "bucket_cap": None if expanded else cap}
    return report


def test_records(rows, testing, accepted, ood_range=(9, 11)):
    train, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["dataset_split"] == "train":
            train[row["pair_bucket_key"]].append(row)
    for row in testing:
        groups[row["pair_bucket_key"]].append(row)
    output = []
    for bucket, members in sorted(groups.items()):
        training = train[bucket]
        ood = not training
        if ood and not ood_range[0] <= len(members) <= ood_range[1] \
                or support_reason(members, training, ood):
            continue
        for row in members:
            output.append({**row, **support_diagnostics(members, training), "dataset_split": "test",
                           "evaluation_regime": "ood" if ood else "id", "query_validation": False})
    return output

def _panel_target(count: int) -> int:
    return 20 if count >= 20 else 10 if count >= 10 else count


def _minimum_cost(candidates: list[dict], validation: int, testing: int) -> tuple[str, ...]:
    states = {(0, 0): (0, 0, (), ())}
    for item in sorted(candidates, key=lambda x: (stable_sha256(f"skin:ood:{x['bucket']}"), x["bucket"])):
        updated = dict(states); token = stable_sha256(f"skin:ood:{item['bucket']}")
        for (valid, test), value in states.items():
            state = min(validation, valid + item["validation"]), min(testing, test + item["test"])
            choice = (value[0] + item["cost"], value[1] + 1,
                      value[2] + (token,), value[3] + (item["bucket"],))
            if state not in updated or choice[:3] < updated[state][:3]:
                updated[state] = choice
        states = updated
    return states[(validation, testing)][3]


def choose_skin_ood(rows: list[dict], candidates: dict, levels, ood_range) -> tuple[set[str], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_bucket_key"]].append(row)
    selected, report = set(), {}
    for level in levels:
        options = []
        for bucket, evidence in groups.items():
            if evidence[0]["level"] != level or (bucket, True) not in candidates:
                continue
            non_test = [row for row in evidence if row["dataset_split"] != "test_gate_only"]
            test = [row for row in evidence if row["dataset_split"] == "test_gate_only"]
            valid_test = ood_range[0] <= len(test) <= ood_range[1] and len({r["parent_id"] for r in test}) >= 4
            options.append({"bucket": bucket, "validation": sum(r["dataset_split"] == "validation" for r in non_test),
                            "test": len(test) if valid_test else 0,
                            "cost": sum(r["dataset_split"] == "train" for r in non_test)})
        available = {"validation": sum(x["validation"] for x in options), "test": sum(x["test"] for x in options)}
        targets = {name: _panel_target(count) for name, count in available.items()}
        chosen = _minimum_cost(options, targets["validation"], targets["test"])
        selected.update(chosen)
        report[level] = {"available_queries": available, "targets": targets,
                         "selected": [item for item in options if item["bucket"] in chosen]}
    return selected, report


def skin_panel(rows: list[dict], flag: str, query_cap=500) -> dict:
    report = {}
    for regime in ("id", "ood"):
        buckets = defaultdict(list)
        for row in rows:
            if row["evaluation_regime"] == regime:
                buckets[row["pair_bucket_key"]].append(row)
        support = "bucket_train_record_count" if regime == "id" else "bucket_total_record_count"
        ordered = sorted(buckets, key=lambda key: (-buckets[key][0][support], stable_sha256(key), key))
        queues = [sorted(buckets[key], key=lambda row: (stable_sha256(row["record_id"]), row["record_id"]))
                  for key in ordered]
        selected = list(islice((row for batch in zip_longest(*queues) for row in batch if row), query_cap))
        for row in selected:
            row[flag] = True
        report[regime] = {"available_queries": sum(map(len, buckets.values())),
                          "selected_queries": len(selected), "available_buckets": len(buckets),
                          "selected_buckets": len({row["pair_bucket_key"] for row in selected})}
    report["selected_queries"] = sum(value["selected_queries"] for value in report.values()
                                     if isinstance(value, dict) and "selected_queries" in value)
    return report

def oral_panel(rows, split, budget=500):
    """Select one record per bucket per round, prioritizing new parents locally."""
    groups = defaultdict(list)
    for row in rows:
        row[f"query_{split}"] = False
        eligible = row["dataset_split"] == split
        eligible |= split == "validation" and row["dataset_split"] == "ood" and row["validation_parent_eligible"]
        if eligible:
            groups[row["pair_bucket_key"]].append(row)
    queues = []
    for bucket in sorted(groups, key=lambda b: (stable_sha256(f"oral-v24.1:bucket:{b}"), b)):
        seen, first, extra = set(), [], []
        for row in sorted(groups[bucket], key=lambda r: (stable_sha256(
                f"oral-v24.1:query:{r['record_id']}"), r["record_id"])):
            (extra if row["parent_id"] in seen else first).append(row)
            seen.add(row["parent_id"])
        queues.append(deque(first + extra))
    selected = []
    while queues and len(selected) < budget:
        for queue in queues:
            if len(selected) == budget:
                break
            row = queue.popleft()
            row[f"query_{split}"] = True
            selected.append(row)
        queues = [queue for queue in queues if queue]
    counts = Counter(r["pair_bucket_key"] for r in selected)
    return {"eligible_buckets": len(groups), "eligible_records": sum(map(len, groups.values())),
        "selected_buckets": len(counts), "queries": len(selected), "shortfall": budget - len(selected),
        "distinct_parents": len({r["parent_id"] for r in selected}),
        "bucket_parents": len({(r["pair_bucket_key"], r["parent_id"]) for r in selected}),
        "queries_per_bucket": dict(sorted(counts.items())),
        "query_distribution": {"min": min(counts.values(), default=0),
            "median": median(counts.values()) if counts else 0, "max": max(counts.values(), default=0)},
        "composition": dict(Counter(f"{r['evaluation_regime']}/{r['measurement_kind']}" for r in selected))}

def record_ranking(rows, calibration, rejected_levels, policy):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_bucket_key']].append(row)
    fps = _fingerprints(rows)
    output = []
    for split, flag in (('validation_ranking','query_validation'),('test_ranking','query_test')):
        for query in rows:
            if not query.get(flag) or query['measurement_kind']=='binary' and query['level'] in rejected_levels:
                continue
            ood = query['evaluation_regime'] == 'ood'
            candidates = [row for row in groups[query['pair_bucket_key']] if row['record_id'] != query['record_id']
                          and (row['dataset_split'] == query['dataset_split'] if ood else row['dataset_split'] == 'train')]
            selected = sorted(candidates,key=lambda row:(-DataStructs.TanimotoSimilarity(
                fps[query['parent_id']],fps[row['parent_id']]),row['record_id']))
            if not ood:
                selected = selected[:50]
            expected = query['bucket_total_record_count']-1 if ood else min(50,query['bucket_train_record_count'])
            if len(selected)!=expected or not ood and expected<12:
                raise ValueError(f'Incomplete ranking group: {query["record_id"]}')
            for rank,reference in enumerate(selected,1):
                similarity=DataStructs.TanimotoSimilarity(fps[query['parent_id']],fps[reference['parent_id']])
                output.append(edge(reference,query,split,'indirect',similarity,
                                   calibration[query['pair_bucket_key']],policy,rank=rank))
    return output

def direct_ranking(groups, calibration, policy):
    all_rows = [row for values in groups.values() for row in values]
    fps, output = _fingerprints(all_rows), []
    parents = defaultdict(list)
    for row in groups['train']:
        parents[row['parent_id']].append(row)
    for source, split in (('valid','validation_ranking'),('test','test_ranking')):
        for query in groups[source]:
            ordered = sorted(parents, key=lambda parent: (-DataStructs.TanimotoSimilarity(
                fps[query['parent_id']], fps[parent]), parent))[:75]
            for rank, parent in enumerate(ordered, 1):
                similarity = DataStructs.TanimotoSimilarity(fps[query['parent_id']], fps[parent])
                for reference in sorted(parents[parent], key=lambda row: row['record_id']):
                    output.append(edge(reference, query, split, 'direct', similarity, calibration, policy, rank=rank))
    return output
