"""Reserve separate binary and continuous molecule budgets in minority-first family order."""

from collections import Counter, defaultdict
import heapq

import numpy as np
from rdkit import DataStructs

from assay_transfer.sampling import balance as fair
from assay_transfer.targets import effective_value

PHASES = (("binary", True), ("continuous", True), ("continuous", False), ("binary", False))


def budget_key(row):
    return ("binary" if row["measurement_kind"] == "binary" else "continuous"), row["parent_id"]


def initialize(rows, calibration, caps):
    usage = {role: Counter() for role in ("query", "reference")}
    state = fair.initialize(rows, calibration, caps, usage)
    state.update(index={r["record_id"]: i for i, r in enumerate(rows)}, calibration=calibration,
                 parent_usage={role: Counter() for role in usage}, pools={}, minority={}, edges=[],
                 binary_values=defaultdict(set))
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        value = effective_value(row, calibration[row["pair_bucket_key"]])
        grouped[row["pair_bucket_key"]][row["parent_id"]][value].append(row["record_id"])
        if row["measurement_kind"] == "binary":
            state["binary_values"][row["pair_bucket_key"]].add(value)
    state["values"] = grouped
    state["parents"] = defaultdict(list)
    for row in rows:
        state["parents"][budget_key(row)].append(row["record_id"])
    for row in rows:
        choices = pools(row, state)
        counts = [len(choices[label]) - int(row["parent_id"] in choices[label]) for label in (0, 1)]
        state["minority"][row["record_id"]] = (int(counts[1] < counts[0]), min(counts))
    return state


def pools(query, state):
    bucket = query["pair_bucket_key"]
    entry = state["calibration"][bucket]
    value = effective_value(query, entry)
    key = bucket, value
    if key not in state["pools"]:
        if len(state["pools"]) >= 128:
            state["pools"].pop(next(iter(state["pools"])))
        sd = entry.get("logit_calibration_sample_standard_deviation") or entry.get("calibration_sample_standard_deviation")
        output = [{}, {}]
        for parent, values in state["values"][bucket].items():
            labels = [[], []]
            for other in values:
                near = value == other if query["measurement_kind"] == "binary" else abs(value - other) / sd <= entry["numeric_center_sd"]
                labels[int(near)].append(other)
            for label in (0, 1):
                if labels[label]:
                    output[label][parent] = labels[label]
        state["pools"][key] = output
    return state["pools"][key]


def phase_limits(state, family, minority):
    query, reference = {}, {}
    for key, row in state["records"].items():
        used = state["usage"]["query"][key]
        active = ("binary" if row["measurement_kind"] == "binary" else "continuous") == family
        limit = state["limits"]["query"][key]
        if minority:
            target = min((limit + 1) // 2, state["minority"][key][1])
        else:
            target = min(limit, 4 * used) if family == "binary" else limit
        query[key] = max(used, target) if active else used
        reference[key] = state["limits"]["reference"][key] if active else state["usage"]["reference"][key]
    return {"query": query, "reference": reference}


def reference_heaps(state, quotas, targets):
    heaps = {}
    for bucket, parents in state["values"].items():
        for parent, values in parents.items():
            for value, keys in values.items():
                heap = [fair.reference_priority(key, state, targets) for key in keys
                        if state["usage"]["reference"][key] < quotas[key]]
                heapq.heapify(heap)
                heaps[bucket, parent, value] = heap
                if state["records"][keys[0]]["measurement_kind"] == "binary":
                    heaps.setdefault((bucket, value), []).extend(heap)
    for key, heap in heaps.items():
        if len(key) == 2:
            heapq.heapify(heap)
    return heaps


def binary_choices(query, label, state, heaps, quotas, targets, count, fps=None):
    bucket = query["pair_bucket_key"]
    value = effective_value(query, state["calibration"][bucket])
    candidates = [(bucket, other) for other in state["binary_values"][bucket]
                  if (other == value) == bool(label)]
    if not candidates:
        return []
    if len(candidates) != 1:
        raise ValueError("Binary sampling requires exactly two category values")
    seen = state["seen"][query["record_id"]] | {query["parent_id"]}
    if fps is None:
        heap, stashed, output = heaps[candidates[0]], [], []
        while len(output) < count:
            result = fair.available_reference(heap, state, quotas, targets)
            if result is None:
                break
            reference, priority = result
            heapq.heappop(heap)
            parent = reference["parent_id"]
            if state["parent_usage"]["reference"][(budget_key(query)[0], parent)] >= fair.PARENT_CAP:
                continue
            stashed.append(priority)
            if parent not in seen:
                output.append((priority, reference))
                seen.add(parent)
        for priority in stashed:
            heapq.heappush(heap, priority)
        return output
    output = []
    for parent, values in state["values"][bucket].items():
        if parent in seen or state["parent_usage"]["reference"][(budget_key(query)[0], parent)] >= fair.PARENT_CAP:
            continue
        choices = [fair.available_reference(heaps[bucket, parent, value], state, quotas, targets)
                   for value in values if (value == query["geometry_value"]) == bool(label)]
        choices = [choice for choice in choices if choice is not None]
        if choices:
            reference, priority = min(choices, key=lambda item: item[1])
            similarity = DataStructs.TanimotoSimilarity(fps[query["parent_id"]], fps[parent])
            output.append(((-similarity, priority), reference))
    return sorted(output, key=lambda item: item[0])[:count]


def choices(query, label, state, heaps, quotas, targets, count, fps=None):
    if query["measurement_kind"] == "binary":
        return binary_choices(query, label, state, heaps, quotas, targets, count, fps)
    output = []
    for parent, values in pools(query, state)[label].items():
        if (parent == query["parent_id"] or parent in state["seen"][query["record_id"]]
                or state["parent_usage"]["reference"][(budget_key(query)[0], parent)] >= fair.PARENT_CAP):
            continue
        candidates = []
        for value in values:
            result = fair.available_reference(heaps[query["pair_bucket_key"], parent, value],
                                              state, quotas, targets)
            if result is not None:
                reference, priority = result
                candidates.append((priority, reference))
        if candidates:
            priority, reference = min(candidates, key=lambda item: item[0])
            similarity = (DataStructs.TanimotoSimilarity(
                fps[query["parent_id"]], fps[parent]) if fps is not None else None)
            output.append(((-similarity, priority) if fps is not None else priority, reference))
    return sorted(output, key=lambda item: item[0])


def sample_pass(state, quotas, targets, family, minority, fps=None):
    heaps = reference_heaps(state, quotas["reference"], targets)
    usage = state["usage"]
    schedule = sorted(state["records"], key=lambda key: (
        usage["query"][key] + usage["reference"][key] > 0, usage["query"][key], state["order"][key]))
    for key in schedule:
        query = state["records"][key]
        count = min(quotas["query"][key] - usage["query"][key],
                    fair.PARENT_CAP - state["parent_usage"]["query"][budget_key(query)])
        if count <= 0:
            continue
        label = state["minority"][key][0] if minority else 1 - state["minority"][key][0]
        available = choices(query, label, state, heaps, quotas["reference"], targets, count, fps)
        for _, reference in available[:count]:
            ref = reference["record_id"]
            state["seen"][key].add(reference["parent_id"])
            for role, row in (("query", query), ("reference", reference)):
                usage[role][row["record_id"]] += 1
                state["parent_usage"][role][budget_key(row)] += 1
            state["near"][key] += label
            state["edges"].append((state["index"][key], state["index"][ref], label))


def sample(rows, calibration, caps, fps=None):
    state = initialize(rows, calibration, caps)
    report = []
    for family, minority in PHASES:
        limits = phase_limits(state, family, minority)
        initial = len(state["edges"])
        while True:
            before = len(state["edges"])
            usage = state["usage"]
            quotas = {role: fair.allocate(state["parents"], limits[role], usage[role],
                      usage["reference" if role == "query" else "query"], state["order"])
                      for role in usage}
            sample_pass(state, quotas, quotas["reference"], family, minority, fps)
            sample_pass(state, {"query": quotas["query"], "reference": limits["reference"]},
                        quotas["reference"], family, minority, fps)
            if len(state["edges"]) == before:
                break
        report.append({"family": family, "pair_type": "minority" if minority else "majority",
                       "pairs": len(state["edges"]) - initial,
                       "parent_family_budgets_at_cap": {role: sum(v == fair.PARENT_CAP for v in counts.values())
                                          for role, counts in state["parent_usage"].items()}})
        print("Oral V24.2 sampling phase", report[-1], flush=True)
    return np.asarray(state["edges"], dtype=np.int32).reshape(-1, 3), report
