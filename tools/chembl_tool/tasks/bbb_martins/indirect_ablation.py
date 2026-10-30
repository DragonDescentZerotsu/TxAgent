"""Prepare the frozen TDC BBB indirect ablation; inference uses the family runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.record_budget import (
    SelectionConfig,
    content_hash,
    select_simple_records,
)

TASK = "bbb_martins"
ASSETS = Path("artifacts/chembl_tool/bbb_indirect_20260924")
PROMPT = Path(__file__).with_name("indirect_applicability_v1.txt")
ARMS = {"direct_guard": 0, "guard50": 50, "balanced20": 20, "balanced50": 50}
PREDICTION = re.compile(
    r"\bin[ -]silico\b|\bqikprop\b|swiss\s*adme|admetlab|admet[s ]?sar|"
    r"\bpredicted\b|\bprediction\b|\bcomputational\b|\badaboost\b",
    re.I,
)
MEASUREMENT = re.compile(
    r"\bmeasured\b|\bexperiment(?:al|ally)?\b|\bin vivo\b|\bin vitro\b|"
    r"microdialysis|perfusion",
    re.I,
)
PREDICTION_FILTER_VERSION = "v2"
COMPUTATION = re.compile(
    r"\bin[ -]silico\b|\bcomputational\b|\bQSAR\b|"
    r"\bPBPK\b|\bmolecular dynamics\b|\bPK[ -]PD simulation\b|"
    r"\bQikProp\b|swiss\s*adme|admetlab|admet[s ]?sar|\bAdaBoost\b|"
    r"\bpkCSM\b|BOILED[ -]Egg|\bADMET\s+(?:prediction\s+)?model\b|"
    r"\b(?:BBB|PAMPA)\s+prediction\s+model\b|"
    r"\bpredicted\s+(?:PAMPA|in[ -]vitro|in[ -]vivo)\b",
    re.I,
)
ASSAY = re.compile(
    r"\bPAMPA(?:[ -]BBB)?\b|parallel artificial membrane permea(?:bility|tion)|"
    r"\bMDCK(?:[ -]MDR1)?\b|\bCaco[ -]?2\b|\bhCMEC/D3\b|"
    r"\btranswell\b|\bmicrodialysis\b|\bperfusion\b",
    re.I,
)
OBSERVED = re.compile(
    r"\b(?:measured|experimentally determined|experimentally tested)\b", re.I
)
NEGATED_EXPERIMENT = re.compile(
    r"\b(?:not|never|no|without)\s+"
    r"(?:(?:any|direct|actual|experimentally|experimental|been|being|was|were)\s+){0,3}"
    r"(?:measur\w*|test\w*|perform\w*|conduct\w*|experiment\w*|"
    r"PAMPA|MDCK|Caco[ -]?2|perfusion|microdialysis)\b|"
    r"\b(?:measurements?|experiments?|assays?|PAMPA)\s+"
    r"(?:(?:was|were|have|has|been)\s+)*(?:not|never)\b",
    re.I,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def pin_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Changed prepared input: {path}; use a fresh output root")
    write_json_atomic(path, value)


def prediction_only(record):
    """Remove prediction-only cards; retain experimental or mixed evidence.

    Recognizes assay-based outcome classifications and ignores negated or
    computational assay mentions. This remains a conservative lexical screen,
    not a source-paper semantic audit.
    """
    card = record["card"]
    # Keep fields separate: a computational context must not hide a separate
    # measured result, and a generic numeric endpoint is not proof of an assay.
    fields = [
        card.get(key, "")
        for key in (
            "assay_context",
            "support_text",
            "qualifying_conditions",
            "extra_details",
        )
    ]
    details = card.get("experimental_details") or {}
    fields.extend(
        details.get(key, "")
        for key in (
            "assay_type",
            "biological_system",
            "measurement_method",
            "evidence_type",
        )
    )
    text = "\n".join(str(value) for value in fields if value)
    prediction_text = "\n".join(str(value) for value in fields[:3] if value)
    if not (PREDICTION.search(prediction_text) or COMPUTATION.search(text)):
        return False
    computational_context = bool(COMPUTATION.search(text))
    assay_mention_only = computational_context or bool(NEGATED_EXPERIMENT.search(text))
    for field in fields:
        for clause in re.split(r"[;\n]|(?<=[.!?])\s+|\bbut\b", str(field)):
            if NEGATED_EXPERIMENT.search(clause):
                continue
            if OBSERVED.search(clause):
                return False
            if COMPUTATION.search(clause):
                continue
            if ASSAY.search(clause) and re.search(
                r"\b(?:showed|exhibited|yielded|tested|performed|conducted|monolayers?|studies)\b",
                clause,
                re.I,
            ):
                return False
            if MEASUREMENT.search(clause) and not computational_context:
                return False
            if ASSAY.search(clause) and not assay_mention_only:
                return False
    return True


def select_indirect(candidates, original_fill_ids, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm == "direct_guard":
        return []
    if arm == "guard50":
        by_id = {record["id"]: record for record in candidates}
        return [
            by_id[key] for key in original_fill_ids if not prediction_only(by_id[key])
        ]
    eligible = [record for record in candidates if not prediction_only(record)]
    return select_simple_records(
        eligible,
        SelectionConfig(
            policy="group_balanced", budget=ARMS[arm], per_molecule=3, seed=0
        ),
    )[0]


def render(query, direct, indirect):
    """Preserve the exact historical compact prompt, ordering and card aliases."""
    records = sorted(
        direct + indirect,
        key=lambda r: (
            r["kind"] != "direct",
            -r["similarity"],
            r.get("knn_rank", 0),
            r["id"],
        ),
    )
    aliases = {f"C{i:03d}": record["id"] for i, record in enumerate(records, 1)}
    evidence = [
        dict(
            card_id=alias,
            kind=record["kind"],
            smiles=record["canonical_smiles"],
            similarity=record["similarity"],
            family=record["group"],
            source=record.get("evidence_source", ""),
            **record["card"],
        )
        for alias, record in zip(aliases, records)
    ]
    payload = {
        "query": query["query"],
        "query_prior": query["query_prior"],
        "evidence": evidence,
        "required_json_schema": query["required_json_schema"],
    }
    messages = [
        {"role": "system", "content": PROMPT.read_text()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return messages, aliases, payload


def prepare(bundle_path, output_root, arms):
    bundle_path, output_root = Path(bundle_path), Path(output_root)
    receipt = read_json(bundle_path.with_suffix(".manifest.json"))
    if sha256_file(bundle_path) != receipt["sha256"]:
        raise ValueError("Frozen input bundle hash mismatch")
    with gzip.open(bundle_path, "rt") as stream:
        bundle = json.load(stream)
    if content_hash(PROMPT.read_text()) != bundle["system_prompt_hash"]:
        raise ValueError("Prompt changed; create a new versioned experiment")
    benchmark_rows = bundle["evaluation_rows"]
    input_path = (output_root / "test.jsonl").resolve()
    input_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark_rows
    )
    if input_path.exists() and input_path.read_text() != input_text:
        raise ValueError("Changed evaluation input; use a fresh output root")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(input_text)
    benchmark = {
        "profile": bundle["manifest"]["benchmark_profile"],
        "tasks": {
            TASK: {
                "test": {"path": str(input_path), "sha256": sha256_file(input_path)},
            }
        },
    }
    pin_json(output_root / "benchmark_manifest.json", benchmark)
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        manifest = deepcopy(bundle["manifest"])
        manifest.update(
            setting="direct" if arm == "direct_guard" else "direct_indirect",
            selection={
                "policy": "group_balanced"
                if arm.startswith("balanced")
                else "neighbor_fill",
                "budget": ARMS[arm],
            },
            inputs={TASK: {"input": benchmark["tasks"][TASK]["test"]}},
            preparation={
                "version": "bbb_indirect_portable.v1",
                "prediction_filter": PREDICTION_FILTER_VERSION,
                "test_tuned": True,
                "bundle_sha256": receipt["sha256"],
                "prompt_hash": bundle["system_prompt_hash"],
                "code_sha256": sha256_file(Path(__file__)),
            },
        )
        root = output_root / arm
        pin_json(root / "experiment_manifest.json", manifest)
        counts = []
        for index, query in enumerate(bundle["queries"]):
            direct = [bundle["records"][key] for key in query["direct"]]
            candidates = [bundle["records"][key] for key in query["candidates"]]
            selected = select_indirect(
                candidates,
                query["original_fill_ids"],
                arm,
            )
            messages, aliases, payload = render(query, direct, selected)
            prepared = {
                **query["prepared_base"],
                "manifest_hash": content_hash(manifest),
                "messages": messages,
                "request_hash": content_hash(messages),
                "card_alias_map": aliases,
                "prompt_payload": payload,
                "direct_ids": [r["id"] for r in direct],
                "indirect_ids": [r["id"] for r in selected],
                "n_direct": len(direct),
                "n_indirect": len(selected),
                "prompt_version": "bbb_applicability_compact.v1",
            }
            directory = root / "runs" / TASK / f"query_{index:05d}"
            pin_json(directory / "prepared.json", prepared)
            pin_json(
                directory / "selection.json", {"direct": direct, "selected": selected}
            )
            pin_json(
                directory / "evaluation.json",
                {"label": int(benchmark_rows[index]["Y"])},
            )
            counts.append(len(selected))
        pin_json(
            root / "preparation_receipt.json",
            {
                "n_queries": len(counts),
                "indirect_counts": counts,
                "prediction_filter": PREDICTION_FILTER_VERSION,
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ASSETS / "inputs.json.gz")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=tuple(ARMS), default=["direct_guard", "balanced20"]
    )
    args = parser.parse_args(argv)
    prepare(
        args.bundle,
        args.output_root,
        args.arms,
    )


if __name__ == "__main__":
    main()
