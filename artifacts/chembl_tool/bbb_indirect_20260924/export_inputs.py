"""Rebuild the portable, selected-record snapshot from the frozen local baseline."""

import argparse
import gzip
import json
from pathlib import Path

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.record_budget import content_hash
from tools.chembl_tool.tasks.bbb_martins.indirect_ablation import PROMPT


def export(source, iteration, target):
    read = lambda path: json.loads(path.read_text())
    manifest = read(iteration / "balanced20/experiment_manifest.json")
    rows = [
        json.loads(line)
        for line in Path(manifest["inputs"]["bbb_martins"]["input"]["path"])
        .read_text()
        .splitlines()
    ]
    retained = (
        "version",
        "benchmark_profile",
        "model",
        "generation",
        "max_tokens",
        "request_extra_body",
        "task_contracts",
        "direct_identity_policy",
        "indirect_identity_policy",
        "visibility_mode",
        "split_scheme",
        "evaluation_subset",
    )
    bundle = {
        "version": "bbb_indirect_frozen_inputs.v1",
        "manifest": {key: manifest[key] for key in retained},
        "system_prompt_hash": content_hash(PROMPT.read_text()),
        "records": {},
        "queries": [],
        "evaluation_rows": rows,
    }
    source_hashes = {}
    for index in range(len(rows)):
        relative = Path(f"bbb_martins/runs/bbb_martins/query_{index:05d}")
        prepared_path = source / "both_fill" / relative / "prepared.json"
        prepared = read(prepared_path)
        assert content_hash(prepared["messages"]) == prepared["request_hash"]
        source_hashes[str(prepared_path.relative_to(source))] = sha256_file(
            prepared_path
        )
        candidates = {}
        for setting in ("both_fill", "both_cap", "both_group"):
            path = source / setting / relative / "selection.json"
            selection = read(path)
            assert selection["query_hash"] == prepared["query_hash"]
            source_hashes[str(path.relative_to(source))] = sha256_file(path)
            if setting == "both_fill":
                direct = selection["direct"]
                original_fill_ids = [row["id"] for row in selection["selected"]]
            for row in selection["selected"]:
                if row["id"] in candidates:
                    assert candidates[row["id"]] == row
                candidates[row["id"]] = row

        def refs(records):
            keys = []
            for record in records:
                key = content_hash(record)
                bundle["records"][key] = record
                keys.append(key)
            return keys

        base_keys = (
            "version",
            "query_hash",
            "tool_prefetch_complete",
            "context_ready",
            "candidate_pool_hash",
            "query_prior_source",
        )
        payload = prepared["prompt_payload"]
        bundle["queries"].append(
            {
                **{
                    key: payload[key]
                    for key in ("query", "query_prior", "required_json_schema")
                },
                "prepared_base": {
                    key: prepared[key] for key in base_keys if key in prepared
                },
                "direct": refs(direct),
                "candidates": refs(candidates.values()),
                "original_fill_ids": original_fill_ids,
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0
    )
    if target.exists() and target.read_bytes() != data:
        raise ValueError("Frozen export changed; use a new versioned bundle")
    target.write_bytes(data)
    write_json_atomic(
        target.with_suffix(".manifest.json"),
        {
            "version": bundle["version"],
            "sha256": sha256_file(target),
            "n_queries": len(rows),
            "n_unique_record_payloads": len(bundle["records"]),
            "test_tuned": True,
            "pool": "ID-deduplicated union of saved both_fill/both_cap/both_group selections; not full retrieval",
            "source_manifest_sha256": sha256_file(
                iteration / "balanced20/experiment_manifest.json"
            ),
            "source_files": source_hashes,
        },
    )
    print(f"Exported {len(rows)} queries, {len(data)} compressed bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument(
        "--target", type=Path, default=Path(__file__).with_name("inputs.json.gz")
    )
    args = parser.parse_args()
    export(args.source, args.iteration, args.target)
