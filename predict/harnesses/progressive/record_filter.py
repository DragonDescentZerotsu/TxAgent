"""Query-condition a frozen 100-record semantic panel before prediction."""

from __future__ import annotations

from copy import copy, deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from predict.llm_io.response import call_with_json_validation, structured_response_is_valid
from predict.harnesses.progressive.prompt import prompt_assets, template_environment
from predict.utils.json import sha256_file, write_json_atomic


FILTER_VERSION = "query_conditioned_record_filter_v5"
FILTER_ROOT = Path(__file__).with_name("prompts") / FILTER_VERSION
MIN_SELECTED = 10
MAX_SELECTED = 25
CANDIDATE_COUNT = 100
VISIBLE_RECORD_FIELDS = (
    "endpoint", "reported_value", "reported_unit", "assay_context", "species",
    "qualifying_conditions", "experimental_details", "support_text",
)


def asset_manifest() -> dict[str, Any]:
    paths = {
        name: FILTER_ROOT / name
        for name in ("system.txt", "user.jinja", "provenance.json")
    }
    return {"version": FILTER_VERSION, "files": {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }}


def _ordered_cards(prepared: Mapping[str, Any]) -> list[tuple[str, str, dict, dict]]:
    rows = []
    for molecule_id, molecule in prepared["active_evidence"].items():
        for card_id, card in molecule.get("cards", {}).items():
            rows.append((molecule_id, card_id, molecule, card))
    rows.sort(key=lambda row: (int(row[3]["_selection_rank"]), row[1]))
    if len(rows) != CANDIDATE_COUNT:
        raise ValueError(f"record filter requires exactly {CANDIDATE_COUNT} candidates")
    return rows


def _scientific_card(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: card[key] for key in VISIBLE_RECORD_FIELDS
        if card.get(key) not in (None, "", [], {})
    }


def _candidate_packet(prepared: Mapping[str, Any], rows) -> tuple[list[dict], dict[str, tuple]]:
    candidates, lookup = [], {}
    for index, row in enumerate(rows, 1):
        molecule_id, card_id, molecule, card = row
        alias = f"R{index:03d}"
        lookup[alias] = row
        candidates.append({
            "record_id": alias,
            "reference_parent_smiles": str(card.get("reference_smiles") or molecule["canonical_smiles"]),
            "morgan_similarity": float(molecule["morgan_similarity"]),
            "scientific_record": _scientific_card(card),
        })
    return candidates, lookup


def _messages(prepared: Mapping[str, Any], candidates: list[dict]) -> list[dict[str, str]]:
    task = prompt_assets("reranked_progressive_l1_context_v4_no_query_prior")["tasks"][
        prepared["task"]
    ]
    molecules, by_parent = [], {}
    for candidate in candidates:
        parent = candidate["reference_parent_smiles"]
        if parent not in by_parent:
            by_parent[parent] = {
                "smiles": parent,
                "morgan_similarity": candidate["morgan_similarity"],
                "records": [],
            }
            molecules.append(by_parent[parent])
        record = {"record_id": candidate["record_id"], **candidate["scientific_record"]}
        by_parent[parent]["records"].append(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )
    user = template_environment().get_template(
        f"prompts/{FILTER_VERSION}/user.jinja"
    ).render(task=task, prepared=prepared, molecules=molecules)
    return [
        {"role": "system", "content": (FILTER_ROOT / "system.txt").read_text()},
        {"role": "user", "content": user},
    ]


def selection_errors(content: Any, known: set[str]) -> list[str]:
    if not isinstance(content, dict) or set(content) != {"selected_record_ids"}:
        return ["response must contain only selected_record_ids"]
    selected = content["selected_record_ids"]
    if not isinstance(selected, list) or not MIN_SELECTED <= len(selected) <= MAX_SELECTED:
        return [f"selected_record_ids must contain {MIN_SELECTED} to {MAX_SELECTED} IDs"]
    if any(not isinstance(item, str) or item not in known for item in selected):
        return ["selected_record_ids contains an unknown or malformed ID"]
    if len(selected) != len(set(selected)):
        return ["selected_record_ids must be unique"]
    return []


def _materialize(prepared: dict, selected: list[str], lookup: Mapping[str, tuple]) -> dict:
    active, first_rank = {}, {}
    for rank, alias in enumerate(selected):
        molecule_id, card_id, molecule, card = lookup[alias]
        if molecule_id not in active:
            active[molecule_id] = deepcopy(molecule)
            active[molecule_id]["cards"] = {}
            first_rank[molecule_id] = rank
        active[molecule_id]["cards"][card_id] = deepcopy(card)
        active[molecule_id]["cards"][card_id]["_selection_rank"] = rank
    for molecule_id, molecule in active.items():
        molecule["_selection_rank"] = first_rank[molecule_id]
    result = deepcopy(prepared)
    result["active_evidence"] = active
    result["new_card_ids"] = [lookup[alias][1] for alias in selected]
    result["n_active_molecules"] = len(active)
    result["n_active_cards"] = len(selected)
    result["should_call_model"] = True
    result["record_filter"] = {"version": FILTER_VERSION, "selected_record_ids": selected,
                               "selected_count": len(selected)}
    result["selection_audit"]["indirect_record_limit_per_level"] = MAX_SELECTED
    result["selection_audit"]["indirect_record_limit_for_current_level"] = len(selected)
    result["retrieval_audit"]["n_visible_records"] = len(selected)
    return result


def _selection_receipt(prepared, selected, lookup):
    records = []
    for rank, alias in enumerate(selected, 1):
        molecule_id, card_id, molecule, card = lookup[alias]
        records.append({"rank": rank, "record_id": alias, "card_id": card_id,
            "canonical_record_id": card["_canonical_record_id"], "molecule_id": molecule_id,
            "reference_parent_smiles": card.get("reference_smiles") or molecule["canonical_smiles"],
            "morgan_similarity": molecule["morgan_similarity"],
            "semantic_bucket_id": card.get("_semantic_bucket_id"),
            "semantic_rank": card.get("_semantic_rank")})
    selected_ids = {row["canonical_record_id"] for row in records}
    control_ids = set(prepared["selection_audit"].get("matched_control_record_ids") or [])
    semantic_ids = set(prepared["selection_audit"].get("original_semantic_record_ids") or [])
    parent_count = len({row["molecule_id"] for row in records})
    bucket_count = len({row["semantic_bucket_id"] for row in records})
    return {"status": "ok", "version": FILTER_VERSION,
            "task": prepared["task"], "query_index": prepared["query_index"],
            "benchmark_row_id": prepared["benchmark_row_id"], "level": prepared["level"],
            "candidate_count": CANDIDATE_COUNT, "selected_count": len(selected),
            "selected_parent_count": parent_count, "selected_semantic_bucket_count": bucket_count,
            "mean_morgan_similarity": sum(row["morgan_similarity"] for row in records) / len(records),
            "repeated_parent_records": len(records) - parent_count,
            "overlap_semantic_top25": len(selected_ids & semantic_ids),
            "overlap_control25": len(selected_ids & control_ids), "records": records}


def _resume(level_dir, prepared, lookup):
    selection_path, final_path = level_dir / "filter" / "selection.json", level_dir / "prepared.json"
    if not selection_path.is_file():
        return False
    receipt = json.loads(selection_path.read_text())
    if receipt.get("version") != FILTER_VERSION:
        raise ValueError(f"saved record-filter version differs: {selection_path}")
    selected = [row["record_id"] for row in receipt.get("records", [])]
    if receipt.get("status") != "ok" or selection_errors({"selected_record_ids": selected}, set(lookup)):
        raise ValueError(f"invalid saved record-filter selection: {selection_path}")
    if not final_path.is_file():
        write_json_atomic(final_path, _materialize(prepared, selected, lookup))
    return True


def query_steps(args, prepared_query, client):
    from predict.harnesses.progressive.inference import query_steps as prediction_steps

    level = int(args.indirect_level)
    level_dir = prepared_query.query_dir / "levels" / f"level_{level}"
    candidate = json.loads((level_dir / "candidate_prepared.json").read_text())
    candidates, lookup = _candidate_packet(candidate, _ordered_cards(candidate))
    filter_dir = level_dir / "filter"
    filter_dir.mkdir(parents=True, exist_ok=True)
    if not _resume(level_dir, candidate, lookup):
        messages = _messages(candidate, candidates)
        request = {"messages": messages, "prompt_characters": sum(len(row["content"]) for row in messages),
                   "assets": asset_manifest(), "candidate_count": len(candidates)}
        write_json_atomic(filter_dir / "request.json", request)
        known = set(lookup)
        response = yield {"messages": messages, "task": candidate["task"], "level": level,
            "stage_kind": "record_filter", "request_namespace": (
                f"{FILTER_VERSION}:{candidate['benchmark_row_id']}:L{level}"),
            "reasoning_transport": None, "reasoning_grammar": None,
            "execute": lambda: call_with_json_validation(
                client.chat_json, messages, required_fields=("selected_record_ids",),
                content_validator=lambda content: selection_errors(content, known),
                branch_name=f"{candidate['task']} L{level} record filter", max_attempts=4)}
        write_json_atomic(filter_dir / "output.json", response)
        if not structured_response_is_valid(response):
            raise ValueError(f"record filter failed validation: {filter_dir}")
        selected = response["content"]["selected_record_ids"]
        write_json_atomic(filter_dir / "selection.json", _selection_receipt(candidate, selected, lookup))
        write_json_atomic(level_dir / "prepared.json", _materialize(candidate, selected, lookup))
    prediction_args = copy(args)
    prediction_args.indirect_record_limit_per_level = MAX_SELECTED
    return (yield from prediction_steps(prediction_args, prepared_query, client))
