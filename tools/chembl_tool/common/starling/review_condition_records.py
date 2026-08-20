"""Semantically pre-review every proposed context-conditioned gold record.

This runner creates a model-assisted semantic-review ledger.  A downstream
policy may freeze conservative multi-pass consensus into ``review_verdicts``;
that remains model-derived and must never be described as human annotation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic
from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient


SYSTEM_PROMPT = """You are auditing one source record proposed for a molecular benchmark.
Judge the literal record, not whether the proposed condition is scientifically plausible.
Accept only if all are true:
1. the proposed condition is an actual exposure arm, formulation, co-treatment, host state,
   disease/population context, or administration context applying to this molecule;
2. it is not merely background discussion, an outcome, mechanism, intrinsic molecular
   property, species tag, dose-only description, comparator arm, or another molecule;
3. all condition atoms in the proposed exact signature jointly apply to the same record arm;
4. the endpoint label is directly supported under that condition and is attributable to the
   query molecule (a combination may be retained only when the full combination is explicit);
5. the record meets the task-specific endpoint contract below.

Do not merge or simplify composite signatures. Reject contradictory or multiple alternative
arms represented as one signature. Exactness applies to effect-modifying conditions in the
benchmark ontology; phase, dose, species, dates, and administrative metadata are deliberately
not group atoms and need not appear in the signature. Return one JSON object only with keys:
review_status (accepted or rejected), review_reason (short snake_case), review_notes (one
concise sentence), condition_group, condition_atoms, Y, label_method.
For rejection, preserve condition_group/condition_atoms/Y/label_method from the proposal.
"""


TASK_CONTRACTS = {
    "Bioavailability_Ma": """Endpoint: direct absolute human oral bioavailability F,
thresholded at 20%. Reject every relative bioavailability, exposure ratio, food effect,
fold change, percent increase/decrease, comparator-only statement, or other effect estimate,
even when the conditioned arm also has an absolute percentage. For example, reject records
framed as 'rifampin decreased F from 14% to 7%', 'food increased F to 40%', or 'F was 25%
higher in cirrhosis': the reported conditioned value does not convert a relative-effect claim
into an eligible absolute-F record. Also reject a mixed claim that reports both absolute and
relative bioavailability. By contrast, an intravenous reference used only to calculate a
directly reported absolute F is not a relative-effect comparison and may be accepted. Reject
non-human or unresolved populations, mechanisms, intrinsic molecular properties, and records
without a direct absolute-F label. External condition atoms must describe the actual arm under
which that absolute F was measured.""",
    "BBB_Martins": """Endpoint: meaningful CNS/brain access after systemic administration.
Direct brain, unbound brain, brain-to-systemic ratio, CSF proxy, PET/autoradiography, or an
explicit in-vivo BBB outcome may qualify. Reject in-vitro-only permeability, computational
prediction, indirect efficacy, non-systemic CNS administration, and measurements of a
different analyte/metabolite. Disease, barrier manipulation, transporter inhibition,
formulation, age, anesthesia, and systemic route may be retained only when they truly apply
to the measured access experiment.""",
    "Skin_Reaction": """Endpoint: direct human skin sensitization/contact-allergy outcome.
Reject irritation/corrosion, phototoxicity/photoallergy unless the target endpoint explicitly
is ordinary sensitization, prediction-only or AOP intermediate evidence, non-human studies,
and vehicle/occlusion/disease words that occur only in background or an alternative arm.
When two vehicles are alternative arms rather than one mixture, reject the combined signature.""",
    "ClinTox": """Endpoint: a frozen AACT toxicity-failure trial linked to the positive-source
query molecule. Reject efficacy-only failure, administrative/strategic discontinuation,
generic safety monitoring, a reason that explicitly says safety was not involved, and toxicity
attributed only to an arm not containing the query. Disease, route, formulation, population,
or explicit co-treatment may qualify. A combination arm may be accepted without isolating
individual-drug causality when the query is in that arm and every drug co-treatment is present
in the exact signature. Do not infer that merely mentioned prior/comparator drugs are
co-treatments. This extension contains only positive labels.""",
}


REVIEW_PROTOCOL_VERSION = "condition_semantic_prereview.v2"


def review_contract_sha256(task: str) -> str:
    payload = "\n\n".join((REVIEW_PROTOCOL_VERSION, SYSTEM_PROMPT, TASK_CONTRACTS[task]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_LOCAL = threading.local()


def _client(args: argparse.Namespace) -> OpenAICompatibleClient:
    client = getattr(_LOCAL, "client", None)
    if client is None:
        client = OpenAICompatibleClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
            temperature=0.0,
            tool_service_url="http://127.0.0.1:8765",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=False,
        )
        _LOCAL.client = client
    return client


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prompt(task: str, row: Mapping[str, Any]) -> list[dict[str, str]]:
    visible = {
        key: row.get(key)
        for key in (
            "source_record_id", "molecule_name", "drug", "pmid", "condition_text",
            "support_text", "raw_value", "assay_model", "species", "extra_details",
            "bioavailability_report_type", "comparator",
            "proposed_condition_group", "proposed_condition_atoms", "proposal_reason",
            "Y", "label_method",
        )
        if key in row
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + TASK_CONTRACTS[task]},
        {
            "role": "user",
            "content": "Audit this candidate record:\n" + json.dumps(visible, ensure_ascii=False),
        },
    ]


def _normalize(
    row: Mapping[str, Any],
    content: Mapping[str, Any],
    *,
    model: str,
    task: str,
) -> dict[str, Any]:
    status = str(content.get("review_status") or "").strip().lower()
    if status not in {"accepted", "rejected"}:
        raise ValueError(f"invalid review_status={status!r}")
    proposed_atoms = [str(value) for value in row.get("proposed_condition_atoms") or []]
    atoms = [str(value) for value in content.get("condition_atoms") or proposed_atoms]
    group = str(content.get("condition_group") or row.get("proposed_condition_group") or "")
    if status == "accepted" and (not group or not atoms):
        raise ValueError("accepted verdict lacks condition signature")
    if set(atoms) != set(proposed_atoms) or group != row.get("proposed_condition_group"):
        raise ValueError("reviewer attempted to rewrite exact proposed signature")
    if int(content.get("Y", row["Y"])) != int(row["Y"]):
        raise ValueError("reviewer attempted to change deterministic endpoint label")
    return {
        "source_record_id": row["source_record_id"],
        "source_payload_sha256": row["source_payload_sha256"],
        "review_contract_sha256": review_contract_sha256(task),
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "review_status": status,
        "reviewer": f"{model}_semantic_prereview_v2",
        "review_reason": str(content.get("review_reason") or "unspecified")[:120],
        "review_notes": str(content.get("review_notes") or "")[:1000],
        "condition_group": group,
        "condition_atoms": atoms,
        "Y": int(row["Y"]),
        "label_method": str(row.get("label_method") or ""),
    }


def _review_one(task: str, row: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, args.attempts + 1):
        try:
            response = _client(args).chat_json(_prompt(task, row))
            verdict = _normalize(row, response["content"], model=args.model, task=task)
            verdict["attempt_count"] = attempt
            verdict["model_id"] = response.get("model") or args.model
            verdict["usage"] = response.get("usage") or {}
            verdict["reasoning_content"] = response.get("reasoning_content") or ""
            verdict["raw_content"] = response.get("raw_content") or ""
            return verdict
        except Exception as exc:  # the ledger records terminal failures for resumption
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "source_record_id": row["source_record_id"],
        "source_payload_sha256": row["source_payload_sha256"],
        "review_contract_sha256": review_contract_sha256(task),
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "review_status": "error",
        "reviewer": f"{args.model}_semantic_prereview_v2",
        "review_reason": "model_review_failed",
        "review_notes": last_error[:1000],
        "attempt_count": args.attempts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue = _read_jsonl(args.queue)
    prior = {row["source_record_id"]: row for row in _read_jsonl(args.output)} if args.output.exists() else {}
    contract_sha256 = review_contract_sha256(args.task)
    pending = [
        row for row in queue
        if prior.get(row["source_record_id"], {}).get("source_payload_sha256") != row["source_payload_sha256"]
        or prior.get(row["source_record_id"], {}).get("review_contract_sha256") != contract_sha256
        or prior.get(row["source_record_id"], {}).get("review_status") not in {"accepted", "rejected"}
    ]
    completed = dict(prior)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_review_one, args.task, row, args): row for row in pending}
        for number, future in enumerate(as_completed(futures), 1):
            verdict = future.result()
            completed[verdict["source_record_id"]] = verdict
            if number % args.checkpoint_every == 0:
                write_jsonl_atomic(args.output, [completed[key] for key in sorted(completed)])
                print(f"reviewed={number}/{len(pending)} total_ledger={len(completed)}", flush=True)
    ordered = [completed[row["source_record_id"]] for row in queue]
    write_jsonl_atomic(args.output, ordered)
    summary = {
        "task": args.task,
        "queue": str(args.queue),
        "output": str(args.output),
        "model": args.model,
        "served_model_ids": sorted(
            {
                str(row.get("model_id"))
                for row in ordered
                if str(row.get("model_id") or "").strip()
            }
        ),
        "execution": {
            "base_url": args.base_url,
            "reasoning_effort": args.reasoning_effort,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "attempts": args.attempts,
            "workers": args.workers,
            "thinking_parameter_enabled": False,
            "tools_enabled": False,
        },
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "review_contract_sha256": contract_sha256,
        "n_queue": len(queue),
        "n_reviewed_this_run": len(pending),
        "status_counts": {
            status: sum(row.get("review_status") == status for row in ordered)
            for status in ("accepted", "rejected", "error")
        },
        "notice": (
            "Model-assisted semantic review; any consensus promotion remains "
            "model-derived and is not human annotation."
        ),
    }
    write_json_atomic(args.output.with_suffix(".summary.json"), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONTRACTS))
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
