"""Execute prepared progressive prompts and checkpoint each validated state.

This is the third progressive runtime stage.  ``query_steps`` reads prepared
level records, asks ``prompt.build_level_messages`` for the exact request,
executes the model call, validates the structured response, and persists the
state before advancing.  Levels within one molecule remain sequential because
each prompt includes the preceding state; independent molecules may be run in
parallel by a launcher such as ``runner.py`` or ``matrix.py``.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any, TYPE_CHECKING

from predict.harnesses.progressive.grammar import render_reasoning_grammar
from predict.harnesses.progressive.prompt import build_level_messages, prompt_assets
from predict.harnesses.progressive.state import (
    card_alias_maps,
    progressive_state_errors,
    restore_card_ids,
    state_from_content,
)
from predict.api_client.pool import OpenAIProviderPool, ProviderPoolExhausted
from predict.llm_io.response import call_with_json_validation, structured_response_is_valid
from predict.utils.json import canonical_json_bytes, write_json_atomic

if TYPE_CHECKING:
    from predict.harnesses.progressive.runner import PreparedQuery


_PROVENANCE_FIELDS = (
    "supportive_card_ids",
    "contradictory_card_ids",
    "prediction_basis_card_ids",
)

_FULL_FLAT_PREDICTION_ALIASES = {
    "bbb_martins": (
        {"pass", "positive", "bbb+", "bbb_positive", "1"},
        {"fail", "negative", "bbb-", "bbb_negative", "0"},
    ),
    "bioavailability_ma": (
        {"pass", "high", "positive", "bioavailability_positive", "1"},
        {"fail", "low", "negative", "bioavailability_negative", "0"},
    ),
}


def _normalize_full_flat_prediction(task: str, contract: Any, value: Any) -> Any:
    """Map reviewed task aliases to the prompt contract's canonical labels."""
    token = ("" if value is None else str(value)).strip().lower()
    positive, negative = _FULL_FLAT_PREDICTION_ALIASES[task]
    if token in positive:
        return contract.positive_prediction
    if token in negative:
        return contract.negative_prediction
    return value


def _derive_claim_provenance(
    content: Any, *, required_claim_count: int | None = 3
) -> tuple[Any, list[str]]:
    """Make claim citations the single model-authored provenance source."""
    if not isinstance(content, dict):
        return content, []
    normalized = {key: value for key, value in content.items() if key not in _PROVENANCE_FIELDS}
    for field in _PROVENANCE_FIELDS:
        normalized[field] = []
    claims = normalized.get("claims")
    if not isinstance(claims, list):
        message = (
            f"claims must contain exactly {required_claim_count} items"
            if required_claim_count is not None
            else "claims must contain at least one item"
        )
        return normalized, [message]
    if required_claim_count is not None and len(claims) != required_claim_count:
        return normalized, [f"claims must contain exactly {required_claim_count} items"]
    if not claims:
        return normalized, ["claims must contain at least one item"]
    by_role = {"supportive": [], "contradictory": []}
    errors: list[str] = []
    for claim in claims:
        role = claim.get("evidence_role") if isinstance(claim, dict) else None
        refs = claim.get("card_ids") if isinstance(claim, dict) else None
        if role not in by_role:
            errors.append("every claim needs evidence_role supportive or contradictory")
        if not isinstance(refs, list) or len(refs) != 1:
            errors.append("every claim must cite exactly one card")
        if role in by_role and isinstance(refs, list) and len(refs) == 1:
            card_id = str(refs[0])
            if card_id not in by_role[role]:
                by_role[role].append(card_id)
    normalized["supportive_card_ids"] = by_role["supportive"]
    normalized["contradictory_card_ids"] = by_role["contradictory"]
    normalized["prediction_basis_card_ids"] = [
        *by_role["supportive"],
        *by_role["contradictory"],
    ]
    return normalized, list(dict.fromkeys(errors))


def _full_flat_errors(
    content: Any,
    *,
    contract: Any,
    reference_index: list[dict[str, Any]],
    prior_state: dict[str, Any] | None,
    relaxed: bool = False,
) -> list[str]:
    if not isinstance(content, dict):
        return ["response must be a JSON object"]
    expected = {"claims", "summary", contract.prediction_field}
    if prior_state is None:
        expected.update({"confidence", "evidence_gaps"})
    else:
        expected.update({"revision_action", "new_evidence_assessment"})
    if relaxed:
        missing = sorted(expected - set(content))
        if missing:
            return ["response is missing fields: " + ", ".join(missing)]
        errors = []
        for field in ("claims", "evidence_gaps" if prior_state is None else "new_evidence_assessment"):
            if not isinstance(content.get(field), list):
                errors.append(f"{field} must be an array")
        for field in ("summary", "confidence" if prior_state is None else "revision_action"):
            if not isinstance(content.get(field), str) or not content[field].strip():
                errors.append(f"{field} must be a nonempty string")
        if content.get(contract.prediction_field) not in contract.prediction_values:
            errors.append(f"invalid {contract.prediction_field}")
        return errors
    if set(content) != expected:
        return ["response fields must be exactly: " + ", ".join(sorted(expected))]
    errors: list[str] = []
    visible = {str(row["visible_id"]): str(row["unit_kind"]) for row in reference_index}
    new = {str(row["visible_id"]) for row in reference_index if row["is_new"]}

    def citations(row: Any, index: int, *, new_only: bool = False) -> None:
        if not isinstance(row, dict):
            errors.append(f"item {index} must be an object")
            return
        cited: list[str] = []
        for field, kinds in (
            ("molecule_ids", {"molecule", "context_conditioned_molecule"}),
            ("record_ids", {"record"}),
        ):
            values = row.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                errors.append(f"item {index} {field} must be a string array")
                continue
            if len(values) != len(set(values)):
                errors.append(f"item {index} {field} contains duplicates")
            invalid = [value for value in values if visible.get(value) not in kinds]
            if invalid:
                errors.append(f"item {index} {field} contains unknown identifiers")
            if new_only and any(value not in new for value in values):
                errors.append(f"item {index} may cite only indirect evidence")
            cited.extend(values)
        if not cited:
            errors.append(f"item {index} must cite at least one molecule or record")

    claims = content.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a nonempty array")
    else:
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or set(claim) != {
                "claim", "molecule_ids", "record_ids", "evidence_role"
            }:
                errors.append(f"claim {index} has invalid fields")
                continue
            if not str(claim.get("claim") or "").strip():
                errors.append(f"claim {index} needs text")
            if claim.get("evidence_role") not in {"supportive", "contradictory"}:
                errors.append(f"claim {index} has invalid evidence_role")
            citations(claim, index)
    if not isinstance(content.get("summary"), str) or not content["summary"].strip():
        errors.append("summary must be a nonempty string")
    prediction = str(content.get(contract.prediction_field) or "")
    if prediction not in contract.prediction_values:
        errors.append(f"invalid {contract.prediction_field}")
    if prior_state is None:
        if content.get("confidence") not in {"high", "moderate", "low"}:
            errors.append("invalid confidence")
        gaps = content.get("evidence_gaps")
        if (
            not isinstance(gaps, list)
            or len(gaps) > 6
            or any(not isinstance(gap, str) or not gap.strip() for gap in gaps)
        ):
            errors.append("evidence_gaps must contain at most 6 nonempty strings")
    else:
        action = str(content.get("revision_action") or "")
        prior_prediction = str(prior_state.get(contract.prediction_field) or "")
        if action not in {"keep", "strengthen", "weaken", "flip"}:
            errors.append("invalid revision_action")
        elif action == "flip" and prediction == prior_prediction:
            errors.append("revision_action=flip requires a changed prediction")
        elif action != "flip" and prediction != prior_prediction:
            errors.append("changed prediction requires revision_action=flip")
        assessments = content.get("new_evidence_assessment")
        if not isinstance(assessments, list):
            errors.append("new_evidence_assessment must be an array")
        else:
            for index, assessment in enumerate(assessments):
                if not isinstance(assessment, dict) or set(assessment) != {
                    "molecule_ids", "record_ids", "applicability",
                    "direction", "decision_effect",
                }:
                    errors.append(f"assessment {index} has invalid fields")
                    continue
                citations(assessment, index, new_only=True)
                if assessment.get("applicability") not in {
                    "high", "moderate", "low", "not_applicable"
                }:
                    errors.append(f"assessment {index} has invalid applicability")
                if assessment.get("direction") not in {
                    "supportive", "contradictory", "neutral_or_unclear"
                }:
                    errors.append(f"assessment {index} has invalid direction")
                if assessment.get("decision_effect") not in {
                    "changed", "strengthened", "weakened", "no_change"
                }:
                    errors.append(f"assessment {index} has invalid decision_effect")
        if action == "flip" and not any(
            value in new
            for claim in claims or [] if isinstance(claim, dict)
            for field in ("molecule_ids", "record_ids")
            for value in claim.get(field) or []
        ):
            errors.append("a flip must cite at least one indirect molecule or record")
    return list(dict.fromkeys(errors))


def _full_flat_state(
    content: dict[str, Any], *, reference_index: list[dict[str, Any]], level: int
) -> dict[str, Any]:
    stable = {str(row["visible_id"]): str(row["stable_id"]) for row in reference_index}
    derived = {
        "supportive_molecule_ids": [], "supportive_card_ids": [],
        "contradictory_molecule_ids": [], "contradictory_card_ids": [],
    }
    for claim in content["claims"]:
        if not isinstance(claim, dict):
            continue
        role = str(claim.get("evidence_role") or "")
        if role not in {"supportive", "contradictory"}:
            continue
        for field, target in (
            ("molecule_ids", f"{role}_molecule_ids"),
            ("record_ids", f"{role}_card_ids"),
        ):
            values = claim.get(field)
            if not isinstance(values, list):
                continue
            for visible_id in values:
                stable_id = stable.get(str(visible_id))
                if stable_id is None:
                    continue
                if stable_id not in derived[target]:
                    derived[target].append(stable_id)
    derived["prediction_basis_molecule_ids"] = [
        *derived["supportive_molecule_ids"], *derived["contradictory_molecule_ids"]
    ]
    derived["prediction_basis_card_ids"] = [
        *derived["supportive_card_ids"], *derived["contradictory_card_ids"]
    ]
    used = set(derived["prediction_basis_molecule_ids"] + derived["prediction_basis_card_ids"])
    return {
        "level": level,
        **content,
        **derived,
        "not_used_evidence_ids": sorted(set(stable.values()) - used),
    }


def query_steps(
    args: argparse.Namespace,
    prepared_query: "PreparedQuery",
    client: OpenAIProviderPool,
):
    """Yield one validated model stage at a time and persist returned state."""
    # Imported lazily so runner can retain compatibility wrappers without a
    # module cycle.  The runner owns preparation and manifests, not model calls.
    from predict.harnesses.progressive import runner

    task = prepared_query.task
    prompt_version = (
        runner._context_prompt_version(args, task)
        if args.profile == "context_records"
        else args.assay_transfer_prompt_version
    )
    prompt_settings = prompt_assets(prompt_version)["settings"]
    output_contract = prompt_settings.get("output_contract")
    full_flat = output_contract in {
        "full_flat_progressive.v1", "full_flat_progressive.v2"
    }
    relaxed_full_flat = output_contract == "full_flat_progressive.v2"
    claim_provenance = prompt_settings.get("claim_provenance")
    derive_claim_provenance = claim_provenance in {"derived_v1", "derived_v2"}
    reasoning_transport = prompt_settings.get("reasoning_transport")
    if reasoning_transport not in {None, "sglang_tokenized_completion.v1"}:
        raise ValueError(f"unsupported reasoning transport: {reasoning_transport!r}")
    contract = runner._task_contract(task, prompt_version)
    levels = runner._run_levels(args, task)
    prior_state: dict[str, Any] | None = None
    n_calls = 0
    for level_row in levels:
        level = int(level_row["level"])
        level_dir = prepared_query.query_dir / "levels" / f"level_{level}"
        output_path = level_dir / "output.json"
        if output_path.is_file():
            existing = runner._read_json(output_path)
            if existing.get("status") in runner.COMPLETE_LEVEL_STATUSES:
                runner._write_level_trace(args, prepared_query, level, output_path, existing)
                prior_state = dict(existing["state"])
                n_calls += int(existing.get("model_called") is True)
                continue
        prepared = runner._read_json(level_dir / "prepared.json")
        if (
            prompt_assets(prompt_version)['settings'].get('tools', True)
            and not prepared.get("tool_prefetch_complete")
        ):
            raise ValueError(f"formal inference requires visible tool prefetch: {level_dir}")
        if not prepared.get("should_call_model"):
            if prior_state is None:
                if args.query_prior == "none":
                    raise ValueError(f"standalone L1 has no visible evidence: {level_dir}")
                state = runner._none_state(
                    contract=contract,
                    level=level,
                    none_final=prepared["reused_none_final"],
                )
                status = "reused_none"
            else:
                state = {**prior_state, "level": level, "revision_action": "keep"}
                status = "carried_forward"
            output = {
                "status": status,
                "model_called": False,
                "state": state,
                "created_at": runner._now(),
            }
            write_json_atomic(output_path, output)
            runner._write_level_trace(args, prepared_query, level, output_path, output)
            prior_state = state
            continue

        active = prepared["active_evidence"]
        card_id_to_alias, alias_to_card_id = card_alias_maps(active)
        record_limits = runner._indirect_record_limits(args, task)
        prompt_record_limits: int | dict[str, int] = (
            next(iter(record_limits.values()))
            if len(set(record_limits.values())) == 1
            else record_limits
        )
        prompt_levels = levels
        prompt_prepared = prepared
        if full_flat and prior_state is None and len(levels) > 1:
            prompt_levels = levels[:1]
            prompt_prepared = dict(prepared)
            prompt_prepared["retrieval_policy"] = {
                **prepared["retrieval_policy"],
                "stages": {"L1": prepared["retrieval_policy"]["stages"]["L1"]},
            }
        messages, reference_index = build_level_messages(
            contract=contract,
            levels=prompt_levels,
            current_level=level,
            prepared=prompt_prepared,
            active=active,
            prior_state=prior_state,
            profile=args.profile,
            prompt_version=prompt_version,
            record_limit=args.record_limit_per_context_level,
            l2_record_limit=args.l2_record_limit_per_context,
            indirect_record_limit=prompt_record_limits,
            molecule_limit=args.context_limit,
            include_indirect=runner._record_cache_enabled(args),
        )
        reasoning_grammar = render_reasoning_grammar(
            prompt_version,
            active,
            card_id_to_alias=card_id_to_alias,
            task=task,
        )
        reference_contract = prompt_assets(prompt_version)['settings'].get(
            'reasoning_reference_contract'
        )
        write_json_atomic(
            level_dir / "request.json",
            {
                "messages": messages,
                "prompt_characters": sum(len(str(row.get("content") or "")) for row in messages),
                "prompt_utf8_bytes": sum(
                    len(str(row.get("content") or "").encode("utf-8")) for row in messages
                ),
                "card_alias_map": alias_to_card_id,
                "reasoning_reference_contract": reference_contract,
                "reasoning_reference_index": reference_index,
                "reasoning_reference_index_sha256": (
                    hashlib.sha256(canonical_json_bytes(reference_index)).hexdigest()
                    if reference_index is not None else None
                ),
                "reasoning_transport": reasoning_transport,
                "reasoning_grammar": reasoning_grammar,
                "reasoning_grammar_sha256": (
                    hashlib.sha256(reasoning_grammar.encode("utf-8")).hexdigest()
                    if reasoning_grammar is not None
                    else None
                ),
            },
        )
        visible_aliases = set(alias_to_card_id)
        new_aliases = {
            card_id_to_alias[card_id]
            for card_id in map(str, prepared.get("new_card_ids") or [])
        }
        execution_provider_attempts: list[dict[str, Any]] = []
        claim_provenance_errors: list[str] = []

        def routed_chat_json(call_messages: list[dict[str, Any]]) -> dict[str, Any]:
            if reasoning_transport:
                routed_response = client.chat_json(
                    call_messages,
                    tokenized_completion=True,
                    reasoning_grammar=reasoning_grammar,
                )
            else:
                routed_response = client.chat_json(call_messages)
            execution_provider_attempts.extend(
                routed_response.get("execution_provider_attempts") or []
            )
            if relaxed_full_flat and isinstance(routed_response.get("content"), dict):
                content = dict(routed_response["content"])
                content[contract.prediction_field] = _normalize_full_flat_prediction(
                    task, contract, content.get(contract.prediction_field)
                )
                routed_response["content"] = content
            if derive_claim_provenance:
                normalized, errors = _derive_claim_provenance(
                    routed_response.get("content"),
                    required_claim_count=3 if claim_provenance == "derived_v1" else None,
                )
                routed_response["content"] = normalized
                routed_response["claim_provenance_derivation"] = claim_provenance
                claim_provenance_errors[:] = errors
            return routed_response

        def execute_stage():
            if full_flat:
                required_fields = (
                    (
                        "claims", "summary", contract.prediction_field,
                        "confidence", "evidence_gaps",
                    )
                    if prior_state is None
                    else (
                        "claims", "summary", contract.prediction_field,
                        "revision_action", "new_evidence_assessment",
                    )
                )
                return call_with_json_validation(
                    routed_chat_json,
                    messages,
                    required_fields=required_fields,
                    allowed_values={
                        contract.prediction_field: contract.prediction_values,
                        **({} if relaxed_full_flat else (
                            {"revision_action": {"keep", "strengthen", "weaken", "flip"}}
                            if prior_state is not None
                            else {
                                "confidence": {"high", "moderate", "low"},
                            }
                        )),
                    },
                    content_validator=lambda content: _full_flat_errors(
                        content,
                        contract=contract,
                        reference_index=reference_index or [],
                        prior_state=prior_state,
                        relaxed=relaxed_full_flat,
                    ),
                    branch_name=f"{task} full-flat progressive level {level}",
                    max_attempts=4,
                    private_reasoning_prefix=bool(reasoning_transport),
                )
            return call_with_json_validation(
                routed_chat_json,
                messages,
                required_fields=(
                    contract.prediction_field,
                    "confidence",
                    "revision_action",
                    "supportive_card_ids",
                    "contradictory_card_ids",
                    "prediction_basis_card_ids",
                    "claims",
                    "new_evidence_assessment",
                    "evidence_gaps",
                    "decision_summary",
                ),
                allowed_values={
                    contract.prediction_field: contract.prediction_values,
                    "confidence": {"high", "moderate", "low"},
                    "revision_action": (
                        {"initial"}
                        if prior_state is None
                        else {"keep", "strengthen", "weaken", "flip"}
                    ),
                },
                content_validator=lambda content: [
                    *claim_provenance_errors,
                    *progressive_state_errors(
                        content,
                        contract=contract,
                        visible_card_ids=visible_aliases,
                        new_card_ids=new_aliases,
                        prior_state=prior_state,
                        max_claims=None if claim_provenance == "derived_v2" else 8,
                    ),
                ],
                branch_name=f"{task} progressive level {level}",
                max_attempts=4,
                private_reasoning_prefix=bool(reasoning_transport),
            )

        response = yield {
            "messages": messages,
            "level": level,
            "task": task,
            "request_namespace": getattr(args, "request_namespace", ""),
            "reasoning_transport": reasoning_transport,
            "reasoning_grammar": reasoning_grammar,
            "execute": execute_stage,
        }
        response = dict(response)
        if execution_provider_attempts:
            response["execution_provider_attempts"] = list(execution_provider_attempts)
        if not response.get("inference_reused", False):
            n_calls += int(
                (response.get("structured_output_validation") or {}).get("attempt_count") or 1
            )
        if not structured_response_is_valid(response):
            output = {
                "status": "error",
                "model_called": True,
                "llm": response,
                "card_alias_map": alias_to_card_id,
                "created_at": runner._now(),
            }
            write_json_atomic(output_path, output)
            runner._write_level_trace(args, prepared_query, level, output_path, output)
            return {
                "task": task,
                "index": prepared_query.index,
                "status": "error",
                "level": level,
            }
        if full_flat:
            state = _full_flat_state(
                response["content"], reference_index=reference_index or [], level=level
            )
        else:
            restored_content = restore_card_ids(
                response["content"], alias_to_card_id=alias_to_card_id
            )
            state = state_from_content(
                restored_content,
                contract=contract,
                level=level,
                visible_card_ids=set(alias_to_card_id.values()),
            )
        output = {
            "status": "ok",
            "model_called": not response.get("inference_reused", False),
            "inference_reuse": response.get("inference_reuse"),
            "state": state,
            "llm": response,
            "card_alias_map": alias_to_card_id,
            "created_at": runner._now(),
        }
        write_json_atomic(output_path, output)
        runner._write_level_trace(args, prepared_query, level, output_path, output)
        prior_state = state
    return {
        "task": task,
        "index": prepared_query.index,
        "status": "ok",
        "n_model_attempts": n_calls,
    }


def run_query(args: argparse.Namespace, prepared_query: "PreparedQuery", client: OpenAIProviderPool):
    """Drive one dependent query chain synchronously."""
    steps = query_steps(args, prepared_query, client)
    response = None
    while True:
        try:
            job = steps.send(response)
        except StopIteration as done:
            return done.value
        response = job["execute"]()


def run_query_safe(
    args: argparse.Namespace,
    prepared_query: "PreparedQuery",
    client: OpenAIProviderPool,
) -> dict[str, Any]:
    """Persist one query failure without terminating unrelated queries."""
    from predict.harnesses.progressive import runner

    try:
        result = run_query(args, prepared_query, client)
        error_path = prepared_query.query_dir / "run_error.json"
        if result.get("status") == "ok" and error_path.is_file():
            write_json_atomic(error_path, {"status": "resolved", "resolved_at": runner._now()})
        return result
    except Exception as exc:  # noqa: BLE001 - persisted for resumable batch audit
        error = {
            "task": prepared_query.task,
            "index": prepared_query.index,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": runner._now(),
        }
        if isinstance(exc, ProviderPoolExhausted):
            error["execution_provider_attempts"] = exc.attempts
        write_json_atomic(prepared_query.query_dir / "run_error.json", error)
        return error
