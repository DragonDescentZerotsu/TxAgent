"""Inference configuration for bbb_martins."""

import json

from .prediction_filter import prediction_only

EVIDENCE_POLICY = "bbb_prediction_only_excluded_l2plus.v2"


def prediction_only_uids(records):
    """Apply the reviewed card predicate to hydrated physical source records."""
    from predict.harnesses.branches.flat import CORE_FIELD_SOURCES, _source_semantic_card

    excluded = set()
    for uid, record in records.items():
        payload = record.get("payload", record)
        if isinstance(payload, str):
            payload = json.loads(payload)
        fields = payload.get("source_fields", payload)
        if "source_contract" in payload:
            card, _ = _source_semantic_card(payload, card_id=str(uid), transfer_likelihood=None)
        else:
            card = {"extra_details": fields.get("extra_details"),
                    "experimental_details": {**(fields.get("experimental_details") or {}), **fields}}
            for key in ("assay_context", "support_text", "qualifying_conditions"):
                for name in CORE_FIELD_SOURCES[key]:
                    value = fields.get(name)
                    if value not in (None, "", [], {}):
                        card[key] = value
                        break
        if prediction_only({"card": card}):
            excluded.add(str(uid))
    return excluded
