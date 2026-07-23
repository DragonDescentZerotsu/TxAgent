"""Editable metadata field policy for the text group-prompt formats.

This is the SINGLE source of truth for *which* metadata fields appear in the new
text group prompts (`morganfingerprint`, `assay_transfer_tool`) and in what order.
Edit the ``include`` flag on any field below to show/hide it -- you do not need to
touch the Jinja templates or the renderer. Every field a dataset can provide is
listed here (even the excluded ones) so this file documents what is available.

Record types are **format-scoped** so the two formats can differ:

* ``morganfingerprint.neighbor`` -- neighbor header for the Morgan format.
* ``morganfingerprint.record``   -- one retrieved evidence record (Morgan format).
* ``assay_transfer_tool.neighbor`` -- neighbor header for the assay-transfer format.
Each assay-transfer selected record is normalized through `minimal_evidence.v1` and
uses ``morganfingerprint.record``. This intentionally prevents retriever-specific
evidence presentation drift.

The ``context`` field, where present, is a nested dict; the renderer flattens it to
``key: value; key: value`` and drops empty entries. Toggle it as one unit.

To vary the policy per dataset/source, add an entry to ``DATASET_OVERRIDES`` keyed by
the neighbor ``evidence_source`` (e.g. ``"starling-labs/bioavailability_ma/Fg"``);
otherwise ``DEFAULT_POLICY`` applies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    key: str            # key looked up on the source dict
    label: str          # human label shown in the prompt
    include: bool       # flip to False to exclude this field from the prompt


# --- Default policy (applies to every dataset unless overridden below) -----------

DEFAULT_POLICY: dict[str, list[FieldSpec]] = {
    # --- morganfingerprint format ---
    # Neighbor header: this format is about structural analogy, so it shows identity
    # and Morgan similarity.
    "morganfingerprint.neighbor": [
        FieldSpec("canonical_smiles", "SMILES", include=True),
        FieldSpec("similarity", "Morgan Tanimoto similarity", include=True),
        FieldSpec("similarity_bucket", "similarity label", include=True),
        FieldSpec("molecule_chembl_id", "molecule id", include=False),
    ],
    # One retrieved evidence record. Sourced from the minimal-evidence `examples`.
    # Provenance/id fields default to excluded.
    "morganfingerprint.record": [
        FieldSpec("endpoint_type", "endpoint", include=True),
        FieldSpec("reported_value", "value", include=True),
        FieldSpec("reported_units", "unit", include=True),
        FieldSpec("context", "assay context", include=True),
        FieldSpec("support_text", "evidence", include=True),
        FieldSpec("source_confidence", "source confidence", include=False),
        FieldSpec("molecule_name", "reported molecule name", include=False),
        FieldSpec("source_record_id", "source record id", include=False),
        FieldSpec("source_index", "source index", include=False),
        FieldSpec("source_id", "source id", include=False),
    ],
    # --- assay_transfer_tool format ---
    # Neighbor header: the molecule is identified by SMILES, and the only ranking
    # signal shown is the record's transfer likelihood. The same molecule may occur
    # in multiple selected entries. Morgan similarity/label/molecule-id
    # are intentionally omitted (they belong to the Morgan format).
    "assay_transfer_tool.neighbor": [
        FieldSpec("canonical_smiles", "SMILES", include=True),
        FieldSpec("assay_transfer_score", "transfer likelihood (0-1)", include=True),
        FieldSpec("molecule_chembl_id", "molecule id", include=False),
        FieldSpec("similarity", "Morgan Tanimoto similarity", include=False),
        FieldSpec("similarity_bucket", "similarity label", include=False),
    ],
}


# --- Optional per-dataset overrides ----------------------------------------------
# Key by the neighbor `evidence_source` string. Each value is a partial policy:
# only the record types you list override DEFAULT_POLICY; others fall back.
DATASET_OVERRIDES: dict[str, dict[str, list[FieldSpec]]] = {}


def included_fields(record_type: str, dataset: str | None = None) -> list[tuple[str, str]]:
    """Return ordered ``(key, label)`` pairs to render for a record type/dataset."""
    policy = DEFAULT_POLICY
    if dataset and dataset in DATASET_OVERRIDES and record_type in DATASET_OVERRIDES[dataset]:
        specs = DATASET_OVERRIDES[dataset][record_type]
    else:
        specs = policy.get(record_type, [])
    return [(spec.key, spec.label) for spec in specs if spec.include]
