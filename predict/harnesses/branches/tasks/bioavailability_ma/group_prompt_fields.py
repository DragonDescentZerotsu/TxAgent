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

Presentation style (``--presentation-style``)
---------------------------------------------
There are two orthogonal axes:

* **Retriever** (Morgan fingerprint vs assay-transfer tool) -- presentation is held
  *invariant* across retrievers: both route records through ``morganfingerprint.record``.
* **Data source** (what fields a record actually holds) -- this is what ``style`` varies.

``style="legacy"`` (default) uses ``DEFAULT_POLICY`` -- the narrow, unified minimal view,
identical for every source. ``style="full"`` consults ``FULL_SOURCE_POLICY``, a *per-source*
expanded view keyed by a prefix of the neighbor ``evidence_source`` (its "source family",
e.g. ``"starling-in-distribution"`` or ``"starling-labs/bioavailability_ma"``). Because both
retrievers share ``morganfingerprint.record``, a per-source ``full`` override is automatically
the same across retrievers -- it exposes source richness without reintroducing retriever drift.
Sources with no ``full`` entry fall back to ``DEFAULT_POLICY``.
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
_SOURCE_CONTRACT_RECORD_FIELDS = [
    FieldSpec(
        "resolved_measurement_display",
        "resolved measurement (extracted scale)",
        include=True,
    ),
    FieldSpec("source_contract", "source-column contract", include=True),
    FieldSpec("source_fields", "source record", include=True),
]

DATASET_OVERRIDES: dict[str, dict[str, list[FieldSpec]]] = {
    # The layered normalized library is source-faithful in every presentation
    # style. Canonical measurement fields remain available to scoring code only.
    "Starling normalized oral bioavailability": {
        "morganfingerprint.record": _SOURCE_CONTRACT_RECORD_FIELDS,
    },
}


# --- Per-source "full" presentation (style="full") -------------------------------
# Keyed by *source family* (a prefix of the neighbor `evidence_source`). Each value is
# a partial policy: only the record types you list override DEFAULT_POLICY for that
# source under the `full` style; everything else falls back to the legacy view.
#
# `full` shows every scientific field a source's records hold. Internal ids/provenance
# (pmid, source_id, record_id, hashes, row numbers) are deliberately omitted -- they are
# noise to the model. The renderer drops empty fields, so records missing a field render
# cleanly.
FULL_SOURCE_POLICY: dict[str, dict[str, list[FieldSpec]]] = {
    # Normalized in-distribution records carry the full scientific scoring payload.
    "starling-in-distribution": {
        "morganfingerprint.record": _SOURCE_CONTRACT_RECORD_FIELDS,
    },
    # The TxAgent evidence library holds report/prose-shaped fields, not scoring fields.
    "starling-labs/bioavailability_ma": {
        # Legacy pre-normalized libraries have no persisted source contract. Keep
        # their historical source-shaped view; the new normalized library below
        # always uses the fail-closed contract projection.
        "morganfingerprint.record": [
            FieldSpec("endpoint_type", "endpoint", include=True),
            FieldSpec("reported_value", "value", include=True),
            FieldSpec("reported_units", "unit", include=True),
            FieldSpec("dose", "dose", include=True),
            FieldSpec("species_or_population", "species/population", include=True),
            FieldSpec("comparator", "comparator", include=True),
            FieldSpec("condition_text", "condition", include=True),
            FieldSpec("oral_exposure_mode", "oral exposure mode", include=True),
            FieldSpec("bioavailability_report_type", "report type", include=True),
            FieldSpec(
                "oral_bioavailability_value_percent",
                "oral bioavailability (%)",
                include=True,
            ),
            FieldSpec("qualifying_conditions", "qualifying conditions", include=True),
            FieldSpec("extra_details", "extra details", include=True),
            FieldSpec("context", "assay context", include=True),
            FieldSpec("support_text", "evidence", include=True),
        ],
    },
    "Starling normalized oral bioavailability": {
        "morganfingerprint.record": _SOURCE_CONTRACT_RECORD_FIELDS,
    },
}


def _full_source_specs(record_type: str, dataset: str | None) -> list[FieldSpec] | None:
    """Longest-prefix match `dataset` against FULL_SOURCE_POLICY source families."""
    if not dataset:
        return None
    best_family: str | None = None
    for family in FULL_SOURCE_POLICY:
        if dataset == family or dataset.startswith(family + "/"):
            if best_family is None or len(family) > len(best_family):
                best_family = family
    if best_family is None:
        return None
    return FULL_SOURCE_POLICY[best_family].get(record_type)


def included_fields(
    record_type: str,
    dataset: str | None = None,
    style: str = "legacy",
) -> list[tuple[str, str]]:
    """Return ordered ``(key, label)`` pairs to render for a record type/dataset.

    ``style="full"`` consults the per-source ``FULL_SOURCE_POLICY`` (matched by a prefix
    of ``dataset``); if no ``full`` spec exists for this source+record_type it falls back
    to the legacy policy. ``style="legacy"`` (default) always uses the legacy policy.
    """
    specs: list[FieldSpec] | None = None
    if dataset and (
        dataset == "Starling normalized oral bioavailability"
        or dataset.startswith("starling-in-distribution/")
    ):
        specs = _SOURCE_CONTRACT_RECORD_FIELDS
    if specs is None and style == "full":
        specs = _full_source_specs(record_type, dataset)
    if specs is None:
        if dataset and dataset in DATASET_OVERRIDES and record_type in DATASET_OVERRIDES[dataset]:
            specs = DATASET_OVERRIDES[dataset][record_type]
        else:
            specs = DEFAULT_POLICY.get(record_type, [])
    return [(spec.key, spec.label) for spec in specs if spec.include]
