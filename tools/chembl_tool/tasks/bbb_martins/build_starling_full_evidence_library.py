"""Build the paper-facing BBB Starling direct/passive/efflux/influx index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.processing.evidence_library.evidence_library import (
    StarlingSourceProfile,
    build_and_write_starling_index,
    build_starling_parquet_evidence_rows,
    starling_molecule_id,
)


DEFAULT_STARLING_DATA_DIR = "data/raw/starling/bbb_martins"
DEFAULT_DIRECT_EVIDENCE_JSONL = (
    "outputs/paper/molecular_evidence_agent/evidence/bbb_starling/all/starling_bbb_evidence.jsonl"
)
DEFAULT_OUT_DIR = "outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full"
EVIDENCE_FILENAME = "starling_bbb_evidence.jsonl"
INDEX_FILENAME = "starling_bbb_neighbor_index.pkl"
META_FILENAME = "starling_bbb_neighbor_index.meta.json"
INDEX_VERSION = "bbb_martins_starling_full_neighbor_index.v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    direct_rows = _load_direct_rows(Path(args.direct_evidence_jsonl))
    mechanism_rows, mechanism_stats = build_starling_parquet_evidence_rows(
        bbb_profiles(Path(args.starling_data_dir), max_rows=args.max_rows_per_source),
        max_record_examples=args.max_record_examples,
        min_confidence=args.min_confidence,
    )
    evidence_rows = [*direct_rows, *mechanism_rows]
    source = {
        "type": "starling_profile_index",
        "dataset": "Starling BBB direct plus task-specific literature acquisitions",
        "families": ["direct_bbb", "passive_permeability", "efflux_transport", "influx_transport"],
        "exact_query_exclusion": True,
    }
    meta = build_and_write_starling_index(
        evidence_rows,
        out_dir=args.out_dir,
        index_version=INDEX_VERSION,
        source=source,
        source_stats={
            "direct": {
                "path": args.direct_evidence_jsonl,
                "n_evidence_rows": len(direct_rows),
            },
            "mechanisms": mechanism_stats,
        },
        evidence_filename=EVIDENCE_FILENAME,
        index_filename=INDEX_FILENAME,
        meta_filename=META_FILENAME,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


def bbb_profiles(data_dir: Path, *, max_rows: int = 0) -> list[StarlingSourceProfile]:
    return [
        StarlingSourceProfile(
            source_id="bbb_passive_permeability",
            path=str(data_dir / "passive_permeability" / "extractions.parquet"),
            group_id="Mechanism.passive_permeability",
            assay_tier="Tier 2",
            endpoint_group="passive_permeability",
            evidence_source="Starling/BBB/passive_permeability",
            endpoint_field="assay_type",
            smiles_field="SMILES",
            value_field="metric_value",
            unit_field="metric_units",
            context_fields=(
                "biological_system",
                "metric_name",
                "metric_uncertainty",
                "passive_bbb_interpretation",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("biological_system", "qualifying_conditions"),
            target_pref_name="passive BBB permeability",
            evidence_role="mechanistic_factor",
            standard_type_prefix="passive BBB permeability",
            include_endpoint_values=(
                "PAMPA-BBB",
                "PAMPA_other",
                "brain_endothelial_cell",
                "MDCK",
                "other_cell_monolayer",
                "membrane_partitioning",
                "experimental_passive_conclusion",
            ),
            extra_example_fields=("passive_bbb_interpretation", "needs_more_context", "pmid"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="bbb_efflux_transport",
            path=str(data_dir / "efflux_transport" / "extractions.parquet"),
            group_id="Mechanism.efflux_transport",
            assay_tier="Tier 3",
            endpoint_group="efflux_transport",
            evidence_source="Starling/BBB/efflux_transport",
            endpoint_field="evidence_type",
            smiles_field="SMILES",
            value_field="quantitative_value",
            context_fields=(
                "transporter_identifier",
                "interaction_conclusion",
                "assay_system",
                "quantitative_metric",
                "perturbation",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("assay_system", "qualifying_conditions"),
            target_pref_name="BBB efflux transport",
            evidence_role="mechanistic_factor",
            standard_type_prefix="BBB efflux transport",
            include_endpoint_values=(
                "bidirectional_transport",
                "genetic_perturbation",
                "pharmacological_inhibition",
                "in_vivo_brain_exposure",
                "qualitative_transporter_claim",
            ),
            extra_example_fields=(
                "transporter_identifier",
                "interaction_conclusion",
                "quantitative_metric",
                "needs_more_context",
                "pmid",
            ),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="bbb_influx_transport",
            path=str(data_dir / "influx_transport" / "extractions.parquet"),
            group_id="Mechanism.influx_transport",
            assay_tier="Tier 4",
            endpoint_group="influx_transport",
            evidence_source="Starling/BBB/influx_transport",
            endpoint_field="transport_mechanism",
            smiles_field="SMILES",
            context_fields=(
                "mediator_name",
                "mediator_identifier",
                "transport_endpoint",
                "evidence_basis",
                "assay_model",
                "reported_result",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("assay_model", "qualifying_conditions"),
            target_pref_name="BBB carrier-mediated influx transport",
            evidence_role="mechanistic_factor",
            standard_type_prefix="BBB influx transport",
            include_endpoint_values=(
                "carrier-mediated influx",
                "transporter-mediated endothelial uptake",
                "receptor-mediated influx",
                "receptor-mediated transcytosis",
                "receptor-mediated endocytosis",
                "unspecified mediated BBB uptake",
            ),
            extra_example_fields=(
                "mediator_name",
                "mediator_identifier",
                "transport_endpoint",
                "reported_result",
                "needs_more_context",
                "pmid",
            ),
            max_rows=max_rows,
        ),
    ]


def _load_direct_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not smiles:
                continue
            row["molecule_chembl_id"] = starling_molecule_id(smiles)
            rows.append(row)
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument("--direct-evidence-jsonl", default=DEFAULT_DIRECT_EVIDENCE_JSONL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
