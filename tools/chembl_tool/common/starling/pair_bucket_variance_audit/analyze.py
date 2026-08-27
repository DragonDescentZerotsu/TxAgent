"""Audit pair-bucket specificity using within- and between-molecule variation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASKS = {
    "BBB": (
        "bbb_martins",
        "tools.chembl_tool.tasks.bbb_martins.starling_schema",
    ),
    "Oral bioavailability": (
        "bioavailability_ma",
        "tools.chembl_tool.tasks.bioavailability_ma.starling_schema",
    ),
    "Skin reaction": (
        "skin_reaction",
        "tools.chembl_tool.tasks.skin_reaction.starling_schema",
    ),
}
BASE_COLUMNS = (
    "source_id",
    "cleaned_record_id",
    "canonical_endpoint_name",
    "canonical_unit_text",
    "canonical_measurement_scale_id",
    "measurement_kind",
    "canonical_category_id",
    "finite_scalar_value",
    "canonical_smiles",
    "retrieval_eligible",
)
SHORT_FIELD = {
    "canonical_measurement_scale_id": "scale",
    "canonical_assay_context": "assay",
    "canonical_species_context": "species",
    "canonical_reference_scope": "scope",
    "canonical_reference_basis": "basis",
    "canonical_assay_type": "assay type",
    "canonical_transporter_identifier": "transporter",
    "canonical_evidence_type": "evidence type",
    "canonical_transport_mechanism": "mechanism",
    "canonical_kinetic_symbol": "kinetic",
    "canonical_bioavailability_report_type": "report type",
    "canonical_bioavailability_evidence_scope": "evidence scope",
    "canonical_biological_matrix": "matrix",
    "canonical_oral_dose_key": "exact dose",
    "canonical_measurement_target_id": "target",
    "canonical_assay_or_test": "test",
    "canonical_species_or_population": "population",
    "canonical_aop_event": "AOP event",
    "canonical_assay_method": "method",
    "canonical_evidence_system": "system",
    "canonical_study_design": "design",
}
ORAL_ENDPOINTS = ("cmax", "tmax", "auc0_inf", "auc0_t", "auc")
UNKNOWN = "__unknown__"


def _normalized(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isna() | values.eq(""), UNKNOWN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_stable(path: Path, columns: list[str]) -> pd.DataFrame:
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    frame = pd.read_parquet(path, columns=columns)
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    if before != after:
        raise RuntimeError(f"artifact changed while being read: {path}")
    return frame


def _continuous_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    value = pd.to_numeric(frame["finite_scalar_value"], errors="coerce")
    unit = frame["canonical_unit_text"].astype("string")
    eligible = (
        (frame["retrieval_eligible"] == True).fillna(False)  # noqa: E712
        & frame["canonical_smiles"].notna()
        & (frame["measurement_kind"] == "continuous").fillna(False)
        & frame["canonical_category_id"].isna()
        & frame["canonical_measurement_scale_id"].isna()
        & np.isfinite(value)
        & value.gt(0)
        & frame["canonical_endpoint_name"].notna()
        & unit.notna()
        & (~unit.str.startswith("log10(", na=False)).fillna(False)
        & unit.ne("logit_response").fillna(False)
    )
    output = frame.loc[eligible].copy()
    output["log10_value"] = np.log10(value.loc[eligible].astype(float))
    return output


def _variation_stats(
    records: pd.DataFrame,
    dimensions: list[str],
    baseline_pairs: int | None = None,
) -> dict[str, Any]:
    frame = records[["canonical_smiles", "log10_value", *dimensions]].copy()
    for column in dimensions:
        frame[column] = _normalized(frame[column])

    cell_dimensions = [*dimensions, "canonical_smiles"]
    cells = frame.groupby(cell_dimensions, dropna=False, sort=False)[
        "log10_value"
    ].agg(["size", "sum", "mean"])
    squared = (
        frame.assign(squared=frame["log10_value"] ** 2)
        .groupby(cell_dimensions, dropna=False, sort=False)["squared"]
        .sum()
    )
    within_ss = float(
        (squared - cells["sum"] ** 2 / cells["size"]).clip(lower=0).sum()
    )

    strata = frame.groupby(dimensions, dropna=False, sort=False)[
        "log10_value"
    ].agg(["size", "mean"])
    stratum_means = strata["mean"].reindex(cells.index.droplevel(-1)).to_numpy()
    between_ss = float(
        (
            cells["size"].to_numpy()
            * (cells["mean"].to_numpy() - stratum_means) ** 2
        ).sum()
    )
    repeat_pairs = int((cells["size"] * (cells["size"] - 1) // 2).sum())
    contexts = cells.reset_index().groupby(
        dimensions, dropna=False, sort=False
    ).agg(molecule_units=("canonical_smiles", "size"), records=("size", "sum"))
    surviving = contexts[contexts["molecule_units"] >= 20]
    within_dex = math.sqrt(within_ss / len(frame))
    between_dex = math.sqrt(between_ss / len(frame))
    return {
        "within_dex": within_dex,
        "between_dex": between_dex,
        "within_fold": 10**within_dex,
        "between_fold": 10**between_dex,
        "repeat_pairs": repeat_pairs,
        "pair_retention": (
            repeat_pairs / baseline_pairs if baseline_pairs else 1.0
        ),
        "surviving_groups": int(len(surviving)),
        "surviving_units": int(surviving["molecule_units"].sum()),
        "surviving_record_fraction": float(surviving["records"].sum() / len(frame)),
    }


def _selected_endpoints(records: pd.DataFrame) -> pd.DataFrame:
    endpoints = records.groupby("canonical_endpoint_name").agg(
        records=("log10_value", "size"),
        molecules=("canonical_smiles", "nunique"),
    )
    repeated = (
        records.groupby(["canonical_endpoint_name", "canonical_smiles"])
        .size()
        .gt(1)
        .groupby(level=0)
        .sum()
        .rename("repeated_molecules")
    )
    endpoints = endpoints.join(repeated).fillna({"repeated_molecules": 0})
    return endpoints[
        (endpoints["records"] >= 100)
        & (endpoints["molecules"] >= 30)
        & (endpoints["repeated_molecules"] >= 20)
    ].sort_values(["records", "molecules"], ascending=False).head(2)


def _cross_source_tables(repo_root: Path) -> tuple[list[str], list[Path]]:
    lines: list[str] = []
    inputs: list[Path] = []
    evidence_root = repo_root / "outputs/chembl_tool/tasks"
    for task, (task_dir, schema_module) in TASKS.items():
        contract = importlib.import_module(schema_module).RECORD_CONTRACT
        pair_dimensions = {
            source: list(spec.canonical_dimensions)
            for source, spec in contract.pair_buckets.items()
        }
        path = (
            evidence_root
            / task_dir
            / "evidence_library/starling_normalized_v7/03_records/records.parquet"
        )
        inputs.append(path)
        columns = list(
            dict.fromkeys(
                [*BASE_COLUMNS, *[c for dims in pair_dimensions.values() for c in dims]]
            )
        )
        cohort = _continuous_cohort(_read_stable(path, columns))
        lines.extend(
            [
                f"### {task}",
                "",
                "| Source | Endpoint | Records / molecules | Base intra / inter | Cumulative intra-fold path | Full intra / inter | Final repeat pairs | `>=20` final |",
                "|---|---|---:|---:|---|---:|---:|---:|",
            ]
        )
        for source, declared in pair_dimensions.items():
            source_records = cohort[cohort["source_id"] == source]
            selected = _selected_endpoints(source_records)
            if selected.empty:
                lines.append(
                    f"| `{source}` | No qualifying endpoint | - | - | - | - | - | - |"
                )
                continue
            additions = [
                column
                for column in declared
                if column not in {"canonical_endpoint_name", "canonical_unit_text"}
            ]
            for endpoint, counts in selected.iterrows():
                records = source_records[
                    source_records["canonical_endpoint_name"] == endpoint
                ]
                current = ["canonical_unit_text"]
                baseline = _variation_stats(records, current)
                path_parts = [f"unit {baseline['within_fold']:.2f}x"]
                final = baseline
                for column in additions:
                    current.append(column)
                    final = _variation_stats(
                        records, current, baseline["repeat_pairs"]
                    )
                    path_parts.append(
                        f"{SHORT_FIELD.get(column, column)} {final['within_fold']:.2f}x"
                    )
                lines.append(
                    f"| `{source}` | `{endpoint}` | "
                    f"{int(counts['records']):,} / {int(counts['molecules']):,} | "
                    f"{baseline['within_dex']:.3f} dex ({baseline['within_fold']:.2f}x) / "
                    f"{baseline['between_dex']:.3f} dex ({baseline['between_fold']:.2f}x) | "
                    f"{' -> '.join(path_parts)} | "
                    f"{final['within_dex']:.3f} dex ({final['within_fold']:.2f}x) / "
                    f"{final['between_dex']:.3f} dex ({final['between_fold']:.2f}x) | "
                    f"{final['pair_retention']:.0%} | "
                    f"{final['surviving_groups']:,} groups / "
                    f"{final['surviving_units']:,} units / "
                    f"{final['surviving_record_fraction']:.0%} rows |"
                )
        lines.append("")
    return lines, inputs


def _dose_key(frame: pd.DataFrame, *, decade: bool) -> pd.Series:
    value = pd.to_numeric(frame["canonical_oral_dose_value"], errors="coerce")
    parsed = (
        frame["canonical_oral_dose_mapping_status"].eq("resolved")
        & value.gt(0)
        & frame["canonical_oral_dose_unit"].notna()
        & frame["canonical_oral_dose_basis"].notna()
    )
    output = pd.Series(UNKNOWN, index=frame.index, dtype="string")
    for index in frame.index[parsed]:
        number = float(value[index])
        amount = (
            f"10^{math.floor(math.log10(number))}"
            if decade
            else np.format_float_positional(number, trim="-")
        )
        output[index] = "|".join(
            (
                str(frame.at[index, "canonical_oral_dose_quantity_kind"]),
                str(frame.at[index, "canonical_oral_dose_basis"]),
                str(frame.at[index, "canonical_oral_dose_unit"]),
                amount,
            )
        )
    return output


def _oral_dose_table(repo_root: Path) -> tuple[list[str], list[Path]]:
    root = repo_root / "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library"
    v7_path = root / "starling_normalized_v7/03_records/records.parquet"
    dimensions = [
        "canonical_unit_text",
        "canonical_species_context",
        "canonical_biological_matrix",
        "canonical_reference_scope",
    ]
    v7_columns = list(
        dict.fromkeys(
            [
                *BASE_COLUMNS,
                *dimensions,
                "canonical_oral_dose_value",
                "canonical_oral_dose_unit",
                "canonical_oral_dose_quantity_kind",
                "canonical_oral_dose_basis",
                "canonical_oral_dose_mapping_status",
            ]
        )
    )
    records = _read_stable(v7_path, v7_columns)
    records = records[records["source_id"] == "oral_exposure"].copy()
    records = _continuous_cohort(records)
    records["dose_exact"] = _dose_key(records, decade=False)
    records["dose_decade"] = _dose_key(records, decade=True)

    lines = [
        "## Oral exact-dose extension",
        "",
        "Normalized exact dose is `quantity kind | basis | canonical unit | exact value`. "
        "Unresolved values remain in one explicit `__unknown__` level. The decade comparison "
        "uses the same normalized components with `floor(log10(value))`.",
        "",
        "| Endpoint | Rows / molecules | Exact dose known | Current pair key within / inter | + normalized exact dose | Exact pair retention | `>=20` after exact | + decade dose |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for endpoint in ORAL_ENDPOINTS:
        subset = records[records["canonical_endpoint_name"] == endpoint]
        current = _variation_stats(subset, dimensions)
        exact = _variation_stats(
            subset, [*dimensions, "dose_exact"], current["repeat_pairs"]
        )
        decade = _variation_stats(
            subset, [*dimensions, "dose_decade"], current["repeat_pairs"]
        )
        known = subset["dose_exact"].ne(UNKNOWN)
        lines.append(
            f"| `{endpoint}` | {len(subset):,} / {subset['canonical_smiles'].nunique():,} | "
            f"{int(known.sum()):,} ({known.mean():.0%}) | "
            f"{current['within_dex']:.3f} dex ({current['within_fold']:.2f}x) / "
            f"{current['between_dex']:.3f} dex ({current['between_fold']:.2f}x) | "
            f"{exact['within_dex']:.3f} dex ({exact['within_fold']:.2f}x) | "
            f"{exact['pair_retention']:.0%} | "
            f"{exact['surviving_groups']:,} groups / "
            f"{exact['surviving_units']:,} units / "
            f"{exact['surviving_record_fraction']:.0%} rows | "
            f"{decade['within_dex']:.3f} dex ({decade['within_fold']:.2f}x), "
            f"pairs {decade['pair_retention']:.0%} |"
        )
    lines.extend(
        [
            "",
            "The diagnostic reads the same canonical V7 exact-dose fields used by the "
            "production pair-bucket key; it does not use raw dose strings as keys.",
            "",
        ]
    )
    return lines, []


def build_report(repo_root: Path) -> str:
    sys.path.insert(0, str(repo_root))
    cross_source, inputs = _cross_source_tables(repo_root)
    dose, dose_inputs = _oral_dose_table(repo_root)
    all_inputs = list(dict.fromkeys([*inputs, *dose_inputs]))
    lines = [
        "# Pair-bucket variance audit",
        "",
        "This is a read-only diagnostic of how cumulative pair-bucket dimensions change "
        "within-molecule and between-molecule variation. It does not modify pair buckets.",
        "",
        "## Method",
        "",
        "- Fixed cohort: retrieval-eligible records with resolved SMILES and positive, "
        "linear continuous measurements.",
        "- Excluded categorical encodings, logits, nonpositive values, and units already "
        "expressed as `log10(...)`.",
        "- Each source contributes its two largest endpoints having at least 100 records, "
        "30 molecules, and 20 repeatedly measured molecules.",
        "- Values are analyzed as `log10(value)`. A standard deviation of 0.301 dex is "
        "displayed as a twofold spread because `10^0.301 = 2`.",
        "- Dimensions are added in the exact order declared by each source's "
        "`PairBucketSpec`. Missing values are an explicit `__unknown__` level.",
        "- Repeat-pair retention is the fraction of same-molecule record pairs that remain "
        "comparable after adding dimensions. The `>=20` gate counts unique collapsed "
        "molecule units per final context.",
        "",
        "## Inputs",
        "",
        "| Artifact | SHA-256 |",
        "|---|---|",
    ]
    for path in all_inputs:
        lines.append(f"| `{path.relative_to(repo_root)}` | `{_sha256(path)}` |")
    lines.extend(["", "## Cross-source results", "", *cross_source, *dose])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    default_root = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.repo_root.resolve())
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
