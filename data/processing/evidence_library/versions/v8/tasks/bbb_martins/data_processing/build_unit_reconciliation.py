"""Build the reviewed, endpoint-independent BBB V8 unit map."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_measurement_resolution import (
    DEFAULT_CLEANED_RECORDS,
    DEFAULT_MAPPING_PATH,
)
from data.processing.evidence_library.versions.v8.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
)
from tools.chembl_tool.common.units import canonicalize_unit, clean_unit


OUTPUT = Path(__file__).parent / "canonicalization_v8/bbb_unit_reconciliation.v1.json"
TASK = "bbb_martins"
REVIEW_DATE = "2026-09-02"
MANUAL_REJECTS = frozenset(
    {
        "AUC0-60",
        "K1",
        "PS",
        "null",
        "unavailable",
        "undefined",
        "unknown",
        "unsure",
        "97.20",
        "µl? g^-1? min^-1",
        "± 0.01",
        "± 0.07",
        "± 0.10",
        "± 0.11",
        "± 0.28",
        "± 0.59",
        "± 0.82",
        "± 0.92",
        "± 104.19",
    }
)
MANUAL_SOURCE_MERGES = {
    "% ID": "%ID",
    "% ID-kg/g": "%ID-kg/g",
    "% ID/cc": "%ID/cc",
    "% ID/g brain": "%ID/g brain",
    "% ID/g tissue": "%ID/g tissue",
    "% ID/organ": "%ID/organ",
    "% ID·kg/g": "%ID-kg/g",
    "% Dose/g": "% dose/g",
    "% SUV": "%SUV",
    "% id/g": "%ID/g",
    "%I.D./g": "%ID/g",
    "%I.D./g Tissue": "%ID/g tissue",
    "%Inj/g": "% injected dose/g",
    "%inj/g": "% injected dose/g",
    "%dose/g": "% dose/g",
    "10-6, cm/s": "10^-6 cm/s",
    "DPM/g": "dpm/g",
    "DPM/mg protein": "dpm/mg protein",
    "Micrograms/Cc": "µg/mL",
    "SUV × min": "SUV·min",
    "SUV units": "SUV",
    "SUV×min": "SUV·min",
    "Å2": "Å^2",
    "cm x min-1": "cm/min",
    "cm x s-1": "cm/s",
    "fold increase": "fold",
    "fold-change": "fold",
    "gamma/ml": "µg/mL",
    "log BB": "log10_ratio",
    "log [cm/s]": "log(cm/s)",
    "mcgm./ml.": "µg/mL",
    "meq/kg": "mEq/kg",
    "mg % whole brain": "mg% whole brain",
    "mg%": "mg %",
    "microgm./gm.": "µg/g",
    "micrograms per cubic centimeter": "µg/mL",
    "micromolar": "µM",
    "milliliters per 100 g/min": "mL/100 g/min",
    "millimolars": "mM",
    "ml hg-1 min-1": "mL/100 g/min",
    "ml/min per hg": "mL/100 g/min",
    "mlmin-1g-1": "mL g^-1 min^-1",
    "nMb": "nM",
    "nMc": "nM",
    "nanograms per milliliter": "ng/mL",
    "nanomoles per gram": "nmol/g",
    "ng/g ww": "ng/g wet weight",
    "ng/g, wet wt.": "ng/g wet weight",
    "ng⋅min/g": "ng·min/g",
    "nm-s-1": "nm/s",
    "nmol/gr": "nmol/g",
    "nmols/g": "nmol/g",
    "p.p.m.": "ppm",
    "parts per billion": "ppb",
    "µL min-1 (g brain)-1": "µL/min/g brain",
    "µg-h/g": "µg·h/g",
    "µg-min/mL": "µg·min/mL",
    "µg/g WW": "µg/g wet weight",
    "µg/g Wet Weight": "µg/g wet weight",
    "µg/g w.w.": "µg/g wet weight",
    "µg/gm of dry weight": "µg/g dry weight",
    "µgm/ml": "µg/mL",
    "µmol (100 g)-1 min-1": "µmol/100 g/min",
    "µmol [100 g]-1 min-1": "µmol/100 g/min",
    "µmol hg-1 min-1": "µmol/100 g/min",
}
MANUAL_SOURCE_MERGES_ROUND_2 = {
    "$\\mu$g/g": "µg/g",
    "$\\mu$g/g tissue": "µg/g tissue",
    "$\\mu$g/mL": "µg/mL",
    "% Dose/g of Tissue": "% dose/g tissue",
    "% Dose/g tissue": "% dose/g tissue",
    "% I.D.": "%ID",
    "% I.D./c.c.": "%ID/cc",
    "% I.D./g brain": "%ID/g brain",
    "% I.D./g tissue": "%ID/g tissue",
    "% Inj/g": "% injected dose/g",
    "% Injected dose/organ": "% injected dose/organ",
    "% Injection dose/g tissue": "% injected dose/g tissue",
    "% Injection dose/organ": "% injected dose/organ",
    "% dose/g of brain": "% dose/g brain",
    "% id": "%ID",
    "% inj dose/g": "% injected dose/g",
    "% injected dose (ID)": "% injected dose",
    "% injected dose (ID)/g": "% injected dose/g",
    "% injected dose (id)": "% injected dose",
    "% of I.D.": "%ID",
    "% of i.d.": "%ID",
    "% of the administered dose per g tissue": "% of administered dose/g of tissue",
    "% of the administrated dose": "% of administered dose",
    "% of the injected dose/g tissue": "% injected dose/g tissue",
    "% of the injected dose/milliliter": "% injected dose/mL",
    "% of the injection dose": "% injected dose",
    "%Dose/g organ": "% (dose/g organ)",
    "%I.D./g tissue": "%ID/g tissue",
    "%ID/cc of tissue": "%ID/cc",
    "%id/g": "%ID/g",
    "%id/g*h": "%ID/g·h",
    "10-6(cm/sec)": "10^-6 cm/s",
    "Micrograms": "µg",
    "Micrograms per Gm": "µg/g",
    "SUVs": "SUV",
    "c.p.m./g": "cpm/g",
    "cm·s− 1": "cm/s",
    "counts per minute per milligram": "cpm/mg",
    "counts/(min g)": "cpm/g",
    "counts/min/mg protein": "cpm/mg protein",
    "cpm/mg proteins": "cpm/mg protein",
    "d.p.m./g": "dpm/g",
    "d.p.m./mg wet wt.": "dpm/mg wet weight",
    "dpm/g Frischgewebe": "DPM/g fresh tissue",
    "dpms / mg": "dpm/mg",
    "femtomoles/mg": "fmol/mg",
    "fold-increase": "fold",
    "hµg/g": "µg·h/g",
    "mL/100 g-min": "mL/100 g/min",
    "microM": "µM",
    "micrograms per gram": "µg/g",
    "ml (g*min)-1": "mL g^-1 min^-1",
    "ml/100 g X min": "mL/100 g/min",
    "ml·(100 g)-1·min-1": "mL/100 g/min",
    "nano-moles/g": "nmol/g",
    "nanograms per gram tissue": "ng/g tissue",
    "nanomolar": "nM",
    "ng/g DW": "ng/g dry weight",
    "ng/g d.w.": "ng/g dry weight",
    "ng/g dry tissue": "ng/g dry weight",
    "ng/g of wet tissue": "ng/g wet tissue",
    "ng/g wet weight of brain": "ng/g wet brain weight",
    "nmol g-1 tissue": "nmol/g tissue",
    "nmol/g (fresh weight)": "nmol/g of fresh tissue",
    "nmol/g of Brain Tissue": "nmol/g brain tissue",
    "nmol/g w.w.": "nmol/g wet weight",
    "nmoles/mg proteins": "nmol/mg protein",
    "percent dose per gram of tissue": "% dose/g tissue",
    "percentage": "%",
    "percentage injected dose per gram of tissue": "% injected dose/g tissue",
    "percentage of administered dose per gram of tissue": "% of administered dose/g of tissue",
    "percentage of injected dose per gram": "% injected dose/g",
    "percentage of injected dose per gram tissue": "% injected dose/g tissue",
    "pmol/min·mgp": "pmol/min/mg of protein",
    "rfu/mg": "RFU/mg",
    "sec-1 × 103": "s^-1 × 10^3",
    "standard uptake values": "SUV",
    "u/ml": "U/mL",
    "unit": "units",
    "unit per cc": "units/cc",
    "µ/ml": "µg/mL",
    "µL min-1 (mg protein)-1": "µL/min/mg protein",
    "µg EB/g brain weight": "µg Evans blue/g brain tissue",
    "µg equiv. /g": "µg Eq/g",
    "µg gadolinium per gram of tissue": "µg Gd/g tissue",
    "µg of EB/g of tissue": "µg Evans blue/g tissue",
    "µg of Evans blue per gram of brain": "µg Evans blue/g brain tissue",
    "µg./g. wet wt.": "µg/g wet weight",
    "µg/g (fresh weight)": "µg/g fresh tissue",
    "µg/g dry wt": "µg/g dry weight",
    "µg/g wet mass": "µg/g wet weight",
    "µg/g, dry weight": "µg/g dry weight",
    "µl/min*g_brain": "µL/min/g brain",
    "µl/min·mgp": "µL/min/mg protein",
    "µmol/gm Wet Weight": "µmol/g wet weight",
    "µmole. min-1. g-1 brain": "µmol/min/g brain",
    "µmole/100 g-min": "µmol/100 g/min",
    "γ/ccm": "µg/mL",
}
MANUAL_SOURCE_REJECTS_ROUND_3 = frozenset({"0.72–0.15% ID/g"})
MANUAL_SOURCE_MERGES_ROUND_3 = {
    "% of the total plasma concentrations": "% of total plasma concentration",
    "% of the total plasma levels": "% of total plasma levels",
    "% per Gram of Tissue": "%/g of tissue",
    "% that of serum concentration": "% of serum concentration",
    "%/g of brain": "%/g brain",
    "%DOSE/CC/MIN": "%dose/cc/min",
    "%Dose/cc": "%dose/cc",
    "%ID/100 mL tissue": "% ID/100 mL tissue",
    "%dose": "% dose",
    "%dose/g ratio": "% dose/g ratio",
    "%dose/g tissue": "% dose/g tissue",
    "A.U": "AU",
    "Counts/min/g fresh brain": "counts/min/g brain tissue",
    "I.D.": "ID",
    "Micrograms/gm Brain Tissue": "µg/g brain tissue",
    "Minuten": "min",
    "Numeric (log BB)": "log10_ratio",
    "Numeric (log PS)": "log PS",
    "Stunden": "h",
    "Units per Gm": "units/g",
    "Units/Ml": "units/mL",
    "counts per minute/gm": "cpm/g",
    "counts/g fresh brain/min": "counts/min/g brain tissue",
    "counts/min/g": "cpm/g",
    "cpm/g tissue": "CPM/g. of tissue",
    "dis/min/g wet wt tissue": "dpm/g wet tissue",
    "disint./min/g brain": "dpm/g brain",
    "dose/g brain": "dose/g of brain",
    "dpm/g wet weight": "dpm/g wet tissue",
    "dpm/g wet wt": "dpm/g wet tissue",
    "dpm/mg tissue wet weight": "dpm/mg wet weight",
    "dpm/mg wet wt tissue": "dpm/mg wet weight",
    "inject/g tissue": "injected dose/g tissue",
    "injected activity": "% of injected activity",
    "injected dose per gram of tissue": "injected dose/g tissue",
    "mL/(g x min)": "mL g^-1 min^-1",
    "mg. per 100 c.cm.": "mg/100 cc",
    "mg. per gram of brain tissue": "mg/g brain tissue",
    "mg.%": "mg %",
    "mg/kg dw": "mg/kg dry weight",
    "mg/kg of lipids": "mg/kg lipids",
    "mg/kg w.w.": "mg/kg wet weight",
    "mg‧h/L": "mg·h/L",
    "microg. per g.": "µg/g",
    "microgm. per milliliter": "µg/mL",
    "microgram/gram": "µg/g",
    "microgram/gram brain tissue": "µg/g brain tissue",
    "microgram/hemisphere": "µg/hemisphere",
    "micrograms per Cc.": "µg/mL",
    "micrograms per gram brain tissue": "µg/g brain tissue",
    "micrograms per kilogram wet weight": "µg/kg wet weight",
    "microl/g/s": "µL/s/g",
    "microliters/g of brain": "µL/g brain",
    "min$^{-1}$": "min^-1",
    "min.-1": "min^-1",
    "min− 1": "min^-1",
    "ml/100 g x min": "mL/100 g/min",
    "ml/g/s × 106": "mL/g/s × 10^6",
    "ml/sec/gram brain wt": "mL/s/g brain",
    "ml·hg-1": "mL/100 g",
    "mmol/hg/min": "mmol/100 g/min",
    "mmol/kg of wet weight tissue": "mmol/kg wet weight",
    "mmol/kg wet wt": "mmol/kg wet weight",
    "mol \\cdot s^{-1}": "mol·s^-1",
    "molar": "M",
    "nM-h": "nM h",
    "nM/min/mg of protein": "nM/min/mg protein",
    "nanogram per gram": "ng/g",
    "nanograms": "ng",
    "nanomoles/kg": "nmol/kg",
    "ng per g of wet brain": "ng/g wet weight brain tissue",
    "ng-eq/g": "ng eq/g",
    "ng-equivalents/g": "ng equiv/g",
    "ng-h/mg": "ng*h/mg",
    "ng. h/ml": "ng·h/mL",
    "ng/g of protein": "ng/g protein",
    "ng/g of the tissue": "ng/g tissue",
    "ng/g wet brain": "ng/g wet weight brain tissue",
    "ng/g wet tissue weight": "ng/g wet weight",
    "ng/g, wet weight": "ng/g wet weight",
    "ng/gr": "ng/g",
    "ng/mg d.w.": "ng/mg dry weight",
    "ng/mgprotein": "ng/mg protein",
    "ng/mlL": "ng/mL",
    "nmol min-1 (mg protein)-1": "nmol/mg protein/min",
    "nmol min−1 g−1 tissue": "nmol/min/g tissue",
    "nmol/(mg protein ⋅ min)": "nmol/mg protein/min",
    "nmol/100 mg of protein": "nmol/100 mg protein",
    "nmol/g of brain": "nmol/g brain",
    "nmol/g wet weight of brain tissue": "nmol/g tissue wet weight",
    "nmol/g-fw": "nmol/g of fresh tissue",
    "nmol/g-min": "nmol/g/min",
    "nmol/gr tissue": "nmol/g tissue",
    "nmol/mg prot": "nmol/mg protein",
    "nmol/min-g of brain": "nmol/min/g brain",
    "nmol/min-mg of protein": "nmol/mg protein/min",
    "of Injected dose": "of injected dose",
    "of injected activity": "% of injected activity",
    "pmol mg-1 protein": "pmol/mg protein",
    "pmol mg−1 wet weight": "pmol/mg wet tissue",
    "pmol/mg dry wt tissue": "pmol/mg dry wt",
    "pmol/mg of protein": "pmol/mg protein",
    "pmol/mg-dry brain": "pmol/mg dry wt",
    "pmol/min/g of brain": "pmol/min/g brain",
    "pmole/gram of tissue": "pmol/g tissue",
    "s-1 x 105": "s^-1 × 10^5",
    "s-1 × 102": "s^-1 × 10^2",
    "standard uptake value (SUV)": "SUV",
    "standardized uptake value": "SUV",
    "standardized uptake values (SUVs)": "SUV",
}
MANUAL_SOURCE_REJECTS_ROUND_4 = frozenset(
    {"of ³H-labeled H₂O", "wet weight"}
)
MANUAL_SOURCE_OVERRIDES_ROUND_4 = {
    "DPM/100 mg": "dpm/100 mg",
    "d.p.m. g⁻¹": "dpm·g^-1",
    "µg%": "µg %",
    "µg-%": "µg %",
    "µg/dry g": "µg/g dry weight",
    "µg/g (dw)": "µg/g dry weight",
    "µg/g de tissu frais": "µg/g fresh tissue",
    "µg/g dry tissue weight": "µg/g dry weight",
    "µg/g of dry weight tissue": "µg/g dry weight",
    "µg/g of wet brain": "µg/g of wet brain tissue",
    "µg/g tissue protein": "µg/g protein",
    "µg/mLµmin": "µg/mL·min",
    "µg/mg of protein": "µg/mg protein",
    "µg/mg prot": "µg/mg protein",
    "µg/mg-brain": "µg/mg brain tissue",
    "µgEq/g": "µg Eq/g",
    "µg·g⁻¹dry weight": "µg/g dry weight",
    "µmol/L*min": "µmol·min/L",
    "µmol/g of tissue": "µmol/g tissue",
    "µmol/min/gprotein": "µmol/min/g_protein",
}
MANUAL_SOURCE_OVERRIDES_ROUND_5 = {
    "% Cumulative dose/gm": "% Cumulative dose/gm",
    "% D/g": "% dose/g",
    "% I.D./g": "% ID/g",
    "10 x 10-6 cm/sec": "10^-6 cm/s",
    "10 x 10^-6 cm/sec": "10^-6 cm/s",
    "10 × 10^-6 cm/sec": "10^-6 cm/s",
    "10×10^{-6} cm/sec": "10^-6 cm/s",
    "10−4 min−1": "10−4 min−1",
    "mg/1": "mg/1",
    "ml/g-1/min-1": "mL g^-1 min^-1",
    "ng/g/h": "ng/g/h",
    "ng/h/g": "ng/h/g",
    "nmol/g × min": "nmol/g × min",
    "µg/min/g": "µg/min/g",
    "µg·min-1·g-1": "µg·min-1·g-1",
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _load_source_units() -> tuple[set[str], dict[str, Any]]:
    payload = json.loads(DEFAULT_VOCABULARY_PATH.read_text(encoding="utf-8"))
    units = {
        entry["unit"]
        for entry in payload["units"]
        if any(source.startswith(f"{TASK}:") for source in entry.get("sources", []))
    }
    return units, payload


def _load_active_units() -> tuple[
    set[str],
    set[str],
    Counter[str],
    Counter[str],
    dict[str, list[dict[str, Any]]],
]:
    cleaned = pq.read_table(
        DEFAULT_CLEANED_RECORDS,
        columns=[
            "cleaned_record_id",
            "source_id",
            "endpoint_name",
            "measurement_text",
            "unit_text",
        ],
    ).to_pylist()
    records = {row["cleaned_record_id"]: row for row in cleaned}
    source_exact: set[str] = set()
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in pq.read_table(
        DEFAULT_CLEANED_RECORDS,
        columns=[
            "measurement_resolution_route",
            "measurement_resolution_exact_unit",
            "unit_text",
        ],
    ).to_pylist():
        unit = _text(row["unit_text"])
        if unit:
            counts[unit] += 1
            source_counts[unit] += 1
        if row["measurement_resolution_route"] == "accept":
            exact = _text(row["measurement_resolution_exact_unit"] or row["unit_text"])
            if exact:
                source_exact.add(exact)
                counts[exact] += 1

    extracted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(DEFAULT_MAPPING_PATH).to_pylist():
        if row["status"] != "ok":
            continue
        source = records[row["cleaned_record_id"]]
        for measurement in json.loads(row["measurements_json"] or "[]"):
            unit = _text(measurement.get("unit"))
            if not unit:
                continue
            counts[unit] += 1
            extracted[unit].append(
                {
                    "cleaned_record_id": row["cleaned_record_id"],
                    "source_id": source["source_id"],
                    "endpoint_name": source["endpoint_name"],
                    "measurement_text": source["measurement_text"],
                    "source_unit": source["unit_text"],
                    "extracted_measurement": measurement.get("measurement"),
                    "extracted_unit": unit,
                }
            )
    return source_exact, set(extracted), counts, source_counts, extracted


def _review_sample(
    new_units: set[str], extracted: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    sample = [sorted(extracted[unit], key=lambda row: row["cleaned_record_id"])[0] for unit in sorted(new_units)]
    used = {row["cleaned_record_id"] for row in sample}
    extras = [
        row
        for unit in sorted(new_units, key=lambda value: (-len(extracted[value]), value))
        for row in sorted(extracted[unit], key=lambda value: value["cleaned_record_id"])[1:]
        if row["cleaned_record_id"] not in used
    ]
    for row in extras:
        if len(sample) == 1_000:
            break
        sample.append(row)
        used.add(row["cleaned_record_id"])
    if len(sample) != 1_000:
        raise ValueError(f"expected 1,000 reviewed rows, found {len(sample):,}")
    return sample


def _group_key(unit: str) -> tuple[str, ...]:
    cleaned = clean_unit(unit) or unit
    parsed = canonicalize_unit(unit, task=TASK)
    if parsed.unknown_tokens:
        return ("exact", cleaned)
    return (
        "parsed",
        parsed.canonical,
        format(parsed.scale, ".15g"),
        parsed.transform,
    )


def build_mapping() -> dict[str, Any]:
    source_units, vocabulary = _load_source_units()
    source_exact, extracted_units, counts, source_counts, extracted = (
        _load_active_units()
    )
    new_extracted = extracted_units - source_units
    sample = _review_sample(new_extracted, extracted)
    all_units = source_units | source_exact | extracted_units

    rejected_units = (
        MANUAL_REJECTS
        | MANUAL_SOURCE_REJECTS_ROUND_3
        | MANUAL_SOURCE_REJECTS_ROUND_4
    )
    grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for unit in all_units - MANUAL_REJECTS:
        grouped[_group_key(unit)].append(unit)
    canonical: dict[str, str] = {}
    for units in grouped.values():
        displays = Counter()
        for unit in units:
            displays[clean_unit(unit) or unit] += counts[unit]
        winner = min(displays, key=lambda value: (-displays[value], len(value), value))
        canonical.update({unit: winner for unit in units})

    canonical_counts = Counter(canonical.values())
    source_review = sorted(
        (
            unit
            for unit in source_units
            if canonical_counts[canonical[unit]] == 1
        ),
        key=lambda unit: (-source_counts[unit], unit),
    )[:500]
    if len(source_review) != 500:
        raise ValueError(f"expected 500 source units to review, found {len(source_review):,}")
    if extra := set(MANUAL_SOURCE_MERGES) - set(source_review):
        raise ValueError(f"manual source merges outside review sample: {sorted(extra)}")
    if missing := set(MANUAL_SOURCE_MERGES.values()) - set(canonical.values()):
        raise ValueError(f"manual source merge targets not in unit map: {sorted(missing)}")
    canonical.update(MANUAL_SOURCE_MERGES)

    canonical_counts = Counter(canonical.values())
    source_review_round_2 = sorted(
        (
            unit
            for unit in source_units - set(source_review)
            if canonical_counts[canonical[unit]] == 1
        ),
        key=lambda unit: (-source_counts[unit], unit),
    )[:500]
    if len(source_review_round_2) != 500:
        raise ValueError(
            f"expected 500 source units in review round 2, found {len(source_review_round_2):,}"
        )
    if extra := set(MANUAL_SOURCE_MERGES_ROUND_2) - set(source_review_round_2):
        raise ValueError(f"round 2 merges outside review sample: {sorted(extra)}")
    if missing := set(MANUAL_SOURCE_MERGES_ROUND_2.values()) - set(canonical.values()):
        raise ValueError(f"round 2 merge targets not in unit map: {sorted(missing)}")
    canonical.update(MANUAL_SOURCE_MERGES_ROUND_2)

    canonical_counts = Counter(canonical.values())
    source_review_round_3 = sorted(
        (
            unit
            for unit in source_units - set(source_review) - set(source_review_round_2)
            if canonical_counts[canonical[unit]] == 1
        ),
        key=lambda unit: (-source_counts[unit], unit),
    )[:500]
    if len(source_review_round_3) != 500:
        raise ValueError(
            f"expected 500 source units in review round 3, found {len(source_review_round_3):,}"
        )
    reviewed_round_3 = set(MANUAL_SOURCE_MERGES_ROUND_3) | MANUAL_SOURCE_REJECTS_ROUND_3
    if extra := reviewed_round_3 - set(source_review_round_3):
        raise ValueError(f"round 3 decisions outside review sample: {sorted(extra)}")
    if missing := set(MANUAL_SOURCE_MERGES_ROUND_3.values()) - set(canonical.values()):
        raise ValueError(f"round 3 merge targets not in unit map: {sorted(missing)}")
    if invalid := {
        unit: target
        for unit, target in MANUAL_SOURCE_MERGES_ROUND_3.items()
        if canonical[unit] == target or target in MANUAL_SOURCE_MERGES_ROUND_3
    }:
        raise ValueError(f"round 3 merges must collapse into stable keys: {invalid}")
    canonical.update(MANUAL_SOURCE_MERGES_ROUND_3)

    reviewed_source_units = (
        set(source_review) | set(source_review_round_2) | set(source_review_round_3)
    )
    canonical_counts = Counter(canonical.values())
    remaining_source_units = source_units - reviewed_source_units
    remaining_singletons = sorted(
        (
            unit
            for unit in remaining_source_units
            if canonical_counts[canonical[unit]] == 1
        ),
        key=lambda unit: (-source_counts[unit], unit),
    )
    remaining_grouped = sorted(
        remaining_source_units - set(remaining_singletons),
        key=lambda unit: (-source_counts[unit], unit),
    )
    source_review_round_4_5 = (remaining_singletons + remaining_grouped)[:1_000]
    if len(source_review_round_4_5) != 1_000:
        raise ValueError(
            f"expected 1,000 source units in review rounds 4-5, "
            f"found {len(source_review_round_4_5):,}"
        )
    source_review_round_4 = source_review_round_4_5[:500]
    source_review_round_5 = source_review_round_4_5[500:]
    changed_reviewed_targets = MANUAL_SOURCE_REJECTS_ROUND_4 | {
        unit
        for unit, target in (
            MANUAL_SOURCE_OVERRIDES_ROUND_4
            | MANUAL_SOURCE_OVERRIDES_ROUND_5
        ).items()
        if unit != target
    }
    for round_id, cohort, overrides, rejects in (
        (
            4,
            source_review_round_4,
            MANUAL_SOURCE_OVERRIDES_ROUND_4,
            MANUAL_SOURCE_REJECTS_ROUND_4,
        ),
        (5, source_review_round_5, MANUAL_SOURCE_OVERRIDES_ROUND_5, frozenset()),
    ):
        canonical_targets = set(canonical.values())
        if extra := (set(overrides) | rejects) - set(cohort):
            raise ValueError(
                f"round {round_id} decisions outside review sample: {sorted(extra)}"
            )
        if missing := {
            target
            for unit, target in overrides.items()
            if target != unit and target not in canonical_targets
        }:
            raise ValueError(
                f"round {round_id} targets absent from unit map: {sorted(missing)}"
            )
        if unstable := {
            unit: target
            for unit, target in overrides.items()
            if target != unit and target in changed_reviewed_targets
        }:
            raise ValueError(
                f"round {round_id} targets must remain stable keys: {unstable}"
            )
        canonical.update(overrides)

    entries = []
    for unit in sorted(all_units):
        entry = {
            "task": TASK,
            "canonical_endpoints": ["*"],
            "input_unit": unit,
            "action": "exclude" if unit in rejected_units else "map",
            "review_basis": (
                "manual_source_review_500"
                if unit in source_review
                else "manual_source_review_500_round_2"
                if unit in source_review_round_2
                else "manual_source_review_500_round_3"
                if unit in source_review_round_3
                else "manual_source_review_500_round_4"
                if unit in source_review_round_4
                else "manual_source_review_500_round_5"
                if unit in source_review_round_5
                else "source_declared_valid"
                if unit in source_units
                else "manual_review_1000"
                if unit in new_extracted
                else "deterministic_source_exact"
            ),
        }
        if entry["action"] == "map":
            entry.update(
                {"canonical_unit": canonical[unit], "scale": "1", "domain": "any"}
            )
        entries.append(entry)

    sample_json = json.dumps(sample, ensure_ascii=False, sort_keys=True)
    return {
        "version": "starling_exact_measurement_units.v2",
        "bbb_v8_contract": {
            "scope": "bbb_only_endpoint_independent",
            "value_transform": "identity",
            "scale_representation": "distinct_unit_identity",
            "review_date": REVIEW_DATE,
            "source_vocabulary": {
                "path": str(DEFAULT_VOCABULARY_PATH),
                "sha256": file_sha256(DEFAULT_VOCABULARY_PATH),
                "global_version": vocabulary["version"],
                "bbb_unit_count": len(source_units),
                "bbb_units": sorted(source_units),
            },
            "measurement_resolution": {
                "path": str(DEFAULT_MAPPING_PATH),
                "sha256": file_sha256(DEFAULT_MAPPING_PATH),
            },
            "review": {
                "method": "all_llm_only_units_once_plus_92_frequency_weighted_contexts",
                "sample_size": len(sample),
                "llm_only_units_reviewed": len(new_extracted),
                "additional_context_rows": len(sample) - len(new_extracted),
                "sample_sha256": hashlib.sha256(sample_json.encode()).hexdigest(),
                "sample_record_ids": [row["cleaned_record_id"] for row in sample],
                "manual_rejected_units": sorted(MANUAL_REJECTS),
            },
            "source_unit_review": {
                "method": "top_500_active_source_units_remaining_as_singleton_keys",
                "sample_size": len(source_review),
                "cleaned_record_occurrences": sum(source_counts[unit] for unit in source_review),
                "sample_sha256": hashlib.sha256(
                    json.dumps(source_review, ensure_ascii=False).encode()
                ).hexdigest(),
                "sample_units": source_review,
                "merged_unit_count": len(MANUAL_SOURCE_MERGES),
                "kept_separate_unit_count": len(source_review)
                - len(MANUAL_SOURCE_MERGES),
                "rejected_unit_count": 0,
            },
            "source_unit_review_round_2": {
                "method": "next_500_active_source_units_remaining_as_singleton_keys_after_round_1",
                "sample_size": len(source_review_round_2),
                "cleaned_record_occurrences": sum(
                    source_counts[unit] for unit in source_review_round_2
                ),
                "sample_sha256": hashlib.sha256(
                    json.dumps(source_review_round_2, ensure_ascii=False).encode()
                ).hexdigest(),
                "sample_units": source_review_round_2,
                "merged_unit_count": len(MANUAL_SOURCE_MERGES_ROUND_2),
                "kept_separate_unit_count": len(source_review_round_2)
                - len(MANUAL_SOURCE_MERGES_ROUND_2),
                "rejected_unit_count": 0,
            },
            "source_unit_review_round_3": {
                "method": "next_500_active_source_units_remaining_as_singleton_keys_after_round_2",
                "sample_size": len(source_review_round_3),
                "cleaned_record_occurrences": sum(
                    source_counts[unit] for unit in source_review_round_3
                ),
                "sample_sha256": hashlib.sha256(
                    json.dumps(source_review_round_3, ensure_ascii=False).encode()
                ).hexdigest(),
                "sample_units": source_review_round_3,
                "merged_unit_count": len(MANUAL_SOURCE_MERGES_ROUND_3),
                "kept_separate_unit_count": len(source_review_round_3)
                - len(MANUAL_SOURCE_MERGES_ROUND_3)
                - len(MANUAL_SOURCE_REJECTS_ROUND_3),
                "rejected_unit_count": len(MANUAL_SOURCE_REJECTS_ROUND_3),
                "rejected_units": sorted(MANUAL_SOURCE_REJECTS_ROUND_3),
            },
            "source_unit_review_round_4": {
                "method": (
                    "all_remaining_singleton_source_units_then_"
                    "frequency_ranked_auto_grouped_units"
                ),
                "sample_size": len(source_review_round_4),
                "cleaned_record_occurrences": sum(
                    source_counts[unit] for unit in source_review_round_4
                ),
                "sample_sha256": hashlib.sha256(
                    json.dumps(source_review_round_4, ensure_ascii=False).encode()
                ).hexdigest(),
                "sample_units": source_review_round_4,
                "decision_counts": {
                    "keep_current": len(source_review_round_4)
                    - len(remaining_singletons),
                    "keep_separate": len(remaining_singletons)
                    - len(MANUAL_SOURCE_OVERRIDES_ROUND_4)
                    - len(MANUAL_SOURCE_REJECTS_ROUND_4),
                    "remap": len(MANUAL_SOURCE_OVERRIDES_ROUND_4),
                    "reject": len(MANUAL_SOURCE_REJECTS_ROUND_4),
                },
                "rejected_units": sorted(MANUAL_SOURCE_REJECTS_ROUND_4),
            },
            "source_unit_review_round_5": {
                "method": "frequency_ranked_auto_grouped_source_unit_audit",
                "sample_size": len(source_review_round_5),
                "cleaned_record_occurrences": sum(
                    source_counts[unit] for unit in source_review_round_5
                ),
                "sample_sha256": hashlib.sha256(
                    json.dumps(source_review_round_5, ensure_ascii=False).encode()
                ).hexdigest(),
                "sample_units": source_review_round_5,
                "decision_counts": {
                    "keep_current": len(source_review_round_5)
                    - len(MANUAL_SOURCE_OVERRIDES_ROUND_5),
                    "remap": sum(
                        unit != target
                        for unit, target in MANUAL_SOURCE_OVERRIDES_ROUND_5.items()
                    ),
                    "split_to_self": sum(
                        unit == target
                        for unit, target in MANUAL_SOURCE_OVERRIDES_ROUND_5.items()
                    ),
                    "reject": 0,
                },
            },
            "counts": {
                "source_declared_units": len(source_units),
                "source_exact_units": len(source_exact),
                "llm_extracted_units": len(extracted_units),
                "combined_units": len(all_units),
            },
        },
        "entries": entries,
    }


def main() -> int:
    payload = build_mapping()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(payload['entries']):,} BBB unit decisions to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
