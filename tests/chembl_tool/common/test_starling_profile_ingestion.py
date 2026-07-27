import pandas as pd

from tools.chembl_tool.common.starling import evidence_library as starling_evidence_library
from tools.chembl_tool.common.starling import StarlingSourceProfile, build_starling_parquet_evidence_rows


def test_profile_maps_different_parquet_columns_and_aggregates_by_molecule(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "structure": "CCO",
                "compound": "a",
                "kind": "AUC",
                "amount": "12",
                "units": "ng*h/mL",
                "study": "human oral",
                "support": "first record",
                "score": 0.9,
            },
            {
                "structure": "CCO",
                "compound": "a",
                "kind": "Cmax",
                "amount": "3",
                "units": "ng/mL",
                "study": "human oral",
                "support": "second record",
                "score": 0.8,
            },
            {
                "structure": "",
                "compound": "missing",
                "kind": "AUC",
                "amount": "1",
                "units": "ng*h/mL",
                "study": "rat",
                "support": "no structure",
                "score": 1.0,
            },
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    profile = StarlingSourceProfile(
        source_id="oral_exposure",
        path="unused.parquet",
        group_id="Observed.oral_exposure",
        assay_tier="Observed",
        endpoint_group="oral_exposure",
        evidence_source="starling/example",
        endpoint_field="kind",
        smiles_field="structure",
        value_field="amount",
        unit_field="units",
        context_fields=("study",),
        scope_fields=("study",),
        name_fields=("compound",),
        support_text_field="support",
        confidence_field="score",
        evidence_role="surrogate_proxy",
    )

    rows, stats = build_starling_parquet_evidence_rows([profile])

    assert len(rows) == 1
    assert rows[0]["source_record_count"] == 2
    assert rows[0]["minimal_evidence"]["group"]["id"] == "Observed.oral_exposure"
    assert rows[0]["minimal_evidence"]["annotations"]["evidence_role"] == "surrogate_proxy"
    assert rows[0]["minimal_evidence"]["annotations"]["scope"] == {"study": ["human oral"]}
    assert {item["endpoint_type"] for item in rows[0]["minimal_evidence"]["examples"]} == {"AUC", "Cmax"}
    assert stats["sources"]["oral_exposure"]["n_missing_smiles"] == 1


def test_profiles_can_split_roles_without_duplicate_source_molecule_identity(monkeypatch):
    frame = pd.DataFrame(
        [
            {"smiles": "OCC", "kind": "F", "value": "40", "unit": "%", "support_text": "direct"},
            {"smiles": "CCO", "kind": "AUC", "value": "20", "unit": "ng*h/mL", "support_text": "proxy"},
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    standardize_calls = []
    original_standardize = starling_evidence_library.standardize_smiles

    def tracked_standardize(smiles):
        standardize_calls.append(smiles)
        return original_standardize(smiles)

    monkeypatch.setattr(starling_evidence_library, "standardize_smiles", tracked_standardize)
    common = {
        "path": "unused.parquet",
        "assay_tier": "Observed",
        "evidence_source": "starling/example",
        "endpoint_field": "kind",
        "value_field": "value",
        "unit_field": "unit",
    }
    profiles = [
        StarlingSourceProfile(
            source_id="direct",
            group_id="Observed.direct",
            endpoint_group="direct",
            evidence_role="direct_outcome",
            include_endpoint_values=("F",),
            **common,
        ),
        StarlingSourceProfile(
            source_id="proxy",
            group_id="Observed.proxy",
            endpoint_group="proxy",
            evidence_role="surrogate_proxy",
            exclude_endpoint_values=("F",),
            **common,
        ),
    ]

    rows, _ = build_starling_parquet_evidence_rows(profiles)

    assert len(rows) == 2
    assert rows[0]["molecule_chembl_id"] == rows[1]["molecule_chembl_id"]
    direct = next(row for row in rows if row["group_id"] == "Observed.direct")
    proxy = next(row for row in rows if row["group_id"] == "Observed.proxy")
    assert direct["standard_value"] == 40.0
    assert direct["minimal_evidence"]["endpoint"]["measurement"]["value"] == 40.0
    assert proxy["standard_value"] == ""
    assert sorted(standardize_calls) == ["CCO", "OCC"]


def test_profile_can_require_context_for_an_ambiguous_endpoint(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "smiles": "CCO",
                "kind": "local_tissue_injury",
                "support_text": "Topical exposure caused epidermal blistering.",
            },
            {
                "smiles": "CCN",
                "kind": "local_tissue_injury",
                "support_text": "Cardiac mitochondrial injury was observed.",
            },
            {
                "smiles": "CCC",
                "kind": "skin_irritation",
                "support_text": "No extra context is required for this unambiguous endpoint.",
            },
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame)
    profile = StarlingSourceProfile(
        source_id="skin_damage",
        path="unused.parquet",
        group_id="Mechanism.skin_damage",
        assay_tier="Tier 3",
        endpoint_group="skin_damage",
        evidence_source="starling/example",
        endpoint_field="kind",
        include_endpoint_values=("local_tissue_injury", "skin_irritation"),
        context_filter_fields=("support_text",),
        required_context_patterns_by_endpoint=(("local_tissue_injury", (r"epiderm", r"dermal")),),
    )

    rows, stats = build_starling_parquet_evidence_rows([profile])

    assert {row["canonical_smiles"] for row in rows} == {"CCO", "CCC"}
    assert stats["sources"]["skin_damage"]["n_filtered_context"] == 1
