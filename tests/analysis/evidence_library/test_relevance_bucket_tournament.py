import json
import tempfile
from collections import Counter
from pathlib import Path

from analysis.evidence_library import relevance_bucket_diagnostic as diagnostic
from analysis.evidence_library import relevance_bucket_pilot as ranking_pilot
from analysis.evidence_library import relevance_bucket_rounds as ranking_rounds
from analysis.evidence_library import relevance_bucket_luna as luna
from analysis.evidence_library import relevance_bucket_tournament as tournament
from analysis.evidence_library import skin_reaction_relevance_bucket_tournament as skin
from analysis.evidence_library.relevance_bucket_tournament import (
    COMPARISONS_PER_REQUEST,
    RELEVANCE_BUCKET_COLUMNS,
    anchor_comparisons,
    audit_comparisons,
    fit_davidson,
    main_comparisons,
    relevance_bucket,
    response_format,
    select_diverse_records,
    validate_response,
    visible_card,
)


def _key(source: str, index: int) -> str:
    record = {"source_id": source}
    record.update({column: f"{column}-{index}" for column in RELEVANCE_BUCKET_COLUMNS[source]})
    return relevance_bucket(record)[0]


def test_relevance_identity_and_cards_exclude_reference_and_molecule_identity():
    record = {
        "source_id": "direct_bbb",
        "canonical_endpoint_name": "brain exposure",
        "measurement_kind": "continuous",
        "canonical_assay_context": None,
        "canonical_reference_scope": "standard control",
        "canonical_reference_basis": "vehicle",
        "canonical_smiles": "CCO",
        "molecule_name": "ethanol",
        "canonical_unit_text": "ratio",
    }
    key, identity = relevance_bucket(record)
    card = visible_card(record)
    assert list(identity) == ["source_id", *RELEVANCE_BUCKET_COLUMNS["direct_bbb"]]
    assert json.loads(key)["canonical_assay_context"] == "__unknown__"
    assert not any("reference" in field for field in card)
    assert "canonical_smiles" not in card and "molecule_name" not in card


def test_diverse_sampling_is_deterministic_deduplicated_and_bounded():
    records = []
    for index in range(12):
        records.append(
            {
                "source_id": "efflux_transport",
                "canonical_record_id": str(11 - index),
                "canonical_smiles": f"molecule-{index % 7}",
                "pair_bucket_key": f"pair-{index % 4}",
                "canonical_endpoint_name": "efflux ratio",
                "measurement_kind": "continuous",
                "canonical_unit_text": "ratio",
                "canonical_species_context": f"species-{index % 3}",
                "canonical_assay_context": f"system-{index % 2}",
                "finite_scalar_value": float(index),
                "canonical_transporter_identifier": "ABCB1",
                "canonical_evidence_type": "substrate",
            }
        )
    first = select_diverse_records(records)
    assert first == select_diverse_records(list(reversed(records)))
    assert len(first) == 5
    assert len({json.dumps(row, sort_keys=True) for row in first}) == 5


def test_pilot_schedule_uses_exact_ten_item_batches_and_quarter_reversal():
    counts = {"direct_bbb": 202, "efflux_transport": 201, "influx_transport": 12, "passive_permeability": 97}
    pilot = [
        _key(source, index)
        for source, count in counts.items()
        for index in range(count)
    ]
    candidates, anchor_rows = anchor_comparisons(pilot)
    assert len(candidates) == 32
    assert len(anchor_rows) == 500
    anchors = candidates[:4]
    main_rows = main_comparisons(pilot, anchors)
    audit_rows = audit_comparisons(main_rows, anchor_rows)
    assert len(main_rows) % COMPARISONS_PER_REQUEST == 0
    assert len(audit_rows) % COMPARISONS_PER_REQUEST == 0
    assert (len(audit_rows) + 4) * 4 == 496 + len(main_rows)


def test_response_contract_is_exact():
    ids = [str(index) for index in range(10)]
    payload = {"items": [{"id": item_id, "winner": "A", "reason": "More relevant endpoint."} for item_id in ids]}
    assert validate_response(payload, ids) == payload["items"]
    assert response_format(ids)["json_schema"]["schema"]["properties"]["items"]["minItems"] == 10
    payload["items"][0]["extra"] = True
    try:
        validate_response(payload, ids)
    except ValueError:
        pass
    else:
        raise AssertionError("extra response fields were accepted")


def test_davidson_recovers_simple_order_and_graph():
    outcomes = [("high", "middle", "A")] * 8 + [("middle", "low", "A")] * 8 + [("high", "low", "A")] * 4
    scores = fit_davidson(outcomes)
    assert scores["high"] > scores["middle"] > scores["low"]


def test_v2_uses_exact_ids_matched_batches_and_outcome_blind_cards():
    sample = json.dumps(
        [
            {
                "canonical_endpoint_name": "brain concentration",
                "canonical_unit_text": "ng/g",
                "finite_scalar_value": 12.3,
                "canonical_category_id": "positive",
            }
        ]
    )
    assert diagnostic.semantic_samples(sample) == [
        {
            "canonical_endpoint_name": "brain concentration",
            "canonical_unit_text": "ng/g",
        }
    ]
    with tempfile.TemporaryDirectory() as directory:
        connection = tournament._connect(Path(directory) / "requests.sqlite3")
        for key in ("bucket-a", "bucket-b"):
            connection.execute(
                "INSERT INTO buckets VALUES (?, 'direct_bbb', ?, ?, 1, 1, NULL)",
                (key, json.dumps({"source_id": "direct_bbb"}), json.dumps([{"canonical_unit_text": "ratio"}])),
            )
        for condition, a, b in (
            ("n10_base", "bucket-a", "bucket-b"),
            ("n10_repeat", "bucket-a", "bucket-b"),
            ("n10_reverse", "bucket-b", "bucket-a"),
        ):
            connection.execute(
                "INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (f"{condition}-000", condition, f"{condition}-batch-000", a, b),
            )
        connection.commit()
        baseline = diagnostic.render_request(connection, "n10_base-batch-000")
        repeated = diagnostic.render_request(connection, "n10_repeat-batch-000")
        reversed_prompt = diagnostic.render_request(connection, "n10_reverse-batch-000")
        assert baseline == repeated
        assert baseline != reversed_prompt
        assert '"bucket_id"' in baseline and '"winner": "A"' not in baseline
        comparisons = diagnostic._comparisons(connection, "n10_base-batch-000")
        payload = {
            "items": [
                {
                    "id": "P000",
                    "winner_bucket_id": diagnostic.bucket_id("bucket-a"),
                    "reason": "More direct.",
                }
            ]
        }
        assert diagnostic.validate_response(payload, comparisons)[0]["winner_key"] == "bucket-a"
        connection.close()


def test_v3_graph_is_connected_near_four_regular_and_batchable_by_five():
    keys = [f"bucket-{index}" for index in range(512)]
    edges = ranking_pilot.comparison_edges(keys)
    degree = {key: sum(key in edge for edge in edges) for key in keys}
    assert len(edges) == 1025
    assert min(degree.values()) == 4 and max(degree.values()) == 5
    assert len(edges) % 5 == 0
    assert tournament._connected([(a, b, "tie") for a, b in edges])


def test_v4_rounds_are_source_local_balanced_connected_and_edge_disjoint():
    counts = {
        "direct_bbb": 202,
        "efflux_transport": 201,
        "influx_transport": 12,
        "passive_permeability": 97,
    }
    keys = [
        _key(source, index)
        for source, count in counts.items()
        for index in range(count)
    ]
    rounds = ranking_rounds.comparison_rounds(keys)
    expected = {
        "round1": {
            "direct_bbb": 606,
            "efflux_transport": 603,
            "influx_transport": 66,
            "passive_permeability": 291,
        },
        "round2": {
            "direct_bbb": 404,
            "efflux_transport": 402,
            "influx_transport": 0,
            "passive_permeability": 194,
        },
        "round3": {
            "direct_bbb": 404,
            "efflux_transport": 402,
            "influx_transport": 0,
            "passive_permeability": 194,
        },
        "round4": {
            "direct_bbb": 404,
            "efflux_transport": 402,
            "influx_transport": 0,
            "passive_permeability": 194,
        },
        "round5": {
            "direct_bbb": 404,
            "efflux_transport": 402,
            "influx_transport": 0,
            "passive_permeability": 194,
        },
        "round6": {
            "direct_bbb": 404,
            "efflux_transport": 402,
            "influx_transport": 0,
            "passive_permeability": 194,
        },
    }
    for source, count in counts.items():
        first = rounds["round1"][source]
        phases = [rounds[phase][source] for phase in rounds]
        for phase in rounds:
            assert len(rounds[phase][source]) == expected[phase][source]
        edge_sets = [
            {tuple(sorted(edge)) for edge in phase}
            for phase in phases
        ]
        assert all(
            not edge_sets[left] & edge_sets[right]
            for left in range(len(edge_sets))
            for right in range(left + 1, len(edge_sets))
        )
        cumulative = [edge for phase in phases for edge in phase]
        assert all(
            json.loads(a)["source_id"] == json.loads(b)["source_id"] == source
            for a, b in cumulative
        )
        assert tournament._connected([(a, b, "A") for a, b in first])
        positions = {
            key: abs(
                sum(a == key for a, _ in cumulative)
                - sum(b == key for _, b in cumulative)
            )
            for key in [item for item in keys if json.loads(item)["source_id"] == source]
        }
        assert max(positions.values()) <= 1


def test_luna_graph_has_degree_twenty_and_batches_at_most_five():
    keys = [
        json.dumps({"source_id": source, "bucket": index}, sort_keys=True)
        for source, count in (("large", 24), ("small", 7))
        for index in range(count)
    ]
    edges_by_source = luna.comparison_edges(keys)
    for source, edges in edges_by_source.items():
        source_keys = [key for key in keys if json.loads(key)["source_id"] == source]
        degree = Counter(key for edge in edges for key in edge)
        expected_degree = min(luna.DEGREE, len(source_keys) - 1)
        assert set(degree) == set(source_keys)
        assert set(degree.values()) == {expected_degree}
        assert tournament._connected([(a, b, "A") for a, b in edges])
        assert max(
            abs(sum(a == key for a, _ in edges) - sum(b == key for _, b in edges))
            for key in source_keys
        ) <= 1
    with tempfile.TemporaryDirectory() as directory:
        connection = tournament._connect(Path(directory) / "requests.sqlite3")
        luna._insert_comparisons(connection, edges_by_source)
        batch_sizes = [
            row[0]
            for row in connection.execute(
                "SELECT count(*) FROM comparisons GROUP BY batch_id"
            )
        ]
        assert batch_sizes and max(batch_sizes) <= 5
        connection.close()


def test_v4_strict_contract_rejects_ties_and_bt_recovers_order():
    with tempfile.TemporaryDirectory() as directory:
        connection = tournament._connect(Path(directory) / "requests.sqlite3")
        connection.execute(
            "INSERT INTO comparisons VALUES ('round1-source-0000', 'round1', "
            "'batch', 'high', 'low', NULL, NULL, NULL)"
        )
        comparisons = diagnostic._comparisons(connection, "batch")
        schema = diagnostic.response_format(comparisons, allow_ties=False)
        winners = schema["json_schema"]["schema"]["properties"]["items"]["items"]["properties"]["winner_bucket_id"]["enum"]
        assert "tie" not in winners
        payload = {"items": [{"id": "P0000", "winner_bucket_id": "tie", "reason": "Equal."}]}
        try:
            diagnostic.validate_response(payload, comparisons, allow_ties=False)
        except ValueError:
            pass
        else:
            raise AssertionError("strict binary contract accepted a tie")
        connection.close()
    outcomes = (
        [("high", "middle", "high")] * 8
        + [("middle", "low", "middle")] * 8
        + [("high", "low", "high")] * 4
    )
    scores = ranking_rounds.fit_bradley_terry(outcomes)
    assert scores["high"] > scores["middle"] > scores["low"]


def test_skin_bucket_cards_and_rounds_are_outcome_blind_and_source_local():
    record = {
        "source_id": "direct_skin_reaction",
        "canonical_endpoint_concept": "sensitization",
        "measurement_kind": "non_scalar",
        "canonical_assay_or_test": "llna",
        "canonical_species_or_population": "mouse",
        "canonical_unit_text": "positive_or_negative",
        "outcome_label": "positive",
        "positive_count": 8,
        "total_tested": 10,
        "canonical_smiles": "CCO",
        "pmid": "123",
        "canonical_reference_scope": "vehicle",
    }
    key, identity = skin.relevance_bucket(record)
    assert list(identity) == [
        "source_id",
        *skin.RELEVANCE_BUCKET_COLUMNS["direct_skin_reaction"],
    ]
    card = skin.visible_card(record)
    assert set(card) == {
        "canonical_endpoint_concept",
        "measurement_kind",
        "canonical_unit_text",
        "canonical_assay_or_test",
        "canonical_species_or_population",
    }
    assert not ({"outcome_label", "positive_count", "total_tested", "pmid"} & set(card))
    assert json.loads(key)["source_id"] == "direct_skin_reaction"

    keys = []
    for source, columns in skin.RELEVANCE_BUCKET_COLUMNS.items():
        for index in range(128):
            item = {"source_id": source, **{column: f"{column}-{index}" for column in columns}}
            keys.append(skin.relevance_bucket(item)[0])
    rounds = skin._rounds(keys)
    for source in skin.RELEVANCE_BUCKET_COLUMNS:
        phases = [rounds[phase][source] for phase in skin.ROUND_CYCLES]
        assert len(phases[0]) == 384
        assert {len(phase) for phase in phases[1:]} == {256}
        edge_sets = [
            {tuple(sorted(edge)) for edge in phase}
            for phase in phases
        ]
        assert all(
            not edge_sets[left] & edge_sets[right]
            for left in range(len(phases))
            for right in range(left + 1, len(phases))
        )
        cumulative = [edge for phase in phases for edge in phase]
        assert all(
            json.loads(a)["source_id"] == json.loads(b)["source_id"] == source
            for a, b in cumulative
        )
        degree = Counter(key for edge in cumulative for key in edge)
        assert set(degree.values()) == {22}
