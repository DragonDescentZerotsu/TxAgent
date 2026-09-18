import json

from semantic_buckets import bbb_semantic_readout_v1 as bbb


def test_bbb_pair_columns_match_v10_keys() -> None:
    examples = {
        "direct_bbb": '["direct_bbb","endpoint","unit","scale","context","human","condition"]',
        "efflux_transport": '["efflux_transport","endpoint","unit","scale","ABCB1","inhibition","context","human"]',
        "influx_transport": '["influx_transport","endpoint","unit","carrier_mediated_influx","Km"]',
        "passive_permeability": '["passive_permeability","endpoint","unit","scale","pampa_bbb","context","pig"]',
    }
    for source, encoded in examples.items():
        values = json.loads(encoded)
        assert values[0] == source
        assert len(values) - 1 == len(bbb.PAIR_COLUMNS[source])


def test_bbb_scope_is_v10_l2_through_l5() -> None:
    assert "/bbb_martins/v10/" in str(bbb.V10_RECORDS)
    assert bbb.LEVELS == ("L2", "L3", "L4", "L5")
    assert sum(bbb.EXPECTED_LEVEL_COUNTS[level] for level in bbb.LEVELS) == 489_980


def test_bbb_uses_capped_direct_dgx_pools() -> None:
    assert [endpoint["name"] for endpoint in bbb.ENDPOINTS] == [
        "dgx024",
        "dgx027",
    ]
    assert all(endpoint["provider"] == "local" for endpoint in bbb.ENDPOINTS)
    assert all(not endpoint["credential_env"] for endpoint in bbb.ENDPOINTS)
    assert [endpoint.get("max_inflight", 128) for endpoint in bbb.ENDPOINTS] == [
        128,
        128,
    ]
    assert all("litellm" not in endpoint["base_url"] for endpoint in bbb.ENDPOINTS)
    assert bbb.REQUEST_TIMEOUT_S == 3_600
    assert bbb.MAX_ATTEMPTS == 12
