from semantic_buckets import publish_completed_weights as completed


def test_completed_sources_make_final_top75_the_latest_llm_epoch() -> None:
    specs = completed._source_specs()
    final = [row for row in specs if row[0] in completed.FINAL_RUNS]

    assert len(final) == 3
    assert {(epoch, priority) for _, epoch, priority, _ in final} == {
        ("v6_top75_complete", 3)
    }
    assert all(path.name == "weights.partial.parquet" for *_, path in final)
