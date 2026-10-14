import json

from analysis.prediction import l1_context_v3


def test_consolidates_complete_prior_and_contrast_runs(tmp_path, monkeypatch):
    results = tmp_path / "runs"
    monkeypatch.setattr(l1_context_v3, "WIDTHS", (10,))
    cache_bundle = tmp_path / "cache.yaml"
    cache_bundle.write_text("version: 9\n")
    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    (batch_root / "matrix.json").write_text("{}\n")
    for prior_dir, prior in (
        ("with_query_prior", "cached"), ("no_query_prior", "none")
    ):
        for minimum in (0, 1, 2):
            root = (
                results / "contrastive_morgan10/assay_transfer" / prior_dir
                / f"k10_m{minimum}_20260914"
            )
            root.mkdir(parents=True)
            (root / "run.json").write_text(json.dumps({
                "status": "complete", "metric_status": "complete",
                "query_prior": prior,
                "prompt_version": "reranked_progressive_l1_context_v3",
                "retrieval_cache_bundle": str(cache_bundle),
                "batch_root": str(batch_root),
            }))
            (root / "macro_f1.tsv").write_text("task\tL1\nbbb_martins\t0.7\n")
            (root / "neighborhood_label_mix.summary.tsv").write_text(
                "task\tn_queries\tmean_minority_share\n"
                "bbb_martins\t397\t0.2\n"
            )
            (root / "reasoning_reference_coverage.summary.tsv").write_text(
                "task\tlevel\tmean_molecule_coverage\n"
                "bbb_martins\t1\t0.6\n"
            )
            (root / "diagnostics_manifest.json").write_text("{}\n")
            (root / "experiment_manifest.json").write_text("{}\n")
    output = l1_context_v3.build(results, tmp_path / "analysis")
    rows = l1_context_v3._read_tsv(output / "macro_f1.tsv")
    assert len(rows) == 6
    assert {(row["m"], row["query_prior"]) for row in rows} == {
        (str(minimum), prior) for minimum in (0, 1, 2) for prior in ("cached", "none")
    }
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_pair_bucket_analysis_forwards_study_settings_and_preserves_counts(tmp_path, monkeypatch):
    from analysis.prediction import progressive_pair_buckets as analysis
    source = tmp_path / "source.json"
    source.write_text("{}")
    bucket = json.dumps(["assay", "endpoint", "unit"])
    lookup = {(f"u{i}", f"r{i}"): bucket for i in range(1, 4)}
    monkeypatch.setattr(analysis, "_pair_bucket_lookup", lambda task: (lookup, source, source))
    seen = {}
    def selected(**kwargs):
        seen.update(kwargs)
        return {"q": {"L1": [dict(record_id="r1", payload=dict(source_row_uid="u1"))],
                      "L2": [dict(record_id=f"r{i}", payload=dict(source_row_uid=f"u{i}")) for i in (2, 3)]}}, {}
    monkeypatch.setattr(analysis, "_selected_records", selected)
    summary, per_query, frequencies, provenance = analysis.analyze(
        source, task_levels={"bbb_martins": 2}, modes=("morgan",), pools=("all",),
        subset="test", molecule_limit=3, l1_limit=4, later_limit=5)
    assert (seen["last_level"], seen["subset"], seen["molecule_limit"], seen["l1_limit"], seen["later_limit"]) == (2, "test", 3, 4, 5)
    assert {row["level"]: row["selected_records_total"] for row in summary} == {"L1": 1, "L2": 2}
    assert sum(row["selected_records"] for row in frequencies) == 3
    assert len(per_query) == 2 and set(provenance) == {"bbb_martins"}
