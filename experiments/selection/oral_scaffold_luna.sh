#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${STUDY_ID:?fresh study identity}"
export PYTHON_BIN=${PYTHON_BIN:-python}
export TASK=bioavailability_ma
export CACHE_BUNDLE=predict/retrieval/assay_reranking/flat_v5_manifests/scaffold_disjoint_parent100_v1/catalog.json
export OUTPUT_ROOT=outputs/predictions
study=outputs/analysis/context_selection/bioavailability_ma/scaffold/luna/$STUDY_ID
mkdir -p "$study"
extension=$study/semantic_extension/projection/manifest.json
semantic_args=()
resolution=$("$PYTHON_BIN" -c 'import json; from optimization.semantic_buckets.inputs import resolve_semantic_bucket_artifacts; print(json.load(open(resolve_semantic_bucket_artifacts("bioavailability_ma").manifest)).get("semantic_weight_resolution", ""))')
if [[ $resolution != reviewed_membership_mean.v1 ]]; then
  if [[ ! -f $extension ]]; then
    "$PYTHON_BIN" -m optimization.selection.search extend-semantics --task "$TASK" \
      --cache-bundle "$CACHE_BUNDLE" --output-root "$study/semantic_extension"
  fi
  semantic_args=(--semantic-extension "$extension")
fi
historical=outputs/analysis/record_selection/luna_bbb_oral_20260924_v1
provider=predict/api_client/providers/gpt6_luna_openrouter_standard_high_64.json
"$PYTHON_BIN" - "$study" "$CACHE_BUNDLE" "$provider" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from predict.utils.json import sha256_file, write_json_atomic
study, cache, provider = map(Path, sys.argv[1:])
if not (study / "study.json").exists():
    grids = [Path(f"optimization/selection/grids/oral_scaffold_luna_{b}_top3.json") for b in ("gold", "tdc")]
    write_json_atomic(study / "study.json", dict(
        study_id=study.name, created_at=datetime.now(timezone.utc).isoformat(),
        task="bioavailability_ma", model="openai/gpt-6-luna", status="running",
        validation_cohorts={"gold_v1": "valid_small", "tdc": "valid"},
        query_counts={"gold_v1": {"valid_small": 100, "test": 269}, "tdc": {"valid": 64, "test": 128}},
        selection_rule="macro_f1 desc, accuracy desc, profile asc; independent benchmark winners",
        fixed_direct={"gold_v1": "ga100_mc010_label025", "tdc": "ga075_mc010_label025"},
        budgets={"direct_contexts": 10, "indirect_uids_joint_L2_L6": 50},
        candidate_authority="outputs/analysis/record_selection/luna_bbb_oral_20260924_v1/{gold,tdc}_oral_indirect_valid.tsv",
        inputs=[dict(path=str(p), sha256=sha256_file(p)) for p in [cache, provider, *grids]],
        prediction=dict(prompt="final_full_flat", layout="level-grouped", reasoning_effort="high",
                        temperature=None, max_tokens=128000, parallelism=64, cached_priors=True),
        artifacts="Generated selections, leaderboards and reports are gitignored; raw predictions remain under outputs/predictions."))
PY

for phase in valid test; do
 for benchmark in gold_v1 tdc; do
  if [[ $benchmark == gold_v1 ]]; then
    cohort=valid_small
    grid=optimization/selection/grids/oral_scaffold_luna_gold_top3.json
    fixed_valid=$historical/gold_oral_direct_valid/direct/ga100_mc010_label025/$TASK/manifest.json
    fixed_test=$historical/gold_oral_direct_test/direct/ga100_mc010_label025/$TASK/manifest.json
    compose_args=(--cache-bundle "$CACHE_BUNDLE")
    candidate_args=()
    if [[ $phase == test ]]; then candidate_args=(--exclude-disjoint-violations); fi
    export BENCHMARK=gold
  else
    cohort=valid
    grid=optimization/selection/grids/oral_scaffold_luna_tdc_top3.json
    compose_args=()
    candidate_args=(--exclude-disjoint-violations)
    export BENCHMARK=tdc
    for direct_split in valid test; do
      direct_root=$study/tdc/$direct_split/direct
      if [[ ! -f $direct_root/grid_manifest.json ]]; then
        "$PYTHON_BIN" -m optimization.selection.search direct --tasks "$TASK" --benchmark tdc \
          --subset "$direct_split" --profiles ga075_mc010_label025 --budget 10 --workers 8 \
          --cache-bundle "$CACHE_BUNDLE" --output-root "$direct_root"
      fi
    done
    fixed_valid=$study/tdc/valid/direct/direct/ga075_mc010_label025/$TASK/manifest.json
    fixed_test=$study/tdc/test/direct/direct/ga075_mc010_label025/$TASK/manifest.json
  fi
  root=$study/$benchmark
  if [[ ! -f $root/$cohort/indirect/grid_manifest.json ]]; then
    "$PYTHON_BIN" -m optimization.selection.search indirect --tasks "$TASK" --benchmark "$benchmark" \
      --subset "$cohort" --grid "$grid" --budget 50 --workers 8 --cache-bundle "$CACHE_BUNDLE" \
      "${semantic_args[@]}" --output-root "$root/$cohort/indirect" "${candidate_args[@]}"
  fi
  mapfile -t profiles < <("$PYTHON_BIN" -c 'import json,sys; print("\n".join(p["name"] for p in json.load(open(sys.argv[1]))["profiles"]))' "$grid")
    split=$phase
    if [[ $split == valid ]]; then
      selection_cohort=$cohort
      direct=$fixed_valid
      selected_profiles=("${profiles[@]}")
    else
      # Winner receipts are immutable and exist before any test prediction.
      selection_cohort=test
      direct=$fixed_test
      mapfile -t selected_profiles < <("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$root/winner.json")
      if [[ ! -f $root/test/indirect/grid_manifest.json ]]; then
       "$PYTHON_BIN" -m optimization.selection.search indirect --tasks "$TASK" --benchmark "$benchmark" \
        --subset test --grid "$grid" --profiles "${selected_profiles[@]}" --budget 50 --workers 8 \
        --cache-bundle "$CACHE_BUNDLE" "${semantic_args[@]}" --output-root "$root/test/indirect" "${candidate_args[@]}"
      fi
    fi
    if [[ $benchmark == gold_v1 ]]; then
      export QUERIES=data/gold_labels/Bioavailability_Ma/v1/scaffold/${split}_molecule_condition_labels.jsonl
      prior=outputs/predictions/query-prior/full_flat_context_v5_six_tasks_upstream_v1/six_task_${split}_query_priors_upstream_v1_20260922_
      if [[ $split == valid ]]; then prior+=0810; else prior+=0738; fi
      prior+=/gold_v1/$TASK/$split/query-prior
    elif [[ $split == valid ]]; then
      export QUERIES=data/gold_labels/TDC/Bioavailability_Ma/valid_molecule_condition_labels.jsonl
      prior=outputs/predictions/query-prior/full_flat_context_v5_six_tasks_upstream_v1/tdc_valid_query_priors_upstream_bbb_oral_exact_native_20260923_v1/tdc/$TASK/valid/query-prior
    else
      export QUERIES=data/gold_labels/TDC/Bioavailability_Ma/test_molecule_condition_labels.jsonl
      prior=outputs/predictions/query-prior/f20_evidence_calibrated_v2/tdc_v1_test_upstream_deepseek_openrouter_20260923_r1/tdc/$TASK/test/query-prior
    fi
    for profile in "${selected_profiles[@]}"; do
      export SELECTION=$root/$selection_cohort/combined/$profile/manifest.json
      if [[ ! -f $SELECTION ]]; then
        "$PYTHON_BIN" -m optimization.selection.search compose --direct-manifest "$direct" \
          --indirect-manifest "$root/$selection_cohort/indirect/indirect/$profile/$TASK/manifest.json" \
          --output "$SELECTION" "${compose_args[@]}"
      fi
      export SUBSET=$split RUN_NAME=oral_scaffold_luna_${STUDY_ID}_${benchmark}_${selection_cohort}_${profile}
      batch=$OUTPUT_ROOT/flat/final_full_flat/$RUN_NAME/$benchmark/$TASK/$split/default
      if [[ -f $batch/metrics.json ]] && "$PYTHON_BIN" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(not (m["n_failed_runs"]==0 and m["n_total"]==m["n_successful"]>0))' "$batch/metrics.json"; then
        continue
      fi
      args=(--prompt final_full_flat --layout level-grouped --query-prior cached --prior-root "$prior"
        --provider-pool-config "$provider" --reasoning-effort high --parallelism 64 --execution-mode throughput
        --max-tokens 128000 --temperature none --disable-thinking --skip-existing)
      # Preparation renders the entire declared cohort and validates frozen inputs, without model calls.
      RUN=0 bash experiments/prediction/manual_uids.sh "${args[@]}" > "$root/${selection_cohort}_${profile}.prepare.log" 2>&1
      RUN=1 bash experiments/prediction/manual_uids.sh "${args[@]}" > "$root/${selection_cohort}_${profile}.prediction.log" 2>&1
    done
    if [[ $split == valid ]]; then
      "$PYTHON_BIN" - "$root" "$grid" "$cohort" "$STUDY_ID" "$benchmark" <<'PY'
import json, sys
from pathlib import Path
from optimization.selection.search import query_cohort
from predict.utils.json import write_json_atomic
root, grid, cohort, study, benchmark = sys.argv[1:]
profiles = json.loads(Path(grid).read_text())["profiles"]
queries, _ = query_cohort("bioavailability_ma", benchmark, cohort)
write_json_atomic(Path(root) / "candidates.json", dict(
    benchmark=benchmark, model="openai/gpt-6-luna", provider="OpenAI", selection_cohort=cohort,
    benchmark_row_ids=list(queries), candidates=[dict(profile=p["name"], coefficients=p, batch=str(
        Path("outputs/predictions/flat/final_full_flat") /
        f'oral_scaffold_luna_{study}_{benchmark}_{cohort}_{p["name"]}' /
        benchmark / "bioavailability_ma/valid/default")) for p in profiles]))
PY
      if [[ ! -f $root/winner.json ]]; then
        "$PYTHON_BIN" -m optimization.selection.search rank --candidates "$root/candidates.json" --output "$root/winner.json"
      fi
    fi
 done
done

"$PYTHON_BIN" - "$study" <<'PY'
import json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
study = Path(sys.argv[1])
results = []
for benchmark, expected in (("gold_v1", 269), ("tdc", 128)):
    winner_path = study / benchmark / "winner.json"
    winner = json.loads(winner_path.read_text())
    profile = winner["winner"]
    batch = Path("outputs/predictions/flat/final_full_flat") / f"oral_scaffold_luna_{study.name}_{benchmark}_test_{profile}" / benchmark / "bioavailability_ma/test/default"
    metrics = json.loads((batch / "metrics.json").read_text())
    predictions = list(read_jsonl(batch / "predictions.jsonl"))
    assert all(metrics[k] == expected for k in ("n_total", "n_evaluable", "n_successful")) and metrics["n_failed_runs"] == 0
    assert len(predictions) == len({p["query_index"] for p in predictions}) == expected
    providers = Counter()
    for p in predictions:
        assert p["status"] == p["final_status"] == "ok"
        response = json.loads((batch / "runs" / p["run_id"] / "final_reasoning_output.json").read_text())
        assert response["status"] == "ok" and response["llm"]["model"] == "openai/gpt-6-luna" and response["llm"]["provider"] == "OpenAI"
        providers[response["llm"]["provider"]] += 1
    results.append(dict(benchmark=benchmark, profile=profile, queries=expected, metrics=metrics,
        served_providers=dict(providers), batch=str(batch), winner_sha256=sha256_file(winner_path),
        metrics_sha256=sha256_file(batch / "metrics.json")))
receipt = dict(status="complete", verified_at=datetime.now(timezone.utc).isoformat(), tests=results)
write_json_atomic(study / "completion.json", receipt)
metadata = json.loads((study / "study.json").read_text())
metadata.update(status="complete", completion_receipt="completion.json")
write_json_atomic(study / "study.json", metadata)
lines = ["# Oral scaffold Luna indirect search", "", "Independent validation-selected winners; direct panels fixed. Gold: valid_small 100. TDC: full valid 64.", "", "| Benchmark | Indirect profile | Validation macro F1 | Test macro F1 | Test accuracy | Complete queries |", "| --- | --- | ---: | ---: | ---: | ---: |"]
for row in results:
    freeze = json.loads((study / row["benchmark"] / "winner.json").read_text())
    selected = next(p for p in freeze["leaderboard"] if p["profile"] == row["profile"])
    lines.append(f'| {row["benchmark"]} | {row["profile"]} | {selected["macro_f1"]:.6f} | {row["metrics"]["macro_f1"]:.6f} | {row["metrics"]["accuracy"]:.6f} | {row["queries"]} |')
lines += ["", "TDC L1 card scores remain uncomputed. Disjoint candidate exclusions for TDC and Gold test are recorded in each selection audit; no parent backfill is performed. Priors are reused unchanged. Generated artifacts are gitignored. No FINAL promotion."]
(study / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(receipt, indent=2))
PY
