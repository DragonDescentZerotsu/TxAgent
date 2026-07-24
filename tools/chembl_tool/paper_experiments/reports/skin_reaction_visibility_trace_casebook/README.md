# Skin Reaction visibility trace casebook

`index.html` is a self-contained English/Chinese report for teammate review. It
uses matched-prefetch validation traces to isolate the effect of query/neighbor
identity visibility while holding retrieval and prefetched tool evidence fixed.

The report opens in English. Use the fixed language button in the upper-right
corner to switch to Chinese.

## Files

- `artifact.json`: bilingual Data Analytics artifact manifest and bounded snapshot.
- `skin_reaction_visibility_trace_casebook_source.jsonl`: compact case-level
  source records used by the artifact.
- `language_config.json`: report chrome and language-switcher labels.
- `index.html`: committed portable report; no server or external asset is required.

## Regeneration

From the repository root:

```bash
node /home/tianang/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input tools/chembl_tool/paper_experiments/reports/skin_reaction_visibility_trace_casebook/artifact.json \
  --output tools/chembl_tool/paper_experiments/reports/skin_reaction_visibility_trace_casebook/index.html

python -m tools.chembl_tool.paper_experiments.inject_portable_report_language_toggle \
  --html tools/chembl_tool/paper_experiments/reports/skin_reaction_visibility_trace_casebook/index.html \
  --config tools/chembl_tool/paper_experiments/reports/skin_reaction_visibility_trace_casebook/language_config.json
```

The first command uses the bundled Data Analytics portable-artifact exporter.
If its installed version changes, locate the current
`deliver_portable_artifact.mjs` under the Data Analytics plugin cache and keep
the artifact/config inputs unchanged.
