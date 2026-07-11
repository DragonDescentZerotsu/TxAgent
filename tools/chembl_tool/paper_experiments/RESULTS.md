# Frozen Full-Run Results

Run date: 2026-07-10 to 2026-07-11.

The matrix contains 21 identity-blind GLM conditions and one scalar KNN baseline across BBB_Martins, Skin_Reaction, ClinTox, and Bioavailability_Ma. Every condition has `n_successful == n_total` and zero failed samples in the frozen metrics. The requested model was `zai-org/GLM-5.2-FP8`; the endpoint reported the served model as `hosted_vllm/nvidia/GLM-5.2-NVFP4`. Total recorded LLM usage was 116,137,805 tokens.

The machine-readable results, 10,000-replicate bootstrap intervals, exact McNemar tests, Holm corrections, coverage, source inventory, and cost audit are in:

```text
outputs/paper/molecular_evidence_agent/analysis/
```

## Main Metrics

| Task | None | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB_Martins | 0.7148 | 0.7325 | 0.7399 | 0.7248 | **0.7972** | n/a | n/a |
| Skin_Reaction | 0.6683 | 0.6450 | 0.6459 | **0.6822** | n/a | n/a | n/a |
| ClinTox | 0.4819 | 0.4819 | 0.5553 | **0.5891** | n/a | n/a | n/a |
| Bioavailability_Ma | 0.5525 | 0.6866 | 0.6715 | 0.6664 | 0.6908 numeric / 0.6579 full | 0.6938 | **0.7101** |

Values are test macro-F1. Bioavailability Starling direct has separate numeric-only and numerical+non-numerical conditions. Its scalar KNN baseline is 0.6952 macro-F1.

## Paired Conclusions

- Bioavailability retrieval is the clearest support for the first claim: none to ChEMBL direct is +0.1342 macro-F1, 95% CI 0.0471 to 0.2186, Holm-adjusted McNemar `p=0.0041`.
- BBB Starling direct is the clearest support for the literature-derived source claim: it exceeds ChEMBL direct by +0.0647, 95% CI 0.0167 to 0.1130, Holm-adjusted `p=0.0218`.
- Bioavailability Starling numeric direct does not significantly exceed ChEMBL direct (+0.0042). Adding non-numerical evidence to the direct condition lowers the point estimate by 0.0330 relative to numeric-only, with a CI crossing zero. The non-numerical claim is therefore not supported by this run.
- Mechanism decomposition has positive point estimates over flat retrieval for Skin_Reaction (+0.0363), ClinTox (+0.0338), and Starling Bioavailability (+0.0164), but none is significant after paired uncertainty and Holm correction. It is negative for BBB and ChEMBL Bioavailability.
- ClinTox direct retrieval has only 3.15% coverage and exactly matches the no-retrieval macro-F1. Flat/mechanism coverage is 99.30%; mechanism reaches 0.5891 and 22.22% toxic recall, but its gain over flat is not significant.
- The scalar KNN and GLM Starling-numeric direct conditions are statistically indistinguishable on Bioavailability. This experiment does not show that the agent beats a simple numerical-neighbor baseline.

The defensible paper conclusion is therefore narrower than the initial three claims: retrieval can help strongly, source quality can matter strongly, and mechanism decomposition can improve point estimates, but all three effects are task- and coverage-dependent. Only the Bioavailability retrieval effect and BBB Starling-vs-ChEMBL source effect survive this matrix's multiplicity correction.

## Audit Notes

- Prompt-boundary structure, identifier, and name leaks: 0 across all 21 LLM conditions.
- One raw BBB flat assistant reasoning trace reconstructed `*CC(C)CO` from allowed property/MMP evidence. The string was absent from all LLM request inputs and was sanitized before final synthesis. It is retained as an assistant-side reconstruction diagnostic, not counted as upstream identity disclosure.
- ClinTox contained three HTTP 413 failures caused by one analog carrying 1,824 evidence rows. The generic transport guard now leaves cleaned group payloads at or below 750 KB unchanged; oversized neighbors use deterministic even-spacing with at most 100 rows and explicit truncation metadata. All affected samples were rerun successfully.
- Structured JSON validation permits four total attempts. The final two only reserialize the same evidence; incomplete branches are rejected by the batch completion invariant.
