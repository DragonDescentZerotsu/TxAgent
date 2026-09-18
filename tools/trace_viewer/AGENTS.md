# Trace Viewer compatibility path

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

The canonical viewer now lives in `predict/traces/viewer/` and reads both the
standard and progressive `predict_trace.v1` records under
`predict/traces/runs/`. The command below remains supported through a thin
wrapper. New viewer code belongs under `predict/traces/viewer/`.

# Trace Viewer Running and Temporary Public Sharing

This directory maintains the final paper trace viewer. The viewer is not a single-file offline page; `viewer.html` dynamically reads condition, `predictions.jsonl`, sample trace, and retrieval files from the same HTTP root directory. Therefore, do not upload only `viewer.html` to a single-file HTML pastebin, and do not treat a local file URL as a complete viewer for sharing.

The viewer registers benchmark datasets through the `.trace_viewer_sources.tsv` generated at startup and registers existing conditions with `.trace_viewer_catalog.tsv`, avoiding per-directory scanning on the public page; if the catalog is missing, it can still fall back to browser-side discovery. Within each dataset, only four official paper roots are registered: identity-blind, matched-prefetch, agentic deployment-visible, and deployment-visible parent-disjoint. Dataset path/label is not hardcoded in `viewer.html`; when adding a new benchmark or split, simply register a new trace root via the startup script. Parent-disjoint samples must read both the manifest and `reuse.json`, clearly display `neighbor_identity_policy`, and distinguish reruns after retrieval changes from artifact reuse when inputs are unchanged.

Condition results must be computed from the current `predictions.jsonl` for sample counts, accuracy, macro-F1, failure counts, and binary confusion matrix when available; do not read task-specific prediction fields, and do not merge samples from different benchmark datasets.

## Display Names and Internal IDs

The viewer can provide concise, readable display names for paper-facing mechanism families, but must not rewrite stable `group_id` in traces or retrieval. Friendly names are used for stage, pill, and retrieval group titles; original IDs continue to be displayed in retrieval title tags, group metadata, and raw JSON to preserve auditability of provenance, input hash, and branch reuse.

Bioavailability display names are uniformly Direct oral bioavailability (F%), Oral exposure proxies (AUC/Cmax), and mechanism descriptions for Fa, Fg, and Fh. Do not display the `Observed` prefix as an additional reasoning branch.

## Local Startup

Run from the repository root:

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

By default, it registers three result roots: Starling random test, Starling scaffold test, and historical TDC test; non-existent roots are skipped. You can also explicitly pass any number of trace roots after the port:

```bash
bash tools/trace_viewer/start_viewer.sh 8776 \
  outputs/paper/molecular_evidence_agent_starling_random \
  outputs/paper/molecular_evidence_agent_starling_scaffold
```

The local page is fixed at:

```text
http://127.0.0.1:8776/.trace_viewer.html?v=paper-v2
```

`start_viewer.sh` runs as a foreground process; the terminal staying at `Serving HTTP ...` and continuously printing GET logs is normal, not a hang. `GET /favicon.ico ... 404` is just the browser automatically requesting a missing icon and does not affect the viewer.

## Installing cloudflared on node002

`node002` is Linux amd64, and `$HOME/.local/bin` is already in PATH. For temporary sharing, prefer the Cloudflare official standalone binary, installed to the user directory, without sudo, and without modifying system package repositories.

The following command reads the current latest release from the Cloudflare official GitHub release API and installs after verifying the SHA256 digest published in the release asset. Do not hardcode a historical version number or checksum in long-term documentation.

```bash
(
  set -euo pipefail

  release_json="$(curl -fsSL https://api.github.com/repos/cloudflare/cloudflared/releases/latest)"
  asset_name="cloudflared-linux-amd64"
  download_url="$(printf '%s' "$release_json" | jq -r --arg name "$asset_name" '.assets[] | select(.name == $name) | .browser_download_url')"
  sha256="$(printf '%s' "$release_json" | jq -r --arg name "$asset_name" '.assets[] | select(.name == $name) | .digest' | sed 's/^sha256://')"
  tmp="$(mktemp)"

  test -n "$download_url"
  test -n "$sha256"
  test "$download_url" != "null"
  test "$sha256" != "null"

  curl -fL --retry 3 -o "$tmp" "$download_url"
  printf '%s  %s\n' "$sha256" "$tmp" | sha256sum -c -

  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp" "$HOME/.local/bin/cloudflared"
  rm -f "$tmp"
)

hash -r
cloudflared --version
```

If the standalone binary is already installed, you can check for updates later with `cloudflared update`.

## Creating a Quick Tunnel

Keep the local viewer terminal running, and open another terminal to execute:

```bash
cloudflared tunnel --url http://127.0.0.1:8776
```

On success, the logs will provide a temporary base address, for example:

```text
https://random-words.trycloudflare.com
```

Cloudflare only prints the public base address and does not know the specific viewer page path. For the full URL to share, append the path after the port from the local URL to the public base address:

```text
https://random-words.trycloudflare.com/.trace_viewer.html?v=paper-v2
```

General concatenation rule:

```text
public base address + /.trace_viewer.html?v=paper-v2
```

Both foreground processes must remain running:

```text
start_viewer.sh serves HTML and trace files.
cloudflared maps local 8776 to a temporary public domain.
```

After sharing ends, press `Ctrl-C` in both terminals. Quick Tunnel typically gets a new random domain after restart; do not write temporary domains into code, documentation, or experiment manifests.

## Log Diagnosis and Troubleshooting

The following logs indicate the tunnel is established normally:

```text
Your quick Tunnel has been created
Registered tunnel connection
SUMMARY: Environment is healthy
```

Quick Tunnel does not require `config.yml`, so `Cannot determine default configuration path` is informational, not a failure. QUIC's UDP receive-buffer warning can also be ignored if `Registered tunnel connection` has already appeared and connectivity pre-checks all PASS.

If QUIC/UDP cannot establish but TCP/443 is available, switch to HTTP/2:

```bash
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8776
```

Before sharing, verify the local page; if local is not working, the tunnel cannot fix the origin:

```bash
curl -I 'http://127.0.0.1:8776/.trace_viewer.html?v=paper-v2'
```

## Security Boundaries

Quick Tunnel is a public, unauthenticated, no-uptime-guarantee temporary development entry point. `start_viewer.sh` only links the trace roots registered this time in the temporary serving directory, not exposing the repository or the entire `outputs/paper/`; however, anyone with the URL may still request other files in these registered roots, not just the currently open sample in the browser. Therefore:

1. Only start the tunnel when you confirm that traces, prompts, SMILES, labels, internal paths, and model/tool return content can be shared.
2. Do not use Quick Tunnel to expose API keys, `.env`, tool service ports, LLM endpoints, or the repository root.
3. Immediately shut down both processes after the temporary demo ends; for long-term, fixed domains, or access control, use Cloudflare named tunnels and Access policies instead of relying on anonymous Quick Tunnel.
