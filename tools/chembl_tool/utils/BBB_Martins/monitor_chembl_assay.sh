#!/usr/bin/env bash
set -u

SESSION="${1:-chembl_assay}"
OUT_DIR="${2:-outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw}"
INTERVAL_SECONDS="${3:-5400}"
LOG="${4:-/tmp/chembl_assay_monitor.log}"

cd /data1/tianang/Projects/TxAgent || exit 1

while true; do
  ts="$(date -Is)"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    pane="$(tmux capture-pane -t "${SESSION}" -p -S -120 2>/dev/null || true)"
    if printf "%s\n" "${pane}" | grep -q "Finished BBB assay screening"; then
      echo "${ts} finished_detected" >> "${LOG}"
      python -m tools.chembl_tool.tasks.bbb_martins.summarize_outputs \
        --out-dir "${OUT_DIR}" \
        --report-path "${OUT_DIR}/bbb_health_check.md" >> "${LOG}" 2>&1
      exit 0
    fi
    last="$(printf "%s\n" "${pane}" | grep "\\[progress\\]\\|\\[progress-final\\]\\|\\[stage\\]\\|\\[done\\]" | tail -n 1)"
    echo "${ts} still_running ${last:-no_status_line}" >> "${LOG}"
  else
    echo "${ts} tmux_session_missing_checking_outputs" >> "${LOG}"
    if [ -s "${OUT_DIR}/bbb_assay_candidates.csv" ]; then
      python -m tools.chembl_tool.tasks.bbb_martins.summarize_outputs \
        --out-dir "${OUT_DIR}" \
        --report-path "${OUT_DIR}/bbb_health_check.md" >> "${LOG}" 2>&1
      exit 0
    fi
  fi
  sleep "${INTERVAL_SECONDS}"
done
