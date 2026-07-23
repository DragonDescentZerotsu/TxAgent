#!/usr/bin/env bash
# Shared, non-secret machine paths for bioavailability assay-transfer workflows.
# Select with TXAGENT_RUNTIME_PROFILE=node002 or vast_slurm; any exported value
# may be overridden before sourcing this file.

set -euo pipefail

TXAGENT_RUNTIME_PROFILE="${TXAGENT_RUNTIME_PROFILE:-node002}"

case "${TXAGENT_RUNTIME_PROFILE}" in
  node002)
    : "${TXAGENT_PROJECT_ROOT:=/data1/joseph/TxAgent}"
    : "${TXAGENT_PYTHON:=/data1/joseph/miniconda3/envs/txagent-glm/bin/python}"
    : "${TXAGENT_REASONING_PYTHON:=/data1/joseph/miniconda3/envs/txagent-glm/bin/python}"
    : "${TXAGENT_STARLING_ROOT:=/data1/joseph/starling_assay_transfer}"
    : "${TXAGENT_HF_HOME:=}"
    : "${TXAGENT_LOCAL_FILES_ONLY:=0}"
    ;;
  vast_slurm)
    : "${TXAGENT_PROJECT_ROOT:=/vast/projects/myatskar/design-documents/joseph/TxAgent}"
    : "${TXAGENT_PYTHON:=/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python}"
    : "${TXAGENT_REASONING_PYTHON:=/vast/projects/myatskar/design-documents/conda_env/openrlhf/bin/python}"
    : "${TXAGENT_STARLING_ROOT:=/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer}"
    : "${TXAGENT_HF_HOME:=/vast/projects/myatskar/design-documents/hf_home}"
    : "${TXAGENT_LOCAL_FILES_ONLY:=1}"
    ;;
  *)
    echo "Unknown TXAGENT_RUNTIME_PROFILE=${TXAGENT_RUNTIME_PROFILE}; use node002 or vast_slurm" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

: "${TXAGENT_RERANK_DEVICES:=0,1,2,3}"
: "${TXAGENT_RERANK_BATCH_SIZE:=64}"
: "${TXAGENT_LIBRARY_WORKERS:=32}"

export TXAGENT_RUNTIME_PROFILE TXAGENT_PROJECT_ROOT TXAGENT_PYTHON
export TXAGENT_REASONING_PYTHON
export TXAGENT_STARLING_ROOT TXAGENT_HF_HOME TXAGENT_LOCAL_FILES_ONLY
export TXAGENT_RERANK_DEVICES TXAGENT_RERANK_BATCH_SIZE TXAGENT_LIBRARY_WORKERS

if [[ ! -x "${TXAGENT_PYTHON}" ]]; then
  echo "Profile ${TXAGENT_RUNTIME_PROFILE}: Python is not executable: ${TXAGENT_PYTHON}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${TXAGENT_REASONING_PYTHON}" ]]; then
  echo "Profile ${TXAGENT_RUNTIME_PROFILE}: reasoning Python is not executable: ${TXAGENT_REASONING_PYTHON}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -d "${TXAGENT_PROJECT_ROOT}" ]]; then
  echo "Profile ${TXAGENT_RUNTIME_PROFILE}: project root is missing: ${TXAGENT_PROJECT_ROOT}" >&2
  return 2 2>/dev/null || exit 2
fi
