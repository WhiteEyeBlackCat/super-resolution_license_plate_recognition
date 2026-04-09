#!/usr/bin/env bash
# Run CMRM ablation study (5 experiments: LoRA / LoRA+CMRM variants)
# Experiments that already have history.json are automatically skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${WORKDIR}/model/outputs/cmrm_ablation/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "[$(date)] Starting ablation study — log: ${LOG_FILE}"

cd "${WORKDIR}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m train_cmrm_ablation 2>&1 | tee "${LOG_FILE}"

echo "[$(date)] Done. Outputs at: ${WORKDIR}/model/outputs/cmrm_ablation/"
