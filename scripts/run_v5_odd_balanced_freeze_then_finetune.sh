#!/usr/bin/env bash
set -Eeuo pipefail

GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[1;36m'
RESET=$'\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-reports/logs/train_rul_hybrid_v5_odd_balanced}"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_v5odd_balanced_freeze_then_finetune"
TRAIN_SCRIPT="src/prediction/train_rul_hybrid_v5_odd.py"

# Defaults target the new V5 SOTA heads on the heavier 2048 window.
# mamba is included by default; train_rul_hybrid_v5_odd.py skips it gracefully
# when mamba-ssm is not installed.
# Override examples:
#   WINDOW_SIZES="1024 2048" ./scripts/run_v5_odd_balanced_freeze_then_finetune.sh
#   TEMPORAL_TYPES="patchtst conformer" ./scripts/run_v5_odd_balanced_freeze_then_finetune.sh
WINDOW_SIZES_RAW="${WINDOW_SIZES:-2048}"
TEMPORAL_TYPES_RAW="${TEMPORAL_TYPES:-patchtst conformer mamba}"
read -r -a WINDOW_SIZES_ARR <<< "$WINDOW_SIZES_RAW"
read -r -a TEMPORAL_TYPES_ARR <<< "$TEMPORAL_TYPES_RAW"

N_TRIALS="${N_TRIALS:-10}"
EPOCHS="${EPOCHS:-15}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-XJTU_SY_RUL_HybridV5_SOTA_Balanced}"
RUL_CNN_CHECKPOINT="${RUL_CNN_CHECKPOINT:-models/cnn/best_resnet18_rul.pth}"
CNN_PRETRAIN_SCRIPT="${CNN_PRETRAIN_SCRIPT:-src/prediction/pretrain_cnn_rul.py}"
CNN_PRETRAIN_EPOCHS="${CNN_PRETRAIN_EPOCHS:-10}"
CNN_PRETRAIN_BATCH_SIZE="${CNN_PRETRAIN_BATCH_SIZE:-64}"
CNN_PRETRAIN_LR="${CNN_PRETRAIN_LR:-5e-4}"

if [[ -n "${PY_CMD:-}" ]]; then
  read -r -a PY_RUNNER <<< "$PY_CMD"
elif command -v uv >/dev/null 2>&1; then
  PY_RUNNER=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  PY_RUNNER=(.venv/bin/python)
else
  PY_RUNNER=(python)
fi

ensure_rul_cnn_checkpoint() {
  local log_path="$LOG_DIR/${RUN_ID}_pretrain_cnn_rul.log"

  if [[ "${FORCE_CNN_PRETRAIN:-0}" == "1" && "${SKIP_CNN_PRETRAIN:-0}" == "1" ]]; then
    echo "${YELLOW}[ERROR] FORCE_CNN_PRETRAIN=1 and SKIP_CNN_PRETRAIN=1 are mutually exclusive.${RESET}" >&2
    exit 2
  fi

  if [[ -f "$RUL_CNN_CHECKPOINT" && "${FORCE_CNN_PRETRAIN:-0}" != "1" ]]; then
    echo "${GREEN}[INFO] Reusing RUL CNN checkpoint: ${RUL_CNN_CHECKPOINT}${RESET}"
    return
  fi

  if [[ "${SKIP_CNN_PRETRAIN:-0}" == "1" ]]; then
    echo "${YELLOW}[ERROR] RUL CNN checkpoint is required but missing: ${RUL_CNN_CHECKPOINT}${RESET}" >&2
    exit 1
  fi

  echo "${CYAN}======================================================================${RESET}"
  echo "${CYAN}[INFO] $(date '+%F %T') pretraining RUL CNN checkpoint${RESET}"
  echo "${YELLOW}[INFO] checkpoint: ${RUL_CNN_CHECKPOINT}${RESET}"
  echo "${YELLOW}[INFO] log: ${log_path}${RESET}"
  echo "${CYAN}======================================================================${RESET}"

  "${PY_RUNNER[@]}" "$CNN_PRETRAIN_SCRIPT" \
    --epochs "$CNN_PRETRAIN_EPOCHS" \
    --batch_size "$CNN_PRETRAIN_BATCH_SIZE" \
    --lr "$CNN_PRETRAIN_LR" \
    2>&1 | tee "$log_path"

  if [[ ! -f "$RUL_CNN_CHECKPOINT" ]]; then
    echo "${YELLOW}[ERROR] CNN pretrain finished but checkpoint was not created: ${RUL_CNN_CHECKPOINT}${RESET}" >&2
    exit 1
  fi
  echo "${GREEN}[INFO] RUL CNN checkpoint ready: ${RUL_CNN_CHECKPOINT}${RESET}"
}

BASE_ARGS=(
  --profile balanced
  --n-trials "$N_TRIALS"
  --epochs "$EPOCHS"
  --num-workers "$NUM_WORKERS"
  --experiment-name "$EXPERIMENT_NAME"
  --final-fit-modes frozen finetune
)

run_model() {
  local window_size="$1"
  local temporal_type="$2"
  local log_path="$LOG_DIR/${RUN_ID}_ws${window_size}_${temporal_type}.log"
  local -a cmd

  cmd=("${PY_RUNNER[@]}" "$TRAIN_SCRIPT")

  echo "${CYAN}======================================================================${RESET}"
  echo "${CYAN}[INFO] $(date '+%F %T') starting ws=${window_size} model=${temporal_type} final_modes=frozen,finetune${RESET}"
  echo "${YELLOW}[INFO] runner: ${PY_RUNNER[*]}${RESET}"
  echo "${YELLOW}[INFO] log: ${log_path}${RESET}"
  echo "${CYAN}======================================================================${RESET}"

  "${cmd[@]}" \
    "${BASE_ARGS[@]}" \
    --window-sizes "$window_size" \
    --feature-cache \
    --temporal-types "$temporal_type" \
    2>&1 | tee "$log_path"

  echo "${GREEN}======================================================================${RESET}"
  echo "${GREEN}[INFO] $(date '+%F %T') finished ws=${window_size} model=${temporal_type} final_modes=frozen,finetune${RESET}"
  echo "${GREEN}======================================================================${RESET}"
}

echo "${YELLOW}[INFO] Window sizes: ${WINDOW_SIZES_ARR[*]}${RESET}"
echo "${YELLOW}[INFO] Temporal types: ${TEMPORAL_TYPES_ARR[*]}${RESET}"
echo "${YELLOW}[INFO] Balanced args: n_trials=${N_TRIALS}, epochs=${EPOCHS}, workers=${NUM_WORKERS}${RESET}"

ensure_rul_cnn_checkpoint

for window_size in "${WINDOW_SIZES_ARR[@]}"; do
  for temporal_type in "${TEMPORAL_TYPES_ARR[@]}"; do
    run_model "$window_size" "$temporal_type"
  done
done

echo "${GREEN}[INFO] All requested train_rul_hybrid_v5_odd balanced runs finished.${RESET}"
