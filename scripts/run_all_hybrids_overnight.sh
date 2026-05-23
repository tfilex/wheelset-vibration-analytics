#!/usr/bin/env bash
set -Eeuo pipefail

GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[1;36m'
RESET=$'\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_all_hybrids_overnight"
LOG_ROOT="${LOG_ROOT:-reports/logs/nightly_all_hybrids/${RUN_ID}}"
MASTER_LOG="$LOG_ROOT/${RUN_ID}.log"
mkdir -p "$LOG_ROOT"

V3_SCRIPT="${V3_SCRIPT:-scripts/run_v3_rnn_balanced_freeze_then_finetune.sh}"
V4_SCRIPT="${V4_SCRIPT:-scripts/run_v4_tcn_balanced_freeze_then_finetune.sh}"
V5_SCRIPT="${V5_SCRIPT:-scripts/run_v5_odd_balanced_freeze_then_finetune.sh}"

GLOBAL_N_TRIALS="${N_TRIALS:-30}"
GLOBAL_EPOCHS="${EPOCHS:-25}"
GLOBAL_NUM_WORKERS="${NUM_WORKERS:-0}"
GLOBAL_WINDOW_SIZES="${WINDOW_SIZES:-}"
GLOBAL_TEMPORAL_TYPES="${TEMPORAL_TYPES:-}"
GLOBAL_FORCE_CNN_PRETRAIN="${FORCE_CNN_PRETRAIN:-0}"
GLOBAL_SKIP_CNN_PRETRAIN="${SKIP_CNN_PRETRAIN:-0}"

log() {
  local color="$1"
  local message="$2"
  printf '%s%s%s\n' "$color" "$message" "$RESET" | tee -a "$MASTER_LOG"
}

run_stage() {
  local stage_name="$1"
  local script_path="$2"
  local stage_log_dir="$3"
  local window_sizes="$4"
  local temporal_types="$5"
  local n_trials="$6"
  local epochs="$7"
  local experiment_name="$8"
  local force_pretrain="$9"

  mkdir -p "$stage_log_dir"

  log "$CYAN" "======================================================================"
  log "$CYAN" "[INFO] $(date '+%F %T') starting ${stage_name}"
  log "$YELLOW" "[INFO] script: ${script_path}"
  log "$YELLOW" "[INFO] log_dir: ${stage_log_dir}"
  log "$YELLOW" "[INFO] window_sizes: ${window_sizes}"
  log "$YELLOW" "[INFO] temporal_types: ${temporal_types}"
  log "$YELLOW" "[INFO] n_trials=${n_trials}, epochs=${epochs}, workers=${GLOBAL_NUM_WORKERS}"
  log "$CYAN" "======================================================================"

  LOG_DIR="$stage_log_dir" \
  WINDOW_SIZES="$window_sizes" \
  TEMPORAL_TYPES="$temporal_types" \
  N_TRIALS="$n_trials" \
  EPOCHS="$epochs" \
  NUM_WORKERS="$GLOBAL_NUM_WORKERS" \
  EXPERIMENT_NAME="$experiment_name" \
  FORCE_CNN_PRETRAIN="$force_pretrain" \
  SKIP_CNN_PRETRAIN="$GLOBAL_SKIP_CNN_PRETRAIN" \
  "$script_path" 2>&1 | tee -a "$MASTER_LOG"

  log "$GREEN" "======================================================================"
  log "$GREEN" "[INFO] $(date '+%F %T') finished ${stage_name}"
  log "$GREEN" "======================================================================"
}

log "$CYAN" "======================================================================"
log "$CYAN" "[INFO] Nightly all-hybrids run started: ${RUN_ID}"
log "$YELLOW" "[INFO] master log: ${MASTER_LOG}"
log "$YELLOW" "[INFO] FORCE_CNN_PRETRAIN=${GLOBAL_FORCE_CNN_PRETRAIN}, SKIP_CNN_PRETRAIN=${GLOBAL_SKIP_CNN_PRETRAIN}"
log "$CYAN" "======================================================================"

run_stage \
  "V3 RNN" \
  "$V3_SCRIPT" \
  "$LOG_ROOT/v3_rnn" \
  "${V3_WINDOW_SIZES:-${GLOBAL_WINDOW_SIZES:-1024}}" \
  "${V3_TEMPORAL_TYPES:-${GLOBAL_TEMPORAL_TYPES:-lstm bilstm lstm_attn bigru gru_attn transformer_improved}}" \
  "${V3_N_TRIALS:-$GLOBAL_N_TRIALS}" \
  "${V3_EPOCHS:-$GLOBAL_EPOCHS}" \
  "${V3_EXPERIMENT_NAME:-XJTU_SY_RUL_HybridV3_RNN_Overnight}" \
  "$GLOBAL_FORCE_CNN_PRETRAIN"

run_stage \
  "V4 TCN" \
  "$V4_SCRIPT" \
  "$LOG_ROOT/v4_tcn" \
  "${V4_WINDOW_SIZES:-${GLOBAL_WINDOW_SIZES:-2048}}" \
  "${V4_TEMPORAL_TYPES:-${GLOBAL_TEMPORAL_TYPES:-tcn tcn_ms tcna tcn_bi}}" \
  "${V4_N_TRIALS:-$GLOBAL_N_TRIALS}" \
  "${V4_EPOCHS:-$GLOBAL_EPOCHS}" \
  "${V4_EXPERIMENT_NAME:-XJTU_SY_RUL_HybridV4_TCN_Overnight}" \
  "0"

run_stage \
  "V5 ODD" \
  "$V5_SCRIPT" \
  "$LOG_ROOT/v5_odd" \
  "${V5_WINDOW_SIZES:-${GLOBAL_WINDOW_SIZES:-2048}}" \
  "${V5_TEMPORAL_TYPES:-${GLOBAL_TEMPORAL_TYPES:-patchtst conformer mamba}}" \
  "${V5_N_TRIALS:-$GLOBAL_N_TRIALS}" \
  "${V5_EPOCHS:-$GLOBAL_EPOCHS}" \
  "${V5_EXPERIMENT_NAME:-XJTU_SY_RUL_HybridV5_SOTA_Overnight}" \
  "0"

log "$GREEN" "[INFO] All overnight hybrid runs finished successfully."
