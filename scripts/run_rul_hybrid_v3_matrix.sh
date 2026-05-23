#!/usr/bin/env bash
set -Eeuo pipefail

GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[1;36m'
RESET=$'\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="reports/logs/rul_hybrid_v3_finetune"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_v3_ws1024_2048_all_models_feature_modes"

BASE_ARGS=(
  --profile balanced
  --n-trials 30
  --epochs 40
  --num-workers 0
)

# In current train_rul_hybrid_v3.py HPO always uses frozen CNN features, while
# final fit always fine-tunes the CNN. These modes track the CLI feature-cache
# flag and the artifact suffix: frozen -> --feature-cache, unfrozen -> --no-feature-cache.
run_model() {
  local window_size="$1"
  local feature_mode="$2"
  local feature_flag="$3"
  local temporal_type="$4"
  local log_path="$LOG_DIR/${RUN_ID}_ws${window_size}_${feature_mode}_${temporal_type}.log"

  echo "${CYAN}======================================================================${RESET}"
  echo "${CYAN}[INFO] $(date '+%F %T') starting ws=${window_size} ${feature_mode} ${temporal_type}${RESET}"
  echo "${YELLOW}[INFO] log: ${log_path}${RESET}"
  echo "${CYAN}======================================================================${RESET}"

  uv run python src/prediction/train_rul_hybrid_v3.py \
    "${BASE_ARGS[@]}" \
    --window-sizes "$window_size" \
    "$feature_flag" \
    --temporal-types "$temporal_type" \
    2>&1 | tee "$log_path"

  echo "${GREEN}======================================================================${RESET}"
  echo "${GREEN}[INFO] $(date '+%F %T') finished ws=${window_size} ${feature_mode} ${temporal_type}${RESET}"
  echo "${GREEN}======================================================================${RESET}"
}

for window_size in 1024 2048; do
  for feature_mode in frozen unfrozen; do
    if [[ "$feature_mode" == "frozen" ]]; then
      feature_flag="--feature-cache"
    else
      feature_flag="--no-feature-cache"
    fi

    run_model "$window_size" "$feature_mode" "$feature_flag" lstm
    run_model "$window_size" "$feature_mode" "$feature_flag" gru
    run_model "$window_size" "$feature_mode" "$feature_flag" transformer
  done
done

echo "${GREEN}[INFO] All requested train_rul_hybrid_v3 runs finished.${RESET}"
