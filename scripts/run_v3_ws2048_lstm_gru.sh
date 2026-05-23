#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo $'\033[1;33m[WARN] scripts/run_v3_ws2048_lstm_gru.sh is deprecated.\033[0m'
echo $'\033[1;33m[WARN] Use scripts/run_rul_hybrid_v3_matrix.sh instead.\033[0m'

exec ./scripts/run_rul_hybrid_v3_matrix.sh "$@"
