#!/usr/bin/env bash
# CosmoFlow container entrypoint.
# Wraps mlcommons/hpc/cosmoflow train.py with mode-based config selection.
set -euo pipefail

MODE="quick"
IMPL="default"
DATA_DIR="/data/cosmoflow"
RESULTS_DIR="/results"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --impl) IMPL="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

mkdir -p "$RESULTS_DIR"

if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null || true)" ]]; then
  echo "[ERR] Cosmoflow data not found at $DATA_DIR."
  echo "      Run scripts/download_data.sh cosmoflow from the host."
  exit 1
fi

NGPU="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
NGPU="${NGPU:-1}"
[[ "$NGPU" -lt 1 ]] && NGPU=1

# Config select: mlcommons/hpc ships YAMLs under cosmoflow/configs/.
# quick -> override to tiny run via CLI flags; full -> canonical benchmark.
CONFIG="configs/cosmo.yaml"
[[ -f "configs/cosmo_nv.yaml" ]] && CONFIG="configs/cosmo_nv.yaml"

COMMON_ARGS=(
  "--data-dir" "$DATA_DIR"
  "--output-dir" "$RESULTS_DIR"
  "$CONFIG"
)

case "$MODE" in
  quick)
    COMMON_ARGS+=("--n-epochs" "2" "--n-train" "512" "--n-valid" "64")
    ;;
  full)
    # Use config defaults (MLPerf HPC target).
    :
    ;;
  *) echo "[ERR] --mode must be quick|full"; exit 1 ;;
esac

if [[ "$IMPL" == "pytorch" ]]; then
  echo "[WARN] --impl pytorch not bundled. Set CF_PYTORCH_REPO to a port and rebuild image."
  exit 2
fi

echo "[INFO] CosmoFlow: mode=$MODE gpus=$NGPU config=$CONFIG"
if [[ "$NGPU" -gt 1 ]] && command -v mpirun >/dev/null 2>&1; then
  exec mpirun --allow-run-as-root -np "$NGPU" python train.py "${COMMON_ARGS[@]}" "${EXTRA[@]}"
else
  exec python train.py "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi
