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

# Status JSON written at start, updated on exit (any reason).
# Lets host see WHY a run produced no checkpoints — interrupted, failed, etc.
write_status() {
  cat > "$RESULTS_DIR/status.json" <<JSON
{
  "status": "$1",
  "exit_code": ${2:-null},
  "mode": "$MODE",
  "impl": "$IMPL",
  "ngpu": ${NGPU:-0},
  "started": "$STARTED",
  "ended": "$(date -Iseconds)"
}
JSON
}
STARTED="$(date -Iseconds)"
trap 'rc=$?; write_status "$([[ $rc -eq 0 ]] && echo completed || ([[ $rc -eq 130 ]] && echo interrupted || echo failed))" "$rc"' EXIT
write_status "starting" "null"

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
    # Canonical MLPerf HPC config wants 524288 train + 65536 val files
    # (= full ~1.68TB dataset). If only mini set is staged (~6GB / 1024+128
    # files), the upstream config triggers AssertionError. Auto-cap to
    # actual file count so full schedule runs as a throughput benchmark
    # even on mini data — accuracy target won't be reached but timing is.
    n_tr=$(find "$DATA_DIR/train" -maxdepth 2 \
             \( -name '*.tfrecord' -o -name '*.tfrecord.gz' -o -name '*.tfrecords' \) \
             2>/dev/null | wc -l)
    n_va=$(find "$DATA_DIR/validation" -maxdepth 2 \
             \( -name '*.tfrecord' -o -name '*.tfrecord.gz' -o -name '*.tfrecords' \) \
             2>/dev/null | wc -l)
    if [[ "$n_tr" -lt 524288 ]] || [[ "$n_va" -lt 65536 ]]; then
      echo "[WARN] mini dataset detected: $n_tr train + $n_va val files."
      echo "[WARN] capping --n-train/--n-valid to available; MLPerf accuracy target won't be hit."
      echo "[WARN] for canonical run, stage full ~1.68TB set: COSMOFLOW_VARIANT=full ./scripts/download_data.sh cosmoflow"
      COMMON_ARGS+=("--n-train" "$n_tr" "--n-valid" "$n_va")
    fi
    ;;
  *) echo "[ERR] --mode must be quick|full"; exit 1 ;;
esac

if [[ "$IMPL" == "pytorch" ]]; then
  echo "[WARN] --impl pytorch not bundled. Set CF_PYTORCH_REPO to a port and rebuild image."
  exit 2
fi

echo "[INFO] CosmoFlow: mode=$MODE gpus=$NGPU config=$CONFIG"
if [[ "$NGPU" -gt 1 ]] && command -v mpirun >/dev/null 2>&1; then
  # Upstream train.py uses horovod. Without --distributed it stays solo (rank=0 size=1)
  # and every rank grabs all GPUs -> GPU0 fragments to a few hundred MB and OOMs.
  # --rank-gpu makes train.py call configure_session(gpu=hvd.local_rank()), which
  # picks the right physical device. Do NOT also set CUDA_VISIBLE_DEVICES per rank:
  # that hides all but one GPU, then configure_session(gpu=N) hits OOB
  # ("GPU N unavailable, 1 visible").
  exec mpirun --allow-run-as-root -np "$NGPU" \
    --bind-to none -map-by slot \
    -x NCCL_DEBUG=WARN -x LD_LIBRARY_PATH -x PATH \
    python train.py "${COMMON_ARGS[@]}" --distributed --rank-gpu "${EXTRA[@]}"
else
  exec python train.py "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi
