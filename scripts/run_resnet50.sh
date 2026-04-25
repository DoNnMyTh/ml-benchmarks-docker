#!/usr/bin/env bash
# End-to-end: build (if needed) -> stage data -> run -> collect results.
# Usage: ./run_resnet50.sh [--mode quick|full] [--impl default|mlperf]
#        [--dataset imagenet|imagenette] [--gpus N] [--skip-data-check]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

print_help() {
  cat <<EOF
run_resnet50.sh — ResNet50 / ImageNet-1k benchmark (HF or MLPerf-style)

Flags:
  --mode quick|full     quick=1ep/200 steps, full=90ep convergence (default: quick)
  --impl default|mlperf default=HF microsoft/resnet-50, mlperf=torchvision ref
  --dataset imagenet|imagenette  (default: imagenet; imagenette for 1.5GB smoke test)
  --gpus N              override GPU count
  --skip-data-check     don't verify dataset exists before launch
  --help                this
EOF
}

DATASET="imagenet"
SKIP_DATA=0
parse_common_args "$@"
# extract dataset + skip-data from EXTRA
NEW_EXTRA=()
i=0
while [[ $i -lt ${#EXTRA[@]} ]]; do
  case "${EXTRA[$i]}" in
    --dataset) DATASET="${EXTRA[$((i+1))]}"; i=$((i+2)) ;;
    --skip-data-check) SKIP_DATA=1; i=$((i+1)) ;;
    *) NEW_EXTRA+=("${EXTRA[$i]}"); i=$((i+1)) ;;
  esac
done
if [[ ${#NEW_EXTRA[@]} -gt 0 ]]; then EXTRA=("${NEW_EXTRA[@]}"); else EXTRA=(); fi

log "ResNet50: mode=$MODE impl=$IMPL dataset=$DATASET"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  data_path="${DATA_DIR}/${DATASET}"
  if [[ ! -d "${data_path}/train" ]] || [[ ! -d "${data_path}/val" ]]; then
    warn "Dataset not found at $data_path"
    echo "Run: ./scripts/download_data.sh $DATASET"
    confirm "Run downloader now?" && "$HERE/download_data.sh" "$DATASET" || die "Aborted — no data."
  fi
fi

check_docker
ensure_image "resnet50" "${DOCKER_DIR}/resnet50"

args=(--mode "$MODE" --impl "$IMPL" --dataset "$DATASET")
[[ ${#EXTRA[@]} -gt 0 ]] && args+=("${EXTRA[@]}")

docker_run "resnet50" "${args[@]}"
log "ResNet50 done. Metrics under results/resnet50/<timestamp>/metrics.json"
