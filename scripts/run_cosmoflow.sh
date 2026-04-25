#!/usr/bin/env bash
# End-to-end CosmoFlow runner.
# Usage: ./run_cosmoflow.sh [--mode quick|full] [--impl default|pytorch] [--gpus N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

print_help() {
  cat <<EOF
run_cosmoflow.sh — CosmoFlow MLCommons HPC benchmark

Flags:
  --mode quick|full     quick=2ep/512 train samples, full=canonical MLPerf HPC config
  --impl default|pytorch default=TF reference (official), pytorch=community port (not bundled)
  --gpus N              override GPU count (multi-GPU uses mpirun)
  --skip-data-check     skip dataset existence check
  --help                this
EOF
}

SKIP_DATA=0
parse_common_args "$@"
NEW_EXTRA=()
i=0
while [[ $i -lt ${#EXTRA[@]} ]]; do
  case "${EXTRA[$i]}" in
    --skip-data-check) SKIP_DATA=1; i=$((i+1)) ;;
    *) NEW_EXTRA+=("${EXTRA[$i]}"); i=$((i+1)) ;;
  esac
done
if [[ ${#NEW_EXTRA[@]} -gt 0 ]]; then EXTRA=("${NEW_EXTRA[@]}"); else EXTRA=(); fi

log "CosmoFlow: mode=$MODE impl=$IMPL"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  if [[ ! -d "${DATA_DIR}/cosmoflow" ]] || [[ -z "$(ls -A "${DATA_DIR}/cosmoflow" 2>/dev/null || true)" ]]; then
    warn "CosmoFlow data not found at ${DATA_DIR}/cosmoflow (~6GB mini tar)."
    confirm "Run downloader now?" && "$HERE/download_data.sh" cosmoflow || die "Aborted — no data."
  fi
fi

check_docker
ensure_image "cosmoflow" "${DOCKER_DIR}/cosmoflow"

args=(--mode "$MODE" --impl "$IMPL")
[[ ${#EXTRA[@]} -gt 0 ]] && args+=("${EXTRA[@]}")

docker_run "cosmoflow" "${args[@]}"
log "CosmoFlow done. Results under results/cosmoflow/<timestamp>/"
