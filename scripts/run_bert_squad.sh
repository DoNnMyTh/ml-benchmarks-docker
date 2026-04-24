#!/usr/bin/env bash
# End-to-end BERT-SQuAD2 runner.
# Usage: ./run_bert_squad.sh [--mode quick|full] [--impl default|eval] [--gpus N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

print_help() {
  cat <<EOF
run_bert_squad.sh — BERT-base-uncased fine-tune on SQuAD v2

Flags:
  --mode quick|full     quick=1ep/500 steps, full=2ep convergence
  --impl default|eval   default=train from bert-base-uncased,
                        eval=load deepset/bert-base-uncased-squad2 and score F1/EM only
  --gpus N              override GPU count
  --skip-data-check     skip dataset stage (HF auto-downloads if missing)
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
EXTRA=("${NEW_EXTRA[@]:-}")

log "BERT-SQuAD2: mode=$MODE impl=$IMPL"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  if [[ ! -d "${DATA_DIR}/squad" ]]; then
    log "SQuAD not cached locally; will auto-download in container (~50MB)."
    mkdir -p "${DATA_DIR}/squad"
  fi
fi

check_docker
ensure_image "bert-squad" "${DOCKER_DIR}/bert-squad"

args=(--mode "$MODE" --impl "$IMPL")
[[ ${#EXTRA[@]} -gt 0 ]] && args+=("${EXTRA[@]}")

docker_run "bert-squad" "${args[@]}"
log "BERT-SQuAD2 done. Metrics under results/bert-squad/<timestamp>/metrics.json"
