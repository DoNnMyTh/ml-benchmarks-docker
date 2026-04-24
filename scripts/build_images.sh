#!/usr/bin/env bash
# Build all 3 benchmark images as donnmyth/ml-benchmarks:<tag>.
# Usage: ./build_images.sh [resnet50|cosmoflow|bert-squad|all]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

target="${1:-all}"

build_one() {
  local tag="$1" ctx="${DOCKER_DIR}/$1"
  local full="${DOCKER_USER}/${DOCKER_REPO}:${tag}"
  log "Building $full from $ctx"
  docker build -t "$full" "$ctx"
}

case "$target" in
  resnet50|cosmoflow|bert-squad) build_one "$target" ;;
  all)
    build_one resnet50
    build_one cosmoflow
    build_one bert-squad
    ;;
  *) die "Unknown target: $target" ;;
esac

log "Done. List:"
docker image ls "${DOCKER_USER}/${DOCKER_REPO}"
