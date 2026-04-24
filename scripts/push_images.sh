#!/usr/bin/env bash
# Push images to Docker Hub (assumes `docker login` already done).
# Usage: ./push_images.sh [resnet50|cosmoflow|bert-squad|all]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

target="${1:-all}"

push_one() {
  local tag="$1"
  local full="${DOCKER_USER}/${DOCKER_REPO}:${tag}"
  docker image inspect "$full" >/dev/null 2>&1 || die "$full not built. Run scripts/build_images.sh $tag first."
  log "Pushing $full ..."
  docker push "$full"
}

case "$target" in
  resnet50|cosmoflow|bert-squad) push_one "$target" ;;
  all)
    push_one resnet50
    push_one cosmoflow
    push_one bert-squad
    ;;
  *) die "Unknown target: $target" ;;
esac

log "Pushed. Public at: https://hub.docker.com/r/${DOCKER_USER}/${DOCKER_REPO}/tags"
