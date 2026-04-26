#!/usr/bin/env bash
# Convert ILSVRC2012 / Kaggle ImageNet layout to ImageFolder via hardlinks.
#
# Strategy: prefer host Python (no Docker bind-mount overhead — critical on
# Docker Desktop / Windows where 9P metadata syscalls are 100x slower than
# native). Fall back to a python:3.11-slim container only if no host Python.
#
# Skip (force Docker): set FORCE_DOCKER_PREPARE=1.
# Skip Docker fallback (host only): set FORCE_HOST_PREPARE=1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

ROOT="${DATA_DIR}/imagenet"
PY_SCRIPT="${HERE}/prepare_imagenet.py"

[[ -d "$ROOT/Data/CLS-LOC/train"           ]] || die "Missing $ROOT/Data/CLS-LOC/train — not an ILSVRC2012 layout"
[[ -d "$ROOT/Data/CLS-LOC/val"             ]] || die "Missing $ROOT/Data/CLS-LOC/val"
[[ -d "$ROOT/Annotations/CLS-LOC/val"      ]] || die "Missing $ROOT/Annotations/CLS-LOC/val"
[[ -f "$PY_SCRIPT"                         ]] || die "Missing $PY_SCRIPT"

if [[ -d "${ROOT}/train" ]] && [[ -d "${ROOT}/val" ]] \
   && [[ -n "$(ls -A "${ROOT}/train" 2>/dev/null || true)" ]] \
   && [[ -n "$(ls -A "${ROOT}/val"   2>/dev/null || true)" ]]; then
  log "ImageFolder layout already present at ${ROOT}/{train,val}. Nothing to do."
  exit 0
fi

# Pick a Python interpreter on the host.
host_py=""
if [[ "${FORCE_DOCKER_PREPARE:-0}" != "1" ]]; then
  for cand in python3 python py; do
    if command -v "$cand" >/dev/null 2>&1; then
      # `py` on Windows is the launcher; needs `-3` arg. Skip unless others fail.
      if [[ "$cand" == "py" ]]; then host_py="py -3"; else host_py="$cand"; fi
      break
    fi
  done
fi

if [[ -n "$host_py" ]]; then
  log "Running prepare_imagenet.py on HOST Python ($host_py) — direct filesystem access, fast."
  DATA_ROOT="$ROOT" $host_py "$PY_SCRIPT"
elif [[ "${FORCE_HOST_PREPARE:-0}" == "1" ]]; then
  die "FORCE_HOST_PREPARE=1 set but no host Python found (looked for python3, python, py)."
else
  warn "No host Python found. Falling back to python:3.11-slim container."
  warn "On Docker Desktop / Windows this is *much* slower (9P bind-mount metadata stalls)."
  warn "Install Python on the host or set up WSL2 native-fs to avoid this."
  MSYS_NO_PATHCONV=1 docker run --rm -i \
    -v "${ROOT}:/imagenet" \
    -v "${PY_SCRIPT}:/prepare.py:ro" \
    -e DATA_ROOT=/imagenet \
    python:3.11-slim python /prepare.py
fi

log "Done. ImageFolder ready at:"
log "  ${ROOT}/train  ($(ls "${ROOT}/train" 2>/dev/null | wc -l) classes)"
log "  ${ROOT}/val    ($(ls "${ROOT}/val"   2>/dev/null | wc -l) classes)"
