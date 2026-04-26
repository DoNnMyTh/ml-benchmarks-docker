#!/usr/bin/env bash
# Shared helpers for run_*.sh scripts.
# Source: `. "$(dirname "$0")/../common/lib.sh"`

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR_LIB/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
RESULTS_DIR="${REPO_ROOT}/results"
DOCKER_DIR="${REPO_ROOT}/docker"

# ---- Load .env if present ----
if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

# ---- Defaults ----
DOCKER_USER="${DOCKER_USER:-donnmyth}"
DOCKER_REPO="${DOCKER_REPO:-ml-benchmarks}"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[ERR ]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# Detect GPU count. Prefer NUM_GPUS env, else nvidia-smi.
detect_gpus() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "$NUM_GPUS"; return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo "0"
  fi
}

# Verify Docker daemon + nvidia runtime.
check_docker() {
  command -v docker >/dev/null 2>&1 || die "docker not found in PATH"
  docker info >/dev/null 2>&1 || die "docker daemon not reachable"
  # GPU capability check (best-effort; --gpus flag may still fail on first run if WSL integration not set)
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
    warn "Docker GPU runtime test failed. Ensure: Docker Desktop WSL2 backend + 'Use GPU' enabled, or nvidia-container-toolkit installed on Linux."
  fi
}

# Ensure image present. $1=tag, $2=context dir.
# Order: cached → docker pull from Hub → local build fallback.
# Skip pull with NO_PULL=1, force build with FORCE_BUILD=1.
ensure_image() {
  local tag="$1" ctx="$2"
  local full="${DOCKER_USER}/${DOCKER_REPO}:${tag}"

  if [[ "${FORCE_BUILD:-0}" != "1" ]] && docker image inspect "$full" >/dev/null 2>&1; then
    log "Using cached image $full"
    return
  fi

  if [[ "${FORCE_BUILD:-0}" != "1" ]] && [[ "${NO_PULL:-0}" != "1" ]]; then
    log "Pulling $full from Docker Hub ..."
    if docker pull "$full" 2>/dev/null; then
      log "Pulled $full"
      return
    fi
    warn "Pull failed (image may not be published or no network). Falling back to local build."
  fi

  log "Building $full from $ctx ..."
  docker build -t "$full" "$ctx"
}

# Common docker run flags for a benchmark container.
# Usage: docker_run <tag> <extra args to pass to entrypoint>
docker_run() {
  local tag="$1"; shift
  local full="${DOCKER_USER}/${DOCKER_REPO}:${tag}"
  local gpus; gpus="$(detect_gpus)"
  local gpu_flag=()
  if [[ "$gpus" -gt 0 ]]; then
    gpu_flag=(--gpus all)
    log "Running with $gpus GPU(s)"
  else
    warn "No GPUs detected — running CPU-only (will be slow)."
  fi

  local run_id; run_id="$(date +%Y%m%d-%H%M%S)"
  local out_dir="${RESULTS_DIR}/${tag}/${run_id}"
  mkdir -p "$out_dir"
  log "Results → $out_dir"

  # Pass HF_TOKEN if present
  local env_flags=()
  [[ -n "${HF_TOKEN:-}" ]] && env_flags+=(-e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")

  local tty_flag=()
  [[ -t 0 ]] && tty_flag=(-it)

  # Save run metadata (cmd, image, args) so failed runs are still diagnosable.
  {
    echo "image: $full"
    echo "tag: $tag"
    echo "args: $*"
    echo "started: $(date -Iseconds)"
    echo "host: $(hostname 2>/dev/null || echo unknown)"
    echo "gpus: $gpus"
  } > "${out_dir}/run.meta"

  # MSYS_NO_PATHCONV=1: prevent Git Bash from mangling `/data` and `/results`
  # container paths into `C:\Program Files\Git\data` etc.
  # PYTHONUNBUFFERED=1: belt-and-suspenders so log lines flush even if Dockerfile
  # ENV is overridden or python -u not used. Lets us see why runs die early.
  set +e
  MSYS_NO_PATHCONV=1 docker run --rm "${tty_flag[@]}" \
    "${gpu_flag[@]}" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${DATA_DIR}:/data" \
    -v "${out_dir}:/results" \
    "${env_flags[@]}" \
    -e "NUM_GPUS=${gpus}" \
    -e "PYTHONUNBUFFERED=1" \
    "$full" "$@" 2>&1 | tee "${out_dir}/run.log"
  local rc=${PIPESTATUS[0]}
  set -e

  {
    echo "exit_code: $rc"
    echo "ended: $(date -Iseconds)"
  } >> "${out_dir}/run.meta"

  if [[ "$rc" -eq 0 ]]; then
    log "Container exited 0 → results in $out_dir"
  else
    err "Container exited $rc — partial output (if any) in $out_dir"
    err "Common causes: 137=OOM-killed, 130=Ctrl+C, 1=python traceback (see end of run.log)"
  fi
  return $rc
}

# Confirm before destructive or long-running op.
confirm() {
  local prompt="${1:-Continue?}"
  read -r -p "$prompt [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

parse_common_args() {
  MODE="${BENCH_MODE:-quick}"
  IMPL="default"
  EXTRA=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) MODE="$2"; shift 2 ;;
      --impl) IMPL="$2"; shift 2 ;;
      --gpus) NUM_GPUS="$2"; shift 2 ;;
      --help|-h) print_help; exit 0 ;;
      *) EXTRA+=("$1"); shift ;;
    esac
  done
  [[ "$MODE" =~ ^(quick|full)$ ]] || die "--mode must be quick|full (got: $MODE)"
}
