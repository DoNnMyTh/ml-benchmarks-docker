#!/usr/bin/env bash
# Dataset stager. All downloads land under ./data (gitignored).
# Usage: ./download_data.sh <imagenet|imagenette|cosmoflow|squad>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

which="${1:-}"
[[ -z "$which" ]] && die "Usage: $0 <imagenet|imagenette|cosmoflow|squad>"

mkdir -p "$DATA_DIR"

# ===== ImageNet-1k =====
# Workaround for no image-net.org account:
#   Path A (default): HuggingFace `ILSVRC/imagenet-1k` — gated but free.
#                     1) Create free HF account: https://huggingface.co/join
#                     2) Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
#                     3) Put token in .env as HF_TOKEN=hf_...
#   Path B (torrent): Academic Torrents ILSVRC2012 — needs aria2. Set IMAGENET_SRC=torrent.
imagenet() {
  local out="${DATA_DIR}/imagenet"
  mkdir -p "$out"
  if [[ -d "$out/train" ]] && [[ -d "$out/val" ]]; then
    log "ImageNet already staged at $out"
    return
  fi

  # User-supplied data: detect Kaggle/ILSVRC2012 layout and convert in-place.
  if [[ -d "$out/Data/CLS-LOC/train" ]] && [[ -d "$out/Annotations/CLS-LOC/val" ]]; then
    log "Detected ILSVRC2012/Kaggle layout at $out — converting to ImageFolder via hardlinks (no extra disk, no download)."
    "$HERE/prepare_imagenet.sh"
    return
  fi

  warn "ImageNet-1k is ~150GB. Ensure E: has that free."
  confirm "Proceed with download to $out ?" || die "Aborted."

  local src="${IMAGENET_SRC:-hf}"
  case "$src" in
    hf)
      [[ -z "${HF_TOKEN:-}" ]] && die "HF_TOKEN missing. Get at huggingface.co/settings/tokens, then accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k and set HF_TOKEN in .env"
      log "Downloading ImageNet-1k via HuggingFace (gated; terms acceptance required)..."
      MSYS_NO_PATHCONV=1 docker run --rm -it \
        -v "${out}:/out" \
        -e "HF_TOKEN=${HF_TOKEN}" \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        python:3.11-slim bash -c '
          set -e
          pip install --quiet "huggingface_hub[cli]" datasets pillow
          python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir="/out/_hf_raw",
    local_dir_use_symlinks=False,
    max_workers=8,
)
PY
          echo "[INFO] Raw files in /out/_hf_raw. Now extracting to ImageFolder layout..."
          python - <<PY
# Convert HF parquet/tar shards to ImageFolder. Minimal — expand as needed.
import os, tarfile, glob, shutil
base = "/out/_hf_raw"
# HF dataset ships train/val as tar shards in data/ dir — extract to ImageFolder.
for split_src, split_dst in [("train", "/out/train"), ("val", "/out/val")]:
    os.makedirs(split_dst, exist_ok=True)
    for tar in sorted(glob.glob(f"{base}/data/{split_src}*.tar*")):
        print("Extracting", tar)
        with tarfile.open(tar) as t:
            t.extractall(split_dst)
PY
          echo "[OK] ImageNet extracted to /out/train and /out/val"
        '
      ;;
    torrent)
      die "Torrent path not implemented in this script. Install aria2, grab magnet from https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2 and extract to $out/{train,val}."
      ;;
    *) die "IMAGENET_SRC must be hf|torrent (got: $src)" ;;
  esac
}

# ===== ImageNette (fast smoke test, ~1.5GB, no auth) =====
imagenette() {
  local out="${DATA_DIR}/imagenette"
  mkdir -p "$out"
  if [[ -d "$out/train" ]] && [[ -d "$out/val" ]]; then
    log "ImageNette already staged at $out"; return
  fi
  local url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
  log "Downloading ImageNette (~340MB) ..."
  curl -L "$url" -o "$out/imagenette2-320.tgz"
  tar -xzf "$out/imagenette2-320.tgz" -C "$out" --strip-components=1
  # imagenette uses train/ and val/ already.
  rm "$out/imagenette2-320.tgz"
  log "ImageNette ready at $out"
}

# ===== CosmoFlow (mini ~6GB tar, or full preprocessed v2 ~1.68TB tar) =====
# Variant controlled by COSMOFLOW_VARIANT={mini,full} env, else interactive prompt.
# Custom URL via COSMOFLOW_URL env (overrides variant choice).
cosmoflow() {
  local out="${DATA_DIR}/cosmoflow"
  mkdir -p "$out"

  # User-supplied tarball: auto-extract and skip download.
  local tarballs=( "$out"/*.tar "$out"/*.tar.gz "$out"/*.tgz )
  for tb in "${tarballs[@]}"; do
    [[ -f "$tb" ]] || continue
    log "Found user-supplied tarball $tb — extracting in place."
    tar -xf "$tb" -C "$out"
    rm "$tb"
    break
  done

  if [[ -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
    log "CosmoFlow dir non-empty: $out (skipping download)"; return
  fi

  local base="https://portal.nersc.gov/project/dasrepo/cosmoflow-benchmark"
  local variant="${COSMOFLOW_VARIANT:-}"
  if [[ -z "$variant" ]] && [[ -z "${COSMOFLOW_URL:-}" ]]; then
    echo
    echo "CosmoFlow dataset variants (NERSC public HTTPS):"
    echo "  1) mini  — cosmoUniverse_2019_05_4parE_tf_v2_mini.tar  (~6 GB, 32 train + 32 val)"
    echo "             Smoke / throughput runs. Won't reach MLPerf accuracy target."
    echo "  2) full  — cosmoUniverse_2019_05_4parE_tf_v2.tar       (~1.68 TB, MLPerf HPC reference)"
    echo "             Need ≥2 TB free (download stored as tar then extracted) or use stream mode."
    read -r -p "Choose [1=mini / 2=full] (default 1): " ans
    case "$ans" in
      2|full|FULL) variant="full" ;;
      *) variant="mini" ;;
    esac
  fi

  local url size_label
  if [[ -n "${COSMOFLOW_URL:-}" ]]; then
    url="$COSMOFLOW_URL"; size_label="custom"
  elif [[ "$variant" == "full" ]]; then
    url="$base/cosmoUniverse_2019_05_4parE_tf_v2.tar"; size_label="~1.68TB"
  else
    url="$base/cosmoUniverse_2019_05_4parE_tf_v2_mini.tar"; size_label="~6GB"
  fi

  warn "Downloading CosmoFlow $variant ($size_label) from $url"
  if [[ "$variant" == "full" ]]; then
    warn "Full set is 1.68 TB. Peak disk during extract is ~3.4 TB (tar + extracted)"
    warn "unless you stream-extract: COSMOFLOW_STREAM=1 (pipes curl into tar, no .tar saved)."
  fi
  confirm "Proceed with download to $out ?" || die "Aborted."

  if [[ "${COSMOFLOW_STREAM:-0}" == "1" ]]; then
    log "Streaming download → tar extract (no intermediate file) ..."
    curl -L --fail "$url" | tar -xf - -C "$out"
  else
    local tar_path="$out/cosmoflow_$variant.tar"
    log "Downloading to $tar_path (resumable; re-run to continue partial)"
    curl -L --fail -C - "$url" -o "$tar_path"
    log "Extracting $tar_path ..."
    tar -xf "$tar_path" -C "$out"
    rm "$tar_path"
  fi
  log "CosmoFlow data ready at $out"
}

# ===== SQuAD v2 (~50MB, public) =====
squad() {
  local out="${DATA_DIR}/squad"
  mkdir -p "$out"
  log "SQuAD v2 will auto-download on first benchmark run via HF datasets (cache dir: $out)."
  log "If you want to pre-stage, run:"
  cat <<'EOF'
  docker run --rm -v "$(pwd)/data/squad:/cache" python:3.11-slim bash -c \
    'pip install -q datasets && python -c "from datasets import load_dataset; load_dataset(\"squad_v2\", cache_dir=\"/cache\")"'
EOF
}

case "$which" in
  imagenet)   imagenet ;;
  imagenette) imagenette ;;
  cosmoflow)  cosmoflow ;;
  squad)      squad ;;
  *) die "Unknown dataset: $which" ;;
esac
