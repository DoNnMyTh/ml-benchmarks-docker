#!/usr/bin/env bash
# Convert ILSVRC2012 / Kaggle ImageNet layout to ImageFolder via hardlinks.
#
# Input layout (e.g. Kaggle "imagenet-object-localization-challenge"):
#   data/imagenet/Data/CLS-LOC/train/<wnid>/*.JPEG          # already class-foldered
#   data/imagenet/Data/CLS-LOC/val/ILSVRC2012_val_*.JPEG    # 50k flat files
#   data/imagenet/Annotations/CLS-LOC/val/ILSVRC2012_val_*.xml  # class label in <name>
#
# Output (in-place, hardlinks — no extra disk):
#   data/imagenet/train/<wnid>/*.JPEG
#   data/imagenet/val/<wnid>/ILSVRC2012_val_*.JPEG
#
# Why hardlinks: same NTFS/ext4 volume, 0 bytes extra, Docker Desktop bind mounts
# do NOT follow Windows NTFS junctions (they appear as broken symlinks inside the
# container), so a junction shortcut won't work — only real directory entries do.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/../common/lib.sh"

ROOT="${DATA_DIR}/imagenet"
SRC_TRAIN="${ROOT}/Data/CLS-LOC/train"
SRC_VAL_IMG="${ROOT}/Data/CLS-LOC/val"
SRC_VAL_ANN="${ROOT}/Annotations/CLS-LOC/val"

[[ -d "$SRC_TRAIN"   ]] || die "Missing $SRC_TRAIN — not an ILSVRC2012 layout under $ROOT"
[[ -d "$SRC_VAL_IMG" ]] || die "Missing $SRC_VAL_IMG"
[[ -d "$SRC_VAL_ANN" ]] || die "Missing $SRC_VAL_ANN — need val annotation XMLs to label val images"

if [[ -d "${ROOT}/train" ]] && [[ -d "${ROOT}/val" ]]; then
  log "ImageFolder layout already present at ${ROOT}/{train,val}. Nothing to do."
  exit 0
fi

log "Converting ILSVRC2012 layout → ImageFolder (hardlinks) at $ROOT ..."

# Run the reorg inside a tiny Python container. This avoids any host Python
# dependency and ensures the same code path works on Win/macOS/Linux.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${ROOT}:/imagenet" \
  python:3.11-slim python - <<'PY'
import os, re
from pathlib import Path

ROOT = Path("/imagenet")
SRC_TRAIN   = ROOT / "Data" / "CLS-LOC" / "train"
SRC_VAL_IMG = ROOT / "Data" / "CLS-LOC" / "val"
SRC_VAL_ANN = ROOT / "Annotations" / "CLS-LOC" / "val"
DST_TRAIN   = ROOT / "train"
DST_VAL     = ROOT / "val"

NAME_RE = re.compile(r"<name>(n\d+)</name>")

def link_train():
    if DST_TRAIN.is_dir() and any(DST_TRAIN.iterdir()):
        print("[SKIP] train/ already populated")
        return
    DST_TRAIN.mkdir(exist_ok=True)
    classes = sorted(d for d in SRC_TRAIN.iterdir() if d.is_dir())
    print(f"[TRAIN] {len(classes)} classes")
    total = 0
    for ci, cdir in enumerate(classes, 1):
        ddir = DST_TRAIN / cdir.name
        ddir.mkdir(exist_ok=True)
        for img in cdir.iterdir():
            dst = ddir / img.name
            if not dst.exists():
                try: os.link(img, dst); total += 1
                except OSError as e: print(f"[ERR] {img}: {e}")
        if ci % 100 == 0:
            print(f"  [{ci}/{len(classes)}] links={total}", flush=True)
    print(f"[TRAIN-DONE] {total} hardlinks")

def link_val():
    if DST_VAL.is_dir() and any(DST_VAL.iterdir()):
        print("[SKIP] val/ already populated")
        return
    DST_VAL.mkdir(exist_ok=True)
    xmls = sorted(SRC_VAL_ANN.glob("*.xml"))
    print(f"[VAL] {len(xmls)} XMLs → labels")
    made = 0
    for i, xml in enumerate(xmls, 1):
        stem = xml.stem
        img_src = SRC_VAL_IMG / f"{stem}.JPEG"
        if not img_src.exists():
            continue
        m = NAME_RE.search(xml.read_text(encoding="utf-8", errors="ignore"))
        if not m:
            continue
        wnid = m.group(1)
        ddir = DST_VAL / wnid
        ddir.mkdir(exist_ok=True)
        dst = ddir / f"{stem}.JPEG"
        if not dst.exists():
            try: os.link(img_src, dst); made += 1
            except OSError as e: print(f"[ERR] {stem}: {e}")
        if i % 5000 == 0:
            print(f"  [{i}/{len(xmls)}] links={made}", flush=True)
    print(f"[VAL-DONE] {made} hardlinks across {sum(1 for _ in DST_VAL.iterdir())} classes")

link_train()
link_val()
PY

log "Done. ImageFolder ready at:"
log "  ${ROOT}/train  ($(ls "${ROOT}/train" | wc -l) classes)"
log "  ${ROOT}/val    ($(ls "${ROOT}/val"   | wc -l) classes)"
