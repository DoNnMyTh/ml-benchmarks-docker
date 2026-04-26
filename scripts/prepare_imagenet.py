#!/usr/bin/env python3
"""Convert ILSVRC2012/Kaggle layout under DATA_ROOT to ImageFolder via hardlinks.

Reads:
  $DATA_ROOT/Data/CLS-LOC/train/<wnid>/*.JPEG
  $DATA_ROOT/Data/CLS-LOC/val/ILSVRC2012_val_*.JPEG
  $DATA_ROOT/Annotations/CLS-LOC/val/ILSVRC2012_val_*.xml

Writes (hardlinks, same volume = ~0 extra disk):
  $DATA_ROOT/train/<wnid>/*.JPEG
  $DATA_ROOT/val/<wnid>/ILSVRC2012_val_*.JPEG

Why hardlinks not symlinks/junctions: Docker Desktop bind mounts on Windows
do not follow NTFS reparse points; hardlinks are real directory entries and
work transparently.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DATA_ROOT", "/imagenet"))
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
    print(f"[TRAIN] {len(classes)} classes", flush=True)
    total = 0
    for ci, cdir in enumerate(classes, 1):
        ddir = DST_TRAIN / cdir.name
        ddir.mkdir(exist_ok=True)
        for img in cdir.iterdir():
            dst = ddir / img.name
            if not dst.exists():
                try:
                    os.link(img, dst)
                    total += 1
                except OSError as e:
                    print(f"[ERR] {img}: {e}")
        if ci % 100 == 0:
            print(f"  [{ci}/{len(classes)}] links={total}", flush=True)
    print(f"[TRAIN-DONE] {total} hardlinks")


def link_val():
    if DST_VAL.is_dir() and any(DST_VAL.iterdir()):
        print("[SKIP] val/ already populated")
        return
    DST_VAL.mkdir(exist_ok=True)
    xmls = sorted(SRC_VAL_ANN.glob("*.xml"))
    print(f"[VAL] {len(xmls)} XMLs -> labels", flush=True)
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
            try:
                os.link(img_src, dst)
                made += 1
            except OSError as e:
                print(f"[ERR] {stem}: {e}")
        if i % 5000 == 0:
            print(f"  [{i}/{len(xmls)}] links={made}", flush=True)
    classes = sum(1 for _ in DST_VAL.iterdir())
    print(f"[VAL-DONE] {made} hardlinks across {classes} classes")


def main():
    for p in (SRC_TRAIN, SRC_VAL_IMG, SRC_VAL_ANN):
        if not p.is_dir():
            sys.exit(f"[ERR] missing {p} — not an ILSVRC2012/Kaggle layout under {ROOT}")
    link_train()
    link_val()


if __name__ == "__main__":
    main()
