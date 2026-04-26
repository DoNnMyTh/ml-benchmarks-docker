#!/usr/bin/env python3
"""ResNet50 / ImageNet-1k training benchmark.

Impls:
  default : HuggingFace microsoft/resnet-50 via transformers Trainer (recommended).
  mlperf  : torchvision ResNet50 reference (MLPerf-style from-scratch training).

Modes:
  quick   : 1 epoch, subset / capped steps — sanity + throughput.
  full    : 90 epochs (configurable), target ~76% top-1 (MLPerf target).

Data layout expected at /data/imagenet:
  /data/imagenet/train/<wnid>/*.JPEG
  /data/imagenet/val/<wnid>/*.JPEG
  (OR) /data/imagenette/train|val/... when --dataset imagenette.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["default", "mlperf"], default="default")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--dataset", choices=["imagenet", "imagenette"], default="imagenet")
    p.add_argument("--data-root", default="/data")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def resolve_data_dir(root: str, dataset: str) -> Path:
    d = Path(root) / dataset
    if not (d / "train").is_dir() or not (d / "val").is_dir():
        sys.exit(
            f"[ERR] Expected {d}/train and {d}/val to exist.\n"
            f"      Run scripts/download_data.sh or mount your dataset at {d}."
        )
    return d


def run_hf(args, data_dir: Path):
    """HF microsoft/resnet-50 fine-tune via Trainer.

    Uses torchvision.datasets.ImageFolder (os.scandir-based, fast even on
    parallel/network filesystems) instead of HF `load_dataset("imagefolder", ...)`
    which calls fsspec.glob("**/*") and hashes every path for the arrow cache —
    that walk takes hours-to-days on slow FS for 1.3M ImageNet files.
    """
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets as tv_datasets
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )
    import evaluate

    num_gpus = int(os.environ.get("NUM_GPUS", "0") or torch.cuda.device_count())
    per_device_bs = args.batch_size or (128 if num_gpus else 32)
    epochs = args.epochs or (1 if args.mode == "quick" else 90)
    lr = args.lr or 0.1
    max_steps = args.max_steps if args.max_steps is not None else (200 if args.mode == "quick" else -1)

    print(f"[INFO] impl=hf dataset={args.dataset} gpus={num_gpus} bs/dev={per_device_bs} "
          f"epochs={epochs} max_steps={max_steps}", flush=True)

    val_dir = (data_dir / "val") if (data_dir / "val").is_dir() else (data_dir / "validation")
    print(f"[INFO] indexing train tree at {data_dir/'train'} (torchvision.datasets.ImageFolder) ...",
          flush=True)
    tv_train = tv_datasets.ImageFolder(str(data_dir / "train"))
    print(f"[INFO] train: {len(tv_train)} samples, {len(tv_train.classes)} classes", flush=True)
    print(f"[INFO] indexing val tree at {val_dir} ...", flush=True)
    tv_val = tv_datasets.ImageFolder(str(val_dir))
    print(f"[INFO] val: {len(tv_val)} samples, {len(tv_val.classes)} classes", flush=True)

    labels = tv_train.classes
    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for c, i in label2id.items()}

    model_id = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    class HFImageFolder(Dataset):
        """Wrap a torchvision.ImageFolder so HF Trainer gets dict-shaped items."""
        def __init__(self, tv_ds, processor):
            self.tv = tv_ds
            self.processor = processor
        def __len__(self):
            return len(self.tv)
        def __getitem__(self, idx):
            img, label = self.tv[idx]
            pixel_values = self.processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            return {"pixel_values": pixel_values, "labels": label}

    train_ds = HFImageFolder(tv_train, processor)
    eval_ds  = HFImageFolder(tv_val,   processor)

    acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.argmax(axis=-1)
        return acc.compute(predictions=preds, references=eval_pred.label_ids)

    targs = TrainingArguments(
        output_dir=args.results_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=1e-4,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=8,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    metrics = {"status": "starting", "impl": "hf", "dataset": args.dataset,
               "gpus": num_gpus, "epochs": epochs, "max_steps": max_steps}
    _save = lambda: Path(args.results_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save()  # write skeleton so dir is non-empty even if next step crashes

    t0 = time.time()
    try:
        trainer.train()
        metrics["status"] = "trained"
        metrics["train_seconds"] = time.time() - t0
        _save()
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)
        metrics["status"] = "completed"
    except KeyboardInterrupt:
        metrics["status"] = "interrupted"
        metrics["train_seconds"] = time.time() - t0
        print("[WARN] interrupted — saving partial metrics", flush=True)
    except Exception as e:
        metrics["status"] = "failed"
        metrics["error"] = repr(e)
        metrics["train_seconds"] = time.time() - t0
        print(f"[ERR] {e!r}", flush=True)
        raise
    finally:
        _save()
        print("[DONE]", json.dumps(metrics, indent=2), flush=True)


def run_mlperf(args, data_dir: Path):
    """Minimal torchvision reference. For true MLPerf submission use the official
    mlcommons/training repo; this is a faithful-but-lean variant for benchmarking."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    num_gpus = int(os.environ.get("NUM_GPUS", "0") or torch.cuda.device_count())
    bs = args.batch_size or (256 if num_gpus else 64)
    epochs = args.epochs or (1 if args.mode == "quick" else 90)
    lr = args.lr or 0.1 * max(num_gpus, 1)
    max_steps = args.max_steps if args.max_steps is not None else (200 if args.mode == "quick" else -1)

    print(f"[INFO] impl=mlperf gpus={num_gpus} bs={bs} epochs={epochs} max_steps={max_steps}")

    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = datasets.ImageFolder(str(data_dir / "train"), transform=train_t)
    va = datasets.ImageFolder(str(data_dir / "val"), transform=val_t)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None, num_classes=len(tr.classes)).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    metrics = {"status": "starting", "impl": "mlperf", "gpus": num_gpus,
               "epochs": epochs, "max_steps": max_steps, "epoch_results": []}
    _save = lambda: Path(args.results_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save()

    t0 = time.time()
    global_step = 0
    try:
        for ep in range(epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    out = model(x); loss = crit(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                global_step += 1
                if global_step % 20 == 0:
                    print(f"  step {global_step} loss={loss.item():.4f}", flush=True)
                if max_steps > 0 and global_step >= max_steps:
                    break
            sched.step()
            model.eval(); correct = total = 0
            with torch.no_grad():
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    correct += (pred == y).sum().item(); total += y.size(0)
            acc = correct / max(total, 1)
            print(f"[epoch {ep}] val_acc={acc:.4f}", flush=True)
            metrics["epoch_results"].append({"epoch": ep, "val_acc": acc})
            metrics["train_seconds"] = time.time() - t0
            _save()  # incremental save per epoch
            if max_steps > 0 and global_step >= max_steps:
                break
        metrics["status"] = "completed"
    except KeyboardInterrupt:
        metrics["status"] = "interrupted"
        print("[WARN] interrupted — saving partial metrics", flush=True)
    except Exception as e:
        metrics["status"] = "failed"
        metrics["error"] = repr(e)
        print(f"[ERR] {e!r}", flush=True)
        raise
    finally:
        metrics["train_seconds"] = time.time() - t0
        metrics["completed_steps"] = global_step
        _save()
        print("[DONE]", json.dumps(metrics, indent=2), flush=True)


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args.data_root, args.dataset)
    if args.impl == "default":
        run_hf(args, data_dir)
    else:
        run_mlperf(args, data_dir)


if __name__ == "__main__":
    main()
