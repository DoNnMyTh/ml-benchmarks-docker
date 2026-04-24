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
    import torch
    from datasets import load_dataset
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
          f"epochs={epochs} max_steps={max_steps}")

    ds = load_dataset("imagefolder", data_dir=str(data_dir))
    labels = ds["train"].features["label"].names
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

    def transform(batch):
        imgs = [img.convert("RGB") for img in batch["image"]]
        batch["pixel_values"] = processor(imgs, return_tensors="pt")["pixel_values"]
        return batch

    ds = ds.with_transform(transform)

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
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", ds.get("val", ds["train"])),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    metrics = trainer.evaluate()
    metrics["train_seconds"] = train_s
    Path(args.results_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[DONE]", json.dumps(metrics, indent=2))


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

    metrics = {"epochs": []}
    t0 = time.time()
    global_step = 0
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
                print(f"  step {global_step} loss={loss.item():.4f}")
            if max_steps > 0 and global_step >= max_steps:
                break
        sched.step()
        # eval
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
        acc = correct / max(total, 1)
        print(f"[epoch {ep}] val_acc={acc:.4f}")
        metrics["epochs"].append({"epoch": ep, "val_acc": acc})
        if max_steps > 0 and global_step >= max_steps:
            break

    metrics["train_seconds"] = time.time() - t0
    Path(args.results_dir, "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[DONE]", json.dumps(metrics, indent=2))


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
