#!/usr/bin/env python3
"""BERT-base-uncased on SQuAD v2 — train or eval.

Impls:
  default : fine-tune bert-base-uncased on SQuAD v2 (training benchmark).
  eval    : load deepset/bert-base-uncased-squad2 and report F1/EM only.

Modes (train):
  quick : 1 epoch, max 500 steps.
  full  : 2 epochs (HF reference).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["default", "eval"], default="default")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--base-model", default="bert-base-uncased")
    p.add_argument("--eval-model", default="deepset/bert-base-uncased-squad2")
    p.add_argument("--data-root", default="/data")
    p.add_argument("--results-dir", default="/results")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--doc-stride", type=int, default=128)
    return p.parse_args()


def load_squad(data_root: str):
    """Load SQuAD v2. Prefer local cache at /data/squad, else HF hub."""
    from datasets import load_dataset
    local = Path(data_root) / "squad"
    if (local / "dataset_info.json").exists() or any(local.glob("**/dataset_info.json")):
        print(f"[INFO] Loading SQuAD from local cache {local}")
        return load_dataset(str(local))
    print("[INFO] Loading SQuAD v2 from HuggingFace hub (squad_v2)")
    ds = load_dataset("squad_v2", cache_dir=str(local))
    return ds


def prepare_train_features(examples, tokenizer, max_seq_len, doc_stride):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tok = tokenizer(
        examples["question"], examples["context"],
        truncation="only_second", max_length=max_seq_len,
        stride=doc_stride, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    offset_map = tok.pop("offset_mapping")
    start_pos, end_pos = [], []
    for i, offsets in enumerate(offset_map):
        input_ids = tok["input_ids"][i]
        cls_idx = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tok.sequence_ids(i)
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            start_pos.append(cls_idx); end_pos.append(cls_idx); continue
        s_char = answers["answer_start"][0]
        e_char = s_char + len(answers["text"][0])
        tok_start = 0
        while seq_ids[tok_start] != 1: tok_start += 1
        tok_end = len(input_ids) - 1
        while seq_ids[tok_end] != 1: tok_end -= 1
        if not (offsets[tok_start][0] <= s_char and offsets[tok_end][1] >= e_char):
            start_pos.append(cls_idx); end_pos.append(cls_idx)
        else:
            while tok_start < len(offsets) and offsets[tok_start][0] <= s_char: tok_start += 1
            start_pos.append(tok_start - 1)
            while offsets[tok_end][1] >= e_char: tok_end -= 1
            end_pos.append(tok_end + 1)
    tok["start_positions"] = start_pos
    tok["end_positions"] = end_pos
    return tok


def prepare_val_features(examples, tokenizer, max_seq_len, doc_stride):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tok = tokenizer(
        examples["question"], examples["context"],
        truncation="only_second", max_length=max_seq_len,
        stride=doc_stride, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    tok["example_id"] = []
    for i in range(len(tok["input_ids"])):
        seq_ids = tok.sequence_ids(i)
        sample_idx = sample_map[i]
        tok["example_id"].append(examples["id"][sample_idx])
        tok["offset_mapping"][i] = [
            (o if seq_ids[k] == 1 else None) for k, o in enumerate(tok["offset_mapping"][i])
        ]
    return tok


def postprocess(examples, features, predictions, n_best=20, max_ans_len=30, null_threshold=0.0):
    all_start, all_end = predictions
    ex_to_feats = collections.defaultdict(list)
    for i, f in enumerate(features):
        ex_to_feats[f["example_id"]].append(i)
    preds = {}
    for ex in examples:
        eid = ex["id"]; context = ex["context"]
        min_null = None; best_non_null = None
        for fi in ex_to_feats[eid]:
            sl = all_start[fi]; el = all_end[fi]
            offsets = features[fi]["offset_mapping"]
            cls = features[fi]["input_ids"].index(0) if 0 in features[fi]["input_ids"] else 0
            null_score = sl[cls] + el[cls]
            if min_null is None or null_score < min_null: min_null = null_score
            s_idx = np.argsort(sl)[-n_best:][::-1]
            e_idx = np.argsort(el)[-n_best:][::-1]
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_ans_len: continue
                    score = sl[s] + el[e]
                    if best_non_null is None or score > best_non_null["score"]:
                        best_non_null = {"score": score, "text": context[offsets[s][0]:offsets[e][1]]}
        if best_non_null is None or (min_null is not None and min_null - best_non_null["score"] > null_threshold):
            preds[eid] = ""
        else:
            preds[eid] = best_non_null["text"]
    return preds


def run_train(args):
    import torch
    from transformers import (
        AutoModelForQuestionAnswering, AutoTokenizer,
        Trainer, TrainingArguments, default_data_collator,
    )
    import evaluate

    ds = load_squad(args.data_root)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)

    num_gpus = int(os.environ.get("NUM_GPUS", "0") or torch.cuda.device_count())
    bs = args.batch_size or (32 if num_gpus else 8)
    epochs = args.epochs or (1 if args.mode == "quick" else 2)
    max_steps = args.max_steps if args.max_steps is not None else (500 if args.mode == "quick" else -1)

    print(f"[INFO] train gpus={num_gpus} bs/dev={bs} epochs={epochs} max_steps={max_steps}")

    train_ds = ds["train"].map(
        lambda x: prepare_train_features(x, tok, args.max_seq_len, args.doc_stride),
        batched=True, remove_columns=ds["train"].column_names,
    )
    val_raw = ds["validation"]
    val_ds = val_raw.map(
        lambda x: prepare_val_features(x, tok, args.max_seq_len, args.doc_stride),
        batched=True, remove_columns=val_raw.column_names,
    )

    targs = TrainingArguments(
        output_dir=args.results_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=args.lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
    )

    scores = {"status": "starting", "impl": "train", "gpus": num_gpus,
              "epochs": epochs, "max_steps": max_steps}
    _save = lambda: Path(args.results_dir, "metrics.json").write_text(json.dumps(scores, indent=2))
    _save()

    t0 = time.time()
    try:
        trainer.train()
        scores["status"] = "trained"
        scores["train_seconds"] = time.time() - t0
        _save()

        raw = trainer.predict(val_ds.remove_columns(["example_id", "offset_mapping"]))
        preds = postprocess(val_raw, val_ds, raw.predictions)
        metric = evaluate.load("squad_v2")
        formatted_preds = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in preds.items()]
        refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_raw]
        scores.update(metric.compute(predictions=formatted_preds, references=refs))
        scores["status"] = "completed"
    except KeyboardInterrupt:
        scores["status"] = "interrupted"
        scores["train_seconds"] = time.time() - t0
        print("[WARN] interrupted — saving partial metrics", flush=True)
    except Exception as e:
        scores["status"] = "failed"
        scores["error"] = repr(e)
        scores["train_seconds"] = time.time() - t0
        print(f"[ERR] {e!r}", flush=True)
        raise
    finally:
        _save()
        print("[DONE]", json.dumps(scores, indent=2), flush=True)


def run_eval(args):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    import evaluate

    ds = load_squad(args.data_root)
    tok = AutoTokenizer.from_pretrained(args.eval_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.eval_model)

    val_raw = ds["validation"]
    val_ds = val_raw.map(
        lambda x: prepare_val_features(x, tok, args.max_seq_len, args.doc_stride),
        batched=True, remove_columns=val_raw.column_names,
    )

    num_gpus = int(os.environ.get("NUM_GPUS", "0") or torch.cuda.device_count())
    bs = args.batch_size or (64 if num_gpus else 8)

    targs = TrainingArguments(
        output_dir=args.results_dir,
        per_device_eval_batch_size=bs,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, tokenizer=tok, data_collator=default_data_collator)

    scores = {"status": "starting", "impl": "eval", "gpus": num_gpus, "model": args.eval_model}
    _save = lambda: Path(args.results_dir, "metrics.json").write_text(json.dumps(scores, indent=2))
    _save()

    t0 = time.time()
    try:
        raw = trainer.predict(val_ds.remove_columns(["example_id", "offset_mapping"]))
        scores["eval_seconds"] = time.time() - t0
        preds = postprocess(val_raw, val_ds, raw.predictions)
        metric = evaluate.load("squad_v2")
        formatted = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in preds.items()]
        refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_raw]
        scores.update(metric.compute(predictions=formatted, references=refs))
        scores["status"] = "completed"
    except KeyboardInterrupt:
        scores["status"] = "interrupted"
        scores["eval_seconds"] = time.time() - t0
        print("[WARN] interrupted", flush=True)
    except Exception as e:
        scores["status"] = "failed"
        scores["error"] = repr(e)
        scores["eval_seconds"] = time.time() - t0
        print(f"[ERR] {e!r}", flush=True)
        raise
    finally:
        _save()
        print("[DONE]", json.dumps(scores, indent=2), flush=True)


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    if args.impl == "eval":
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
