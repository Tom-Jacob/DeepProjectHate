#!/usr/bin/env python
"""Fine-tune bert-base-uncased as a generic (non-specialized) toxicity classifier.

This is one half of the "same base model, two methods" experiment: bert-base
is used here with a trained classification head, and separately in
model_registry.BertBaseMlmPromptScorer as a zero-shot masked-LM prompt — same
checkpoint family, different method. Comparing this against toxic-bert also
restores the specialization axis (generic vs. in-domain-specialized encoder).

Trains on a stratified subsample of the train split (full-dataset fine-tuning
is impractical on CPU) with a weighted loss for the ~10% positive rate, and
evaluates on a val subsample after each epoch.

Usage:
    python scripts/train_bert_base_classifier.py
    python scripts/train_bert_base_classifier.py --train-n 12000 --epochs 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from device import get_device  # noqa: E402
from paths import BERT_BASE_MODEL_ID, LOCAL_BERT_BASE_DIR, SEED  # noqa: E402
from splits import sample_split_frame  # noqa: E402

MAX_LENGTH = 128


def encode_texts(tokenizer, texts, max_length=MAX_LENGTH):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def iterate_batches(n, batch_size, shuffle_rng=None):
    idx = np.arange(n)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


@torch.no_grad()
def evaluate(model, tokenizer, texts, labels, device, batch_size=32):
    model.eval()
    probs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = encode_texts(tokenizer, batch_texts)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    pr_auc = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return f1, pr_auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-n", type=int, default=16_000)
    parser.add_argument("--val-n", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save-every", type=int, default=250,
                        help="Checkpoint every N steps (in addition to end-of-epoch), 0 to disable")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing checkpoint in models/bert-base-toxic and start from bert-base-uncased")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_df = sample_split_frame("train", n=args.train_n, seed=SEED)
    val_df = sample_split_frame("val", n=args.val_n, seed=SEED)
    train_texts = train_df["comment_text"].astype(str).tolist()
    train_labels = train_df["toxic"].astype(int).values
    val_texts = val_df["comment_text"].astype(str).tolist()
    val_labels = val_df["toxic"].astype(int).values
    print(f"Train: {len(train_texts)} rows ({train_labels.mean()*100:.1f}% positive)")
    print(f"Val:   {len(val_texts)} rows ({val_labels.mean()*100:.1f}% positive)")

    state_path = LOCAL_BERT_BASE_DIR / "train_state.json"
    resume = (not args.fresh) and (LOCAL_BERT_BASE_DIR / "config.json").exists() and state_path.exists()
    epoch_done = 0
    if resume:
        with open(state_path, encoding="utf-8") as f:
            epoch_done = json.load(f)["epoch_done"]
        print(f"Resuming from checkpoint at {LOCAL_BERT_BASE_DIR} (epoch_done={epoch_done})")
        load_from = str(LOCAL_BERT_BASE_DIR)
    else:
        load_from = BERT_BASE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=2,
        id2label={0: "non-toxic", 1: "toxic"},
        label2id={"non-toxic": 0, "toxic": 1},
    ).to(device)

    # Class weights for the ~10% positive rate (inverse frequency).
    n_pos = max(int(train_labels.sum()), 1)
    n_neg = max(len(train_labels) - n_pos, 1)
    weight = torch.tensor(
        [len(train_labels) / (2.0 * n_neg), len(train_labels) / (2.0 * n_pos)],
        dtype=torch.float,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(SEED)
    n_train = len(train_texts)
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    print(f"{args.epochs} epoch(s) x {steps_per_epoch} steps ({args.batch_size}/batch)")

    if epoch_done >= args.epochs:
        print(f"Checkpoint already has {epoch_done} epoch(s) >= requested {args.epochs}; nothing to do.")
        return

    for epoch in range(epoch_done + 1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        for step, batch_idx in enumerate(iterate_batches(n_train, args.batch_size, rng), start=1):
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_labels = torch.tensor(train_labels[batch_idx], dtype=torch.long, device=device)
            enc = encode_texts(tokenizer, batch_texts)
            enc = {k: v.to(device) for k, v in enc.items()}

            optimizer.zero_grad()
            logits = model(**enc).logits
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 50 == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - t0
                print(
                    f"  epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss={running_loss/step:.4f} ({elapsed:.0f}s elapsed)",
                    flush=True,
                )

            if args.save_every and step % args.save_every == 0:
                LOCAL_BERT_BASE_DIR.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(LOCAL_BERT_BASE_DIR)
                tokenizer.save_pretrained(LOCAL_BERT_BASE_DIR)
                print(f"  mid-epoch checkpoint saved (step {step})", flush=True)

        val_f1, val_pr_auc = evaluate(model, tokenizer, val_texts, val_labels, device)
        print(
            f"epoch {epoch} done in {time.perf_counter()-t0:.0f}s | "
            f"val F1={val_f1:.3f} val PR-AUC={val_pr_auc:.3f}"
        )

        # Checkpoint after every epoch so a crash doesn't lose all progress
        # (CPU training is slow enough that a full restart is expensive).
        LOCAL_BERT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(LOCAL_BERT_BASE_DIR)
        tokenizer.save_pretrained(LOCAL_BERT_BASE_DIR)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"epoch_done": epoch}, f)
        print(f"  checkpoint saved to {LOCAL_BERT_BASE_DIR}")

    print(f"\nFinal fine-tuned bert-base-uncased saved to {LOCAL_BERT_BASE_DIR}")


if __name__ == "__main__":
    main()
