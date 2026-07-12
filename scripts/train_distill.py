#!/usr/bin/env python
"""Train DistilBERT student via soft KD from toxic-bert (6 labels).

Full training (Colab GPU recommended):
    python scripts/train_distill.py --epochs 2

CPU smoke (directional only):
    python scripts/train_distill.py --max-train-samples 2000 --max-val-samples 500 --epochs 1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from distill import DistillConfig, train_distill  # noqa: E402
from paths import (  # noqa: E402
    DISTILL_RESULTS_PATH,
    LOCAL_DISTILBERT_STUDENT_DIR,
    PROCESSED_DIR,
    TOXIC_LABELS,
)
from splits import get_split_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft KD: toxic-bert → DistilBERT (6-label)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DISTILBERT_STUDENT_DIR))
    args = parser.parse_args()

    train_df = get_split_frame("train")
    val_df = get_split_frame("val")
    train_texts = train_df["comment_text"].astype(str).tolist()
    val_texts = val_df["comment_text"].astype(str).tolist()
    train_labels = train_df[TOXIC_LABELS].values
    val_labels = val_df[TOXIC_LABELS].values

    cfg = DistillConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        output_dir=args.output_dir,
    )
    print(
        f"Train n={len(train_texts):,} val n={len(val_texts):,} "
        f"(caps train={cfg.max_train_samples} val={cfg.max_val_samples})"
    )
    meta = train_distill(train_texts, train_labels, val_texts, val_labels, cfg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DISTILL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved student → {cfg.output_dir}")
    print(f"Wrote {DISTILL_RESULTS_PATH}")
    print(f"best_val_macro_f1={meta['best_val_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
