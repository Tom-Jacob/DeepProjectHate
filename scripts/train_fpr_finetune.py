#!/usr/bin/env python
"""Second-stage fine-tune of distilbert-student to reduce identity FPR.

Full run (Colab GPU recommended):
    python scripts/train_fpr_finetune.py --gamma 1.0 --epochs 2

CPU smoke:
    python scripts/train_fpr_finetune.py --gamma 1.0 --epochs 1 --max-train-samples 2000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from distill import FPRFinetuneConfig, fpr_finetune  # noqa: E402
from fairness import build_identity_probe_df  # noqa: E402
from paths import LOCAL_DISTILBERT_STUDENT_DIR, PROCESSED_DIR, TOXIC_LABELS  # noqa: E402
from splits import get_split_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FPR-regularized fine-tune of distilbert-student"
    )
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Weight on FPR penalty (higher = stronger debiasing)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-samples", type=int, default=20_000)
    parser.add_argument("--base-model", type=str,
                        default=str(LOCAL_DISTILBERT_STUDENT_DIR),
                        help="Path to base student model weights")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output dir (auto-named if empty)")
    args = parser.parse_args()

    train_df = get_split_frame("train")
    val_df = get_split_frame("val")
    train_texts = train_df["comment_text"].astype(str).tolist()
    val_texts = val_df["comment_text"].astype(str).tolist()
    train_labels = train_df[TOXIC_LABELS].values
    val_labels = val_df[TOXIC_LABELS].values

    probe_df = build_identity_probe_df()
    print(f"Train n={len(train_texts):,}  val n={len(val_texts):,}  "
          f"probe n={len(probe_df)}  gamma={args.gamma}")

    cfg = FPRFinetuneConfig(
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
    )

    meta = fpr_finetune(
        base_model_path=args.base_model,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        probe_df=probe_df,
        cfg=cfg,
    )

    results_path = PROCESSED_DIR / f"fpr_finetune_g{args.gamma}.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
