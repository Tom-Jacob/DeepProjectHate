#!/usr/bin/env python
"""Fine-tune DistilGPT-2 on D1 yes/no answers for binary toxic.

Colab (recommended):
    python scripts/train_decoder_ft.py --epochs 2 --max-train-samples 8000

CPU smoke:
    python scripts/train_decoder_ft.py --max-train-samples 400 --max-val-samples 100 --epochs 1 --batch-size 2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoder_finetune import DecoderFTConfig, train_decoder_ft  # noqa: E402
from paths import (  # noqa: E402
    DECODER_FT_RESULTS_PATH,
    LOCAL_DISTILGPT2_D1_FT_DIR,
    PROCESSED_DIR,
)
from splits import get_split_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT-2 D1 yes/no (binary toxic)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=str(LOCAL_DISTILGPT2_D1_FT_DIR))
    args = parser.parse_args()

    train_df = get_split_frame("train")
    val_df = get_split_frame("val")
    train_texts = train_df["comment_text"].astype(str).tolist()
    val_texts = val_df["comment_text"].astype(str).tolist()
    train_labels = train_df["toxic"].astype(int).tolist()
    val_labels = val_df["toxic"].astype(int).tolist()

    cfg = DecoderFTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        output_dir=args.output_dir,
    )
    print(
        f"Decoder FT: train_pool={len(train_texts):,} val_pool={len(val_texts):,} "
        f"caps train={cfg.max_train_samples} val={cfg.max_val_samples}"
    )
    meta = train_decoder_ft(train_texts, train_labels, val_texts, val_labels, cfg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECODER_FT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved FT decoder → {cfg.output_dir}")
    print(f"Wrote {DECODER_FT_RESULTS_PATH}")
    print(f"best_val_answer_acc={meta['best_val_answer_acc']:.3f}")


if __name__ == "__main__":
    main()
