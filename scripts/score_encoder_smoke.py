#!/usr/bin/env python
"""Score encoder on the same comments as decoder_smoke_scores.parquet (or balanced sample)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_train_df
from encoder import ToxicBertEncoder
from eval_sample import get_balanced_sample
from paths import (
    DECODER_SMOKE_SCORES_PATH,
    ENCODER_SMOKE_SCORES_PATH,
    MODEL_ID,
    QUICK_MODE,
    TOXIC_LABELS,
)


def _texts_from_decoder_parquet() -> list:
    import pandas as pd

    dec = pd.read_parquet(DECODER_SMOKE_SCORES_PATH)
    return dec["comment_text"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write encoder_smoke_scores.parquet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--from-decoder",
        action="store_true",
        help="Use comment_text from decoder_smoke_scores.parquet (default if file exists)",
    )
    args = parser.parse_args()

    if args.from_decoder or DECODER_SMOKE_SCORES_PATH.exists():
        if not DECODER_SMOKE_SCORES_PATH.exists():
            print(f"Missing {DECODER_SMOKE_SCORES_PATH} — run notebook 02 or omit --from-decoder")
            sys.exit(1)
        texts = _texts_from_decoder_parquet()
        gold = None
    else:
        sample = get_balanced_sample(load_train_df(), quick_mode=QUICK_MODE)
        texts = sample["comment_text"].astype(str).tolist()
        gold = sample[["comment_text"] + TOXIC_LABELS]

    print(f"Scoring {len(texts)} comments with toxic-bert ...")
    encoder = ToxicBertEncoder()
    out = encoder.predict_df(texts, threshold=args.threshold, batch_size=args.batch_size)
    out["model_id"] = MODEL_ID
    out["threshold"] = args.threshold
    out["quick_mode"] = QUICK_MODE

    if gold is not None:
        out = gold.merge(out, on="comment_text", how="left")

    ENCODER_SMOKE_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ENCODER_SMOKE_SCORES_PATH, index=False)
    print(f"Wrote {len(out)} rows to {ENCODER_SMOKE_SCORES_PATH}")


if __name__ == "__main__":
    main()
