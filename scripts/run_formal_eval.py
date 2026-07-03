#!/usr/bin/env python
"""Score toxic-bert on val/test splits and write benchmark_results.json."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from encoder import ToxicBertEncoder
from evaluate import evaluate_multilabel_frame
from paths import (
    BENCHMARK_RESULTS_PATH,
    FORMAL_EVAL_QUICK_N,
    MODEL_ID,
    SEED,
    SPLIT_INDICES_PATH,
    TOXIC_LABELS,
)
from splits import load_split_indices, sample_split_frame, split_counts


def _prob_cols() -> list:
    return [f"enc_p_{l}" for l in TOXIC_LABELS]


def _pred_cols() -> list:
    return [f"enc_pred_{l}" for l in TOXIC_LABELS]


def score_split(encoder: ToxicBertEncoder, split: str, quick_n: int, threshold: float) -> dict:
    df = sample_split_frame(split, n=quick_n)
    texts = df["comment_text"].astype(str).tolist()
    print(f"Scoring {split}: {len(texts):,} comments ...")
    pred_df = encoder.predict_df(texts, threshold=threshold)
    gold = df[["comment_text"] + TOXIC_LABELS].copy()
    result = evaluate_multilabel_frame(
        gold,
        pred_df,
        prob_cols=_prob_cols(),
        pred_cols=_pred_cols(),
        threshold=threshold,
    )
    return {
        "n": result["n"],
        "threshold": threshold,
        "summary": result["summary"],
        "per_label": result["per_label"].round(4).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal toxic-bert eval on val/test splits")
    parser.add_argument("--quick", action="store_true", help=f"Cap each split at {FORMAL_EVAL_QUICK_N} rows")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not SPLIT_INDICES_PATH.exists():
        print(f"Missing {SPLIT_INDICES_PATH} — run: python scripts/make_splits.py")
        sys.exit(1)

    print("Split sizes:")
    print(split_counts().to_string(index=False))

    quick_n = FORMAL_EVAL_QUICK_N if args.quick else None
    encoder = ToxicBertEncoder()
    payload = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "quick_mode": args.quick,
        "quick_n_per_split": quick_n,
        "threshold": args.threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    for split_name in ("val", "test"):
        payload["splits"][split_name] = score_split(
            encoder, split_name, quick_n=quick_n, threshold=args.threshold
        )

    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {BENCHMARK_RESULTS_PATH}")
    for split_name, block in payload["splits"].items():
        s = block["summary"]
        print(
            f"  {split_name}: n={block['n']} macro_f1={s['macro_f1']:.3f} "
            f"micro_f1={s['micro_f1']:.3f} mean_pr_auc={s['mean_pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
