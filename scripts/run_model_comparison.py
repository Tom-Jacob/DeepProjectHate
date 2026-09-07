#!/usr/bin/env python
"""Fair binary-toxicity comparison across encoder models + non-neural baseline.

Every model is reduced to a single P(toxic) and evaluated on the SAME binary
task (Jigsaw ``toxic`` label by default). We log accuracy metrics AND the
fair-comparison inputs the supervisor asks for: parameter count and inference
latency per model.

Usage:
    python scripts/run_model_comparison.py --quick        # fast: small train + 500/split
    python scripts/run_model_comparison.py                # full train + full val/test
    python scripts/run_model_comparison.py --models toxic-bert tfidf-svm
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import TfidfSvmScorer  # noqa: E402
from model_registry import ALL_SPECS, MODEL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, SEED, TOXIC_LABELS  # noqa: E402
from splits import get_split_frame, sample_split_frame, split_counts  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "model_comparison.json"

# Caps used when --quick is passed.
QUICK_TRAIN_N = 20_000
QUICK_EVAL_N = 500


def binary_gold(df: pd.DataFrame, target: str) -> np.ndarray:
    """Binary toxic label from a split frame."""
    if target == "toxic_any":
        return (df[TOXIC_LABELS].sum(axis=1) > 0).astype(int).values
    return df[target].astype(int).values


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    # AUC metrics need both classes present.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def score_with_latency(scorer, texts, batch_size):
    """Return (probs, ms_per_comment)."""
    t0 = time.perf_counter()
    probs = scorer.predict_toxic_proba(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    ms_per = 1000.0 * elapsed / max(len(texts), 1)
    return np.asarray(probs), ms_per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small train sample + 500 rows/split")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS),
                        help=f"Model keys. Choices: {list(ALL_SPECS)}")
    parser.add_argument("--target", choices=["toxic", "toxic_any"], default="toxic")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-n", type=int, default=None,
                        help="Cap rows/split (overrides --quick); useful for the slow decoder")
    args = parser.parse_args()

    if args.eval_n is not None:
        eval_n = args.eval_n
    else:
        eval_n = QUICK_EVAL_N if args.quick else None
    print("Split sizes:")
    print(split_counts().to_string(index=False))

    # Fixed eval frames (same rows for every model -> fair comparison).
    val_df = sample_split_frame("val", n=eval_n)
    test_df = sample_split_frame("test", n=eval_n)
    eval_frames = {"val": val_df, "test": test_df}
    gold = {name: binary_gold(df, args.target) for name, df in eval_frames.items()}
    print(f"\nTarget label: {args.target} | val n={len(val_df)} test n={len(test_df)}")
    for name, y in gold.items():
        print(f"  {name}: positives = {int(y.sum())}/{len(y)} ({100*y.mean():.1f}%)")

    payload = {
        "target": args.target,
        "threshold": args.threshold,
        "quick_mode": args.quick,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_n": {name: int(len(df)) for name, df in eval_frames.items()},
        "models": {},
    }

    for key in args.models:
        if key not in ALL_SPECS:
            print(f"  ! unknown model key {key!r}, skipping")
            continue
        spec = ALL_SPECS[key]
        print(f"\n=== {key} ({spec.model_id}) ===")
        scorer = load_scorer(spec)

        # Train the non-neural baseline on the train split.
        if isinstance(scorer, TfidfSvmScorer):
            train_df = get_split_frame("train")
            if args.quick and len(train_df) > QUICK_TRAIN_N:
                train_df = train_df.sample(QUICK_TRAIN_N, random_state=SEED).reset_index(drop=True)
            y_train = binary_gold(train_df, args.target)
            print(f"  training TF-IDF+SVM on {len(train_df):,} comments ...")
            t0 = time.perf_counter()
            scorer.fit(train_df["comment_text"].astype(str).tolist(), y_train)
            print(f"  trained in {time.perf_counter()-t0:.1f}s | vocab={scorer.n_params:,}")

        model_block = {
            "model_id": spec.model_id,
            "family": spec.family,
            "specialization": spec.specialization,
            "n_params": int(scorer.n_params) if scorer.n_params is not None else None,
            "splits": {},
        }
        for name, df in eval_frames.items():
            texts = df["comment_text"].astype(str).tolist()
            probs, ms_per = score_with_latency(scorer, texts, args.batch_size)
            metrics = binary_metrics(gold[name], probs, args.threshold)
            metrics["ms_per_comment"] = round(ms_per, 3)
            model_block["splits"][name] = metrics
            print(
                f"  {name}: f1={metrics['f1']:.3f} pr_auc={metrics['pr_auc']:.3f} "
                f"roc_auc={metrics['roc_auc']:.3f} ({ms_per:.1f} ms/comment)"
            )
        payload["models"][key] = model_block

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    # Compact leaderboard (test split).
    rows = []
    for key, block in payload["models"].items():
        t = block["splits"].get("test", {})
        rows.append(
            {
                "model": key,
                "family": block["family"],
                "params": block["n_params"],
                "test_f1": round(t.get("f1", float("nan")), 3),
                "test_pr_auc": round(t.get("pr_auc", float("nan")), 3),
                "ms/comment": t.get("ms_per_comment"),
            }
        )
    if rows:
        print("\nLeaderboard (test):")
        print(pd.DataFrame(rows).sort_values("test_f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
