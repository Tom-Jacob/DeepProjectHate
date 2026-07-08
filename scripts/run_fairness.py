#!/usr/bin/env python
"""Per-identity false-positive-rate (FPR) fairness probe across models.

Scores neutral identity-mention sentences (all non-toxic) with every model and
reports how often each identity group is wrongly flagged as toxic.

Usage:
    python scripts/run_fairness.py
    python scripts/run_fairness.py --models toxic-bert roberta-tox tfidf-svm
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import TfidfSvmScorer  # noqa: E402
from fairness import (  # noqa: E402
    CONTROL_TEXTS,
    build_identity_probe_df,
    control_fpr,
    per_group_fpr,
)
from model_registry import MODEL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, SEED, TOXIC_LABELS  # noqa: E402
from splits import get_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "fairness_results.json"
BASELINE_TRAIN_N = 20_000


def get_fitted_scorer(key: str):
    """Load a scorer; train the non-neural baseline on the train split."""
    scorer = load_scorer(MODEL_SPECS[key])
    if isinstance(scorer, TfidfSvmScorer):
        train_df = get_split_frame("train")
        if len(train_df) > BASELINE_TRAIN_N:
            train_df = train_df.sample(BASELINE_TRAIN_N, random_state=SEED).reset_index(drop=True)
        y = (train_df["toxic"]).astype(int).values
        print("  training TF-IDF+SVM baseline ...")
        scorer.fit(train_df["comment_text"].astype(str).tolist(), y)
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    probe = build_identity_probe_df()
    print(f"Probe: {len(probe)} neutral sentences over {probe['group'].nunique()} identity groups")

    payload = {
        "threshold": args.threshold,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_probe": int(len(probe)),
        "models": {},
    }

    for key in args.models:
        if key not in MODEL_SPECS:
            print(f"  ! unknown model {key!r}, skipping")
            continue
        print(f"\n=== {key} ===")
        scorer = get_fitted_scorer(key)
        probs = scorer.predict_toxic_proba(probe["text"].tolist())
        ctrl = scorer.predict_toxic_proba(CONTROL_TEXTS)

        group_df = per_group_fpr(probe, probs, threshold=args.threshold)
        ctrl_stats = control_fpr(ctrl, threshold=args.threshold)
        overall_fpr = float(group_df["fpr"].mean())

        payload["models"][key] = {
            "model_id": MODEL_SPECS[key].model_id,
            "family": MODEL_SPECS[key].family,
            "overall_fpr": overall_fpr,
            "control_fpr": ctrl_stats["fpr"],
            "per_group": group_df.round(4).to_dict(orient="records"),
        }
        worst = group_df.sort_values("fpr", ascending=False).head(3)
        print(f"  overall identity FPR={overall_fpr:.3f} | control FPR={ctrl_stats['fpr']:.3f}")
        print("  worst groups: " + ", ".join(f"{r.group}={r.fpr:.2f}" for r in worst.itertuples()))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
