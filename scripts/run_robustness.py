"""Robustness: how easily do surface perturbations evade toxic detection?

Take genuinely-toxic test comments each model flags, perturb them (typos,
leetspeak, spacing, repetition, casing, punctuation), and measure how many slip
through as non-toxic. Lower evasion = more robust.

Usage:
    python scripts/run_robustness.py --quick     # 200 toxic comments
    python scripts/run_robustness.py --n 500
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import TfidfSvmScorer  # noqa: E402
from model_registry import MODEL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, SEED  # noqa: E402
from perturb import PERTURBATIONS, apply_perturbation  # noqa: E402
from splits import get_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "robustness_results.json"
BASELINE_TRAIN_N = 20_000


def get_fitted_scorer(key: str):
    scorer = load_scorer(MODEL_SPECS[key])
    if isinstance(scorer, TfidfSvmScorer):
        train_df = get_split_frame("train")
        if len(train_df) > BASELINE_TRAIN_N:
            train_df = train_df.sample(BASELINE_TRAIN_N, random_state=SEED).reset_index(drop=True)
        print("  training TF-IDF+SVM baseline ...")
        scorer.fit(train_df["comment_text"].astype(str).tolist(), train_df["toxic"].astype(int).values)
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--n", type=int, default=300, help="Number of toxic comments to probe")
    parser.add_argument("--quick", action="store_true", help="Use 200 comments")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    n = 200 if args.quick else args.n
    test = get_split_frame("test")
    toxic = test[test["toxic"] == 1].sample(
        min(n, int((test["toxic"] == 1).sum())), random_state=SEED
    ).reset_index(drop=True)
    texts = toxic["comment_text"].astype(str).tolist()
    print(f"Probing {len(texts)} genuinely-toxic comments with {len(PERTURBATIONS)} perturbations")

    payload = {
        "threshold": args.threshold,
        "n_toxic": len(texts),
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "perturbations": list(PERTURBATIONS),
        "models": {},
    }

    for key in args.models:
        if key not in MODEL_SPECS:
            print(f"  ! unknown model {key!r}, skipping")
            continue
        print(f"\n=== {key} ===")
        scorer = get_fitted_scorer(key)

        base_prob = scorer.predict_toxic_proba(texts)
        base_pred = base_prob >= args.threshold
        detected = base_pred.sum()
        base_rate = float(base_pred.mean())
        print(f"  baseline detection rate on toxic set: {base_rate:.3f} ({detected}/{len(texts)})")

        block = {"baseline_detection_rate": base_rate, "perturbations": {}}
        for pname in PERTURBATIONS:
            pert_texts = [apply_perturbation(pname, t) for t in texts]
            pert_pred = scorer.predict_toxic_proba(pert_texts) >= args.threshold
            det_rate = float(pert_pred.mean())
            # Evasion among comments detected at baseline.
            if detected > 0:
                evaded = int((base_pred & ~pert_pred).sum())
                evasion = evaded / int(detected)
            else:
                evasion = float("nan")
            block["perturbations"][pname] = {
                "detection_rate": round(det_rate, 4),
                "evasion_rate": round(evasion, 4),
            }
            print(f"    {pname:12s} detection={det_rate:.3f}  evasion={evasion:.3f}")
        payload["models"][key] = block

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
