"""Robustness: bidirectional perturbation stress-test.

1. **Evasion (toxic probe):** Take genuinely-toxic test comments each model
   flags, perturb them, and measure how many slip through as non-toxic.
2. **FP induction (non-toxic probe):** Take correctly-classified non-toxic
   comments, perturb them, and measure how many flip to toxic.

Usage:
    python scripts/run_robustness.py --quick          # 200 toxic + 200 non-toxic
    python scripts/run_robustness.py --n 1000         # 1000 each
    python scripts/run_robustness.py --n 500 --n-nontoxic 800
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
from model_registry import ALL_SPECS, MODEL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, SEED  # noqa: E402
from perturb import PERTURBATIONS, apply_perturbation  # noqa: E402
from splits import get_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "robustness_results.json"
BASELINE_TRAIN_N = 20_000


def get_fitted_scorer(key: str):
    scorer = load_scorer(ALL_SPECS[key])
    if isinstance(scorer, TfidfSvmScorer):
        train_df = get_split_frame("train")
        if len(train_df) > BASELINE_TRAIN_N:
            train_df = train_df.sample(BASELINE_TRAIN_N, random_state=SEED).reset_index(drop=True)
        print("  training TF-IDF+SVM baseline ...")
        scorer.fit(train_df["comment_text"].astype(str).tolist(), train_df["toxic"].astype(int).values)
    return scorer


def _run_toxic_probe(scorer, texts, threshold):
    """Evasion: fraction of detected-toxic comments that slip through after perturbation."""
    base_prob = scorer.predict_toxic_proba(texts)
    base_pred = base_prob >= threshold
    detected = int(base_pred.sum())
    base_rate = float(base_pred.mean())

    result = {"baseline_detection_rate": round(base_rate, 4), "n": len(texts), "perturbations": {}}
    for pname in PERTURBATIONS:
        pert_texts = [apply_perturbation(pname, t) for t in texts]
        pert_pred = scorer.predict_toxic_proba(pert_texts) >= threshold
        det_rate = float(pert_pred.mean())
        if detected > 0:
            evaded = int((base_pred & ~pert_pred).sum())
            evasion = evaded / detected
        else:
            evasion = float("nan")
        result["perturbations"][pname] = {
            "detection_rate": round(det_rate, 4),
            "evasion_rate": round(evasion, 4),
        }
        print(f"    {pname:12s} detection={det_rate:.3f}  evasion={evasion:.3f}")
    return result


def _run_nontoxic_probe(scorer, texts, threshold):
    """FP induction: fraction of correctly-classified non-toxic that flip to toxic after perturbation."""
    base_prob = scorer.predict_toxic_proba(texts)
    base_pred = base_prob >= threshold
    correct_neg = ~base_pred
    n_correct = int(correct_neg.sum())
    base_fp_rate = float(base_pred.mean())

    result = {"baseline_fp_rate": round(base_fp_rate, 4), "n": len(texts),
              "n_correct_neg": n_correct, "perturbations": {}}
    for pname in PERTURBATIONS:
        pert_texts = [apply_perturbation(pname, t) for t in texts]
        pert_pred = scorer.predict_toxic_proba(pert_texts) >= threshold
        if n_correct > 0:
            induced = int((correct_neg & pert_pred).sum())
            fp_induction = induced / n_correct
        else:
            fp_induction = float("nan")
        result["perturbations"][pname] = {
            "fp_induction_rate": round(fp_induction, 4),
        }
        print(f"    {pname:12s} FP-induction={fp_induction:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--n", type=int, default=300, help="Number of toxic comments to probe")
    parser.add_argument("--n-nontoxic", type=int, default=None,
                        help="Number of non-toxic comments (default: same as --n)")
    parser.add_argument("--quick", action="store_true", help="Use 200 comments")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    n_toxic = 200 if args.quick else args.n
    n_nontoxic = args.n_nontoxic if args.n_nontoxic is not None else n_toxic

    test = get_split_frame("test")
    toxic = test[test["toxic"] == 1].sample(
        min(n_toxic, int((test["toxic"] == 1).sum())), random_state=SEED
    ).reset_index(drop=True)
    nontoxic = test[test["toxic"] == 0].sample(
        min(n_nontoxic, int((test["toxic"] == 0).sum())), random_state=SEED
    ).reset_index(drop=True)

    toxic_texts = toxic["comment_text"].astype(str).tolist()
    nontoxic_texts = nontoxic["comment_text"].astype(str).tolist()
    print(f"Probing {len(toxic_texts)} toxic + {len(nontoxic_texts)} non-toxic comments "
          f"with {len(PERTURBATIONS)} perturbations")

    payload = {
        "threshold": args.threshold,
        "n_toxic": len(toxic_texts),
        "n_nontoxic": len(nontoxic_texts),
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "perturbations": list(PERTURBATIONS),
        "models": {},
    }

    for key in args.models:
        if key not in ALL_SPECS:
            print(f"  ! unknown model {key!r}, skipping")
            continue
        print(f"\n=== {key} ===")
        scorer = get_fitted_scorer(key)

        print("  [toxic probe — evasion]")
        toxic_result = _run_toxic_probe(scorer, toxic_texts, args.threshold)

        print("  [non-toxic probe — FP induction]")
        nontoxic_result = _run_nontoxic_probe(scorer, nontoxic_texts, args.threshold)

        payload["models"][key] = {
            "toxic_probe": toxic_result,
            "nontoxic_probe": nontoxic_result,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Merge into existing results rather than overwriting: scoring one extra
    # model must not wipe the rows already collected for the others. Note the
    # top-level n_toxic/n_nontoxic describe THIS run; each model block records
    # the sample size it was actually measured at.
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if isinstance(existing.get("models"), dict):
                merged = dict(existing["models"])
                merged.update(payload["models"])
                payload["models"] = merged
                print(f"  merged into existing {RESULTS_PATH.name} ({len(merged)} models)")
        except Exception as exc:
            print(f"  ! could not merge existing results ({exc}); overwriting")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
