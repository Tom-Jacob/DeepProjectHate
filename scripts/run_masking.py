"""Masking intervention: why do the models over-flag identities?

  A. Inference-time masking (every model in MODEL_SPECS): replace the identity
     term with [MASK] and re-score the neutral probe. A large FPR drop => the
     model relied on the identity token itself (lexical shortcut).
  B. Training-time masking (SVM): retrain with identity terms stripped, then
     measure the fairness gain (identity FPR) vs the cost (recall on genuinely
     identity-directed hate).

Usage: python scripts/run_masking.py [--train-n 20000]
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
from fairness import build_identity_probe_df  # noqa: E402
from intervention import inference_ablation, strip_terms  # noqa: E402
from model_registry import MODEL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, PROJECT_ROOT, SEED  # noqa: E402
from plots import plot_masking_ablation, plot_masking_tradeoff, save_figure  # noqa: E402
from splits import get_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "masking_results.json"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def _recall(scorer, texts, threshold=0.5) -> float:
    if not len(texts):
        return float("nan")
    preds = np.asarray(scorer.predict_toxic_proba(list(texts))) >= threshold
    return float(preds.mean())  # all rows are positive -> mean(pred)=recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-n", type=int, default=20_000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS),
                        help="Model keys to ablate (default: all of MODEL_SPECS)")
    args = parser.parse_args()

    probe = build_identity_probe_df()
    print(f"Probe: {len(probe)} neutral identity-mention sentences")

    payload = {
        "threshold": args.threshold,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inference_ablation": {},
        "svm_training_masking": {},
    }

    # ---- A. Inference-time masking on every model in MODEL_SPECS ----
    print("\n[A] Inference-time identity masking ([MASK] the identity term)")
    train_df = get_split_frame("train")
    if len(train_df) > args.train_n:
        train_df = train_df.sample(args.train_n, random_state=SEED).reset_index(drop=True)
    train_texts = train_df["comment_text"].astype(str).tolist()
    y_train = train_df["toxic"].astype(int).values

    svm = None
    for key in args.models:
        if key not in MODEL_SPECS:
            print(f"  ! unknown model {key!r}, skipping")
            continue
        spec = MODEL_SPECS[key]
        try:
            scorer = load_scorer(spec)
        except (FileNotFoundError, OSError) as exc:
            print(f"  ! {key}: skipping, weights unavailable ({exc})")
            continue
        if isinstance(scorer, TfidfSvmScorer):
            print("  training baseline SVM ...")
            scorer.fit(train_texts, y_train)
            svm = scorer
        payload["inference_ablation"][key] = inference_ablation(scorer, probe, args.threshold)

    for name, res in payload["inference_ablation"].items():
        print(f"  {name:20s} identity FPR {res['fpr_with_identity']:.3f} -> masked {res['fpr_masked']:.3f} "
              f"(drop {res['fpr_reduction']:+.3f})")

    # ---- B. Training-time masking on the SVM ----
    # Only possible if the SVM was actually part of this run; when --models
    # selects other models the baseline SVM is never fitted, and part B must be
    # skipped rather than dereferencing None. Existing results are merged back
    # in below so a partial run does not discard the full-roster numbers.
    if svm is None:
        print("\n[B] Skipped: tfidf-svm not in --models, so there is no fitted "
              "baseline SVM to compare a masked retrain against.")
    else:
        print("\n[B] Training-time masking: retrain SVM without identity terms")
        _svm_training_masking(payload, svm, train_texts, y_train, probe, args)

    _merge_and_write(payload)


def _svm_training_masking(payload, svm, train_texts, y_train, probe, args):
    svm_masked = TfidfSvmScorer().fit([strip_terms(t) for t in train_texts], y_train)

    # Fairness gain: identity FPR on the neutral probe.
    fpr_base = float((np.asarray(svm.predict_toxic_proba(probe["text"].tolist())) >= args.threshold).mean())
    fpr_masked = float((np.asarray(svm_masked.predict_toxic_proba(probe["text"].tolist())) >= args.threshold).mean())

    # Cost: recall on genuinely identity-directed hate vs on general toxicity.
    test_df = get_split_frame("test")
    ih_texts = test_df.loc[test_df["identity_hate"] == 1, "comment_text"].astype(str).tolist()
    tox_texts = test_df.loc[test_df["toxic"] == 1, "comment_text"].astype(str).tolist()

    payload["svm_training_masking"] = {
        "identity_fpr_before": fpr_base,
        "identity_fpr_after": fpr_masked,
        "identity_hate_recall_before": _recall(svm, ih_texts, args.threshold),
        "identity_hate_recall_after": _recall(svm_masked, ih_texts, args.threshold),
        "toxic_recall_before": _recall(svm, tox_texts, args.threshold),
        "toxic_recall_after": _recall(svm_masked, tox_texts, args.threshold),
        "n_identity_hate": len(ih_texts),
        "n_toxic": len(tox_texts),
    }
    b = payload["svm_training_masking"]
    print(f"  identity FPR         {b['identity_fpr_before']:.3f} -> {b['identity_fpr_after']:.3f}")
    print(f"  identity_hate recall {b['identity_hate_recall_before']:.3f} -> {b['identity_hate_recall_after']:.3f}  (cost)")
    print(f"  general toxic recall {b['toxic_recall_before']:.3f} -> {b['toxic_recall_after']:.3f}")


def _merge_and_write(payload):
    """Merge into existing results, then write JSON and regenerate figures.

    Merging matters because `--models` can select a subset: scoring one extra
    model must not discard the rows already collected for the others, and a run
    that skips part B must not wipe an existing SVM training-masking block.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            merged = dict(existing.get("inference_ablation") or {})
            merged.update(payload["inference_ablation"])
            payload["inference_ablation"] = merged
            if not payload.get("svm_training_masking"):
                payload["svm_training_masking"] = existing.get("svm_training_masking") or {}
            print(f"  merged into existing {RESULTS_PATH.name} "
                  f"({len(merged)} models in inference ablation)")
        except Exception as exc:
            print(f"  ! could not merge existing results ({exc}); overwriting")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    fig, _ = plot_masking_ablation(payload)
    print("Saved", save_figure(fig, FIGURES_DIR / "masking_inference_ablation.png"))
    if payload.get("svm_training_masking"):
        fig, _ = plot_masking_tradeoff(payload)
        print("Saved", save_figure(fig, FIGURES_DIR / "masking_svm_tradeoff.png"))


if __name__ == "__main__":
    main()
