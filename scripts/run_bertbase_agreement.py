#!/usr/bin/env python
"""Same-model, two-methods: bert-base-classifier vs. bert-base-mlm-prompt.

Scores the same test sample with the SAME pretrained checkpoint used two ways
-- a trained classification head vs. zero-shot masked-LM prompting -- and
reports binary agreement, accuracy against gold, and disagreement examples.
This is the controlled classifier-vs-prompting comparison (see model_registry
docstring); run_model_comparison.py/run_fairness.py/run_robustness.py already
cover accuracy/fairness/robustness for both models, this adds the pairwise
agreement view (parallel to run_agreement.py's encoder-vs-decoder analysis).

Usage: python scripts/run_bertbase_agreement.py --n 800
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import encoder_decoder_agreement  # noqa: E402
from model_registry import ALL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, SEED  # noqa: E402
from splits import sample_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "bertbase_agreement_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    sample = sample_split_frame(args.split, n=args.n).drop_duplicates("comment_text").reset_index(drop=True)
    texts = sample["comment_text"].astype(str).tolist()
    gold = sample["toxic"].astype(int).values
    print(f"Sample: {len(texts)} comments from {args.split}")

    print("Scoring bert-base-classifier (trained head) ...")
    clf = load_scorer(ALL_SPECS["bert-base-classifier"])
    p_clf = np.asarray(clf.predict_toxic_proba(texts))

    print("Scoring bert-base-mlm-prompt (zero-shot fill-mask) ...")
    prompt = load_scorer(ALL_SPECS["bert-base-mlm-prompt"])
    p_prompt = np.asarray(prompt.predict_toxic_proba(texts))

    pred_clf = (p_clf >= args.threshold).astype(int)
    pred_prompt = (p_prompt >= args.threshold).astype(int)

    joined = sample[["comment_text", "toxic"]].copy()
    joined["pred_clf"] = pred_clf
    joined["pred_prompt"] = pred_prompt
    agree = encoder_decoder_agreement(
        joined, label="toxic", enc_pred_col="pred_clf", dec_pred_col="pred_prompt"
    )

    acc_clf = float((pred_clf == gold).mean())
    acc_prompt = float((pred_prompt == gold).mean())

    dis = joined[joined["pred_clf"] != joined["pred_prompt"]]
    cases = dis[["comment_text", "toxic", "pred_clf", "pred_prompt"]].head(12)

    payload = {
        "n": int(len(joined)),
        "split": args.split,
        "seed": SEED,
        "threshold": args.threshold,
        "agreement": agree["agreement"],
        "classifier_vs_gold_acc": acc_clf,
        "prompt_vs_gold_acc": acc_prompt,
        "n_disagree": int(len(dis)),
        "disagreement_cases": cases.to_dict(orient="records"),
    }

    print(f"\nbert-base-classifier vs bert-base-mlm-prompt agreement: {agree['agreement']:.3f}")
    print(f"vs gold: classifier={acc_clf:.3f} prompt={acc_prompt:.3f}")
    print(f"Disagreements: {len(dis)}/{len(joined)}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
