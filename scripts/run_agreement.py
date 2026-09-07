"""P7 integration: encoder vs decoder agreement + disagreement case studies.

Scores the same test sample with toxic-bert (6-label) and Tom's DistilGPT-2 D1
prompting (6-label), then measures how often the two architectures agree, how
well each matches gold, and surfaces comments where they disagree (the gold of
qualitative analysis). Reuses Tom's evaluate.py / plots.py unchanged.

DistilGPT-2 is slow on CPU (per label, per comment) — keep --n small.

Usage: python scripts/run_agreement.py --n 150
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoder import predict_multilabel_d1  # noqa: E402
from encoder import load_toxic_bert  # noqa: E402
from evaluate import agreement_by_label  # noqa: E402
from model_registry import ALL_SPECS, load_scorer  # noqa: E402
from paths import PROCESSED_DIR, PROJECT_ROOT, SEED, TOXIC_LABELS  # noqa: E402
from plots import plot_agreement_by_label, plot_gold_agreement_panel, save_figure  # noqa: E402
from splits import sample_split_frame  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "agreement_results.json"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    sample = sample_split_frame(args.split, n=args.n).drop_duplicates("comment_text").reset_index(drop=True)
    texts = sample["comment_text"].astype(str).tolist()
    print(f"Sample: {len(texts)} comments from {args.split}")

    print("Scoring encoder (toxic-bert, 6-label) ...")
    enc = load_toxic_bert()
    enc_df = enc.predict_df(texts)  # comment_text, enc_p_{label}, enc_pred_{label}

    print("Scoring decoder (DistilGPT-2 D1, 6-label) — slow on CPU ...")
    dec = load_scorer(ALL_SPECS["distilgpt2-d1"])
    trunc = [dec._truncate(t) for t in texts]  # keep prompt under DistilGPT-2's 1024 ctx
    dec_df = predict_multilabel_d1(dec.model, dec.tokenizer, trunc, dec.device)
    dec_df["comment_text"] = texts  # restore original text so the join matches

    gold = sample[["comment_text"] + TOXIC_LABELS]
    joined = gold.merge(enc_df, on="comment_text").merge(dec_df, on="comment_text")

    # Encoder vs decoder agreement (model-vs-model), per label.
    agree = agreement_by_label(joined)
    # Each method vs gold (accuracy), per label.
    gold_rows = []
    for label in TOXIC_LABELS:
        g = joined[label]
        gold_rows.append({
            "label": label,
            "encoder_vs_gold": float((g == joined[f"enc_pred_{label}"]).mean()),
            "decoder_vs_gold": float((g == joined[f"d1_pred_logprob_{label}"]).mean()),
        })

    payload = {
        "n": int(len(joined)),
        "split": args.split,
        "seed": SEED,
        "overall_enc_dec_agreement": float(agree["agreement"].mean()),
        "agreement_by_label": agree.round(4).to_dict(orient="records"),
        "vs_gold": gold_rows,
    }

    # Disagreement case studies on 'toxic' (where the architectures differ).
    dis = joined[joined["enc_pred_toxic"] != joined["d1_pred_logprob_toxic"]]
    cases = dis[["comment_text", "toxic", "enc_pred_toxic", "d1_pred_logprob_toxic"]].head(12)
    payload["n_disagree_toxic"] = int(len(dis))
    payload["disagreement_cases"] = cases.to_dict(orient="records")

    print(f"\nOverall enc-dec agreement: {payload['overall_enc_dec_agreement']:.3f}")
    print(f"Disagreements on 'toxic': {len(dis)}/{len(joined)}")
    print("vs gold (toxic):",
          f"encoder={gold_rows[0]['encoder_vs_gold']:.3f}",
          f"decoder={gold_rows[0]['decoder_vs_gold']:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")

    fig, _, _ = plot_gold_agreement_panel(joined)
    print("Saved", save_figure(fig, FIGURES_DIR / "enc_dec_gold_agreement.png"))
    ax, _ = plot_agreement_by_label(joined)
    print("Saved", save_figure(ax.get_figure(), FIGURES_DIR / "enc_dec_agreement.png"))


if __name__ == "__main__":
    main()
