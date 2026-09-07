#!/usr/bin/env python
"""Render report figures from data/processed/model_comparison.json.

Produces two PNGs in report/figures/:
  - model_comparison_bars_<split>.png : F1 / PR-AUC per model, params annotated
  - accuracy_latency_<split>.png      : F1 vs inference latency trade-off

Run scripts/run_model_comparison.py first to create the JSON.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from plots import plot_accuracy_latency, plot_model_comparison_bars, save_figure  # noqa: E402

RESULTS_PATH = PROCESSED_DIR / "model_comparison.json"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH} — run: python scripts/run_model_comparison.py")
        sys.exit(1)

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    fig, _, df = plot_model_comparison_bars(results, split=args.split)
    p1 = save_figure(fig, FIGURES_DIR / f"model_comparison_bars_{args.split}.png")

    fig, _, _ = plot_accuracy_latency(results, split=args.split)
    p2 = save_figure(fig, FIGURES_DIR / f"accuracy_latency_{args.split}.png")

    print(df.to_string(index=False))
    print(f"\nSaved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
