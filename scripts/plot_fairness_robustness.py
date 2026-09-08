#!/usr/bin/env python
"""Render fairness + robustness figures into report/figures/.

Run scripts/run_fairness.py and scripts/run_robustness.py first.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from plots import plot_fairness_heatmap, plot_robustness_heatmap, save_figure  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def _load(path: Path):
    if not path.exists():
        print(f"Missing {path} — run the matching script first.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    fair = _load(PROCESSED_DIR / "fairness_results.json")
    if fair:
        fig, _, _ = plot_fairness_heatmap(fair)
        print("Saved", save_figure(fig, FIGURES_DIR / "fairness_fpr_heatmap.png"))

    robust = _load(PROCESSED_DIR / "robustness_results.json")
    if robust:
        fig, _, _ = plot_robustness_heatmap(robust, value="evasion_rate", probe="toxic_probe")
        print("Saved", save_figure(fig, FIGURES_DIR / "robustness_evasion_heatmap.png"))
        fig, _, _ = plot_robustness_heatmap(robust, value="fp_induction_rate", probe="nontoxic_probe")
        print("Saved", save_figure(fig, FIGURES_DIR / "robustness_fp_induction_heatmap.png"))


if __name__ == "__main__":
    main()
