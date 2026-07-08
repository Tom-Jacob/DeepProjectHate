"""Comparison plots for encoder vs decoder smoke / formal eval."""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from evaluate import agreement_by_label, evaluate_multilabel_frame
from paths import TOXIC_LABELS


def plot_agreement_by_label(
    joined: pd.DataFrame,
    labels: Optional[Sequence[str]] = None,
    dec_pred_tpl: str = "d1_pred_logprob_{}",
    ax=None,
    title: str = "Encoder vs D1 logprob agreement",
):
    labels = list(labels or TOXIC_LABELS)
    agree_df = agreement_by_label(joined, labels=labels, dec_pred_tpl=dec_pred_tpl)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=agree_df, x="label", y="agreement", ax=ax, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Agreement rate")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    return ax, agree_df


def plot_gold_agreement_panel(
    joined: pd.DataFrame,
    enc_pred_tpl: str = "enc_pred_{}",
    dec_pred_tpl: str = "d1_pred_logprob_{}",
    labels: Optional[Sequence[str]] = None,
    figsize=(10, 4),
):
    """Side-by-side bar chart: each method vs gold by label."""
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for label in labels:
        gold = joined[label]
        enc = joined[enc_pred_tpl.format(label)]
        dec = joined[dec_pred_tpl.format(label)]
        rows.append({"label": label, "method": "encoder", "agreement": (gold == enc).mean()})
        rows.append({"label": label, "method": "decoder D1", "agreement": (gold == dec).mean()})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=plot_df, x="label", y="agreement", hue="method", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Agreement with gold labels")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    fig.tight_layout()
    return fig, ax, plot_df


def plot_prob_scatter(
    joined: pd.DataFrame,
    label: str = "toxic",
    enc_prob_col: Optional[str] = None,
    dec_prob_col: Optional[str] = None,
    ax=None,
):
    enc_prob_col = enc_prob_col or f"enc_p_{label}"
    dec_prob_col = dec_prob_col or f"d1_p_yes_{label}"
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(joined[enc_prob_col], joined[dec_prob_col], alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel(f"Encoder P({label})")
    ax.set_ylabel(f"Decoder D1 P(yes) — {label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Probability scatter — {label}")
    return ax


def plot_macro_f1_comparison(
    enc_metrics: dict,
    dec_metrics: dict,
    split_name: str = "eval",
    figsize=(6, 4),
):
    """Bar chart of macro/micro F1 from evaluate_multilabel_frame summaries."""
    rows = [
        {"method": "encoder", "metric": "macro_f1", "value": enc_metrics["summary"]["macro_f1"]},
        {"method": "encoder", "metric": "micro_f1", "value": enc_metrics["summary"]["micro_f1"]},
        {"method": "decoder D1", "metric": "macro_f1", "value": dec_metrics["summary"]["macro_f1"]},
        {"method": "decoder D1", "metric": "micro_f1", "value": dec_metrics["summary"]["micro_f1"]},
    ]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x="metric", y="value", hue="method", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"F1 comparison ({split_name})")
    ax.legend(title="")
    fig.tight_layout()
    return fig, ax, df


def metrics_from_joined(
    joined: pd.DataFrame,
    method: str = "encoder",
    threshold: float = 0.5,
) -> dict:
    gold = joined[["comment_text"] + TOXIC_LABELS]
    if method == "encoder":
        prob_cols = [f"enc_p_{l}" for l in TOXIC_LABELS]
        pred_cols = [f"enc_pred_{l}" for l in TOXIC_LABELS]
    elif method == "decoder":
        prob_cols = [f"d1_p_yes_{l}" for l in TOXIC_LABELS]
        pred_cols = [f"d1_pred_logprob_{l}" for l in TOXIC_LABELS]
    else:
        raise ValueError(f"unknown method: {method}")
    return evaluate_multilabel_frame(
        gold,
        joined,
        prob_cols=prob_cols,
        pred_cols=pred_cols,
        threshold=threshold,
    )


def comparison_frame(results: dict, split: str = "test") -> pd.DataFrame:
    """Flatten model_comparison.json into a tidy DataFrame for one split."""
    rows = []
    for key, block in results["models"].items():
        m = block["splits"].get(split, {})
        rows.append(
            {
                "model": key,
                "family": block.get("family"),
                "params": block.get("n_params"),
                "f1": m.get("f1"),
                "pr_auc": m.get("pr_auc"),
                "roc_auc": m.get("roc_auc"),
                "ms_per_comment": m.get("ms_per_comment"),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def plot_model_comparison_bars(results: dict, split: str = "test", figsize=(9, 5)):
    """Grouped F1 / PR-AUC bars per model, annotated with parameter counts.

    This is the fair-comparison figure: accuracy shown next to model capacity so
    'bigger' is never conflated with 'better'.
    """
    df = comparison_frame(results, split=split)
    long = df.melt(
        id_vars=["model", "params"],
        value_vars=["f1", "pr_auc"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=long, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.set_title(f"Binary toxic detection — {split} (bars) with parameter counts")
    ax.legend(title="")

    # Annotate each model with its parameter/capacity count under the axis.
    def _fmt_params(p):
        if p is None:
            return "?"
        return f"{p/1e6:.0f}M" if p >= 1e6 else f"{p/1e3:.0f}k"

    labels = [f"{row.model}\n({_fmt_params(row.params)})" for row in df.itertuples()]
    ax.set_xticklabels(labels)
    fig.tight_layout()
    return fig, ax, df


def plot_accuracy_latency(results: dict, split: str = "test", figsize=(7, 5)):
    """Scatter of F1 vs inference latency (log-x); the efficiency trade-off."""
    df = comparison_frame(results, split=split)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df["ms_per_comment"], df["f1"], s=90, color="steelblue", zorder=3)
    for row in df.itertuples():
        ax.annotate(
            row.model,
            (row.ms_per_comment, row.f1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("inference latency (ms/comment, log scale)")
    ax.set_ylabel(f"{split} F1")
    ax.set_title("Accuracy vs. inference cost")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax, df


def save_figure(fig, path: Path, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
