"""Comparison plots for encoder vs decoder smoke / formal eval."""
import copy
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


def fairness_matrix(results: dict, value: str = "fpr") -> pd.DataFrame:
    """Pivot fairness_results.json into a group x model matrix (grouped by category)."""
    rows = []
    for model, block in results["models"].items():
        for rec in block["per_group"]:
            rows.append(
                {"model": model, "group": rec["group"], "category": rec["category"], value: rec[value]}
            )
    long = pd.DataFrame(rows)
    order = (
        long[["category", "group"]]
        .drop_duplicates()
        .sort_values(["category", "group"])["group"]
        .tolist()
    )
    mat = long.pivot(index="group", columns="model", values=value).reindex(order)
    return mat


def plot_fairness_heatmap(results: dict, value: str = "fpr", figsize=(7, 9)):
    """Heatmap of per-identity FPR (rows = identity groups, cols = models)."""
    mat = fairness_matrix(results, value=value)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
        cbar_kws={"label": value}, ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_title("Per-identity false-positive rate (neutral sentences)")
    ax.set_xlabel("")
    ax.set_ylabel("identity group")
    fig.tight_layout()
    return fig, ax, mat


def robustness_matrix(
    results: dict, value: str = "evasion_rate", probe: str = "toxic_probe"
) -> pd.DataFrame:
    """Pivot robustness_results.json into a perturbation x model matrix.

    Supports both the current bidirectional schema (``block[probe]["perturbations"]``,
    ``probe`` = "toxic_probe" for evasion_rate or "nontoxic_probe" for
    fp_induction_rate) and the older flat schema (``block["perturbations"]``)
    for backward compatibility with any pre-existing results JSON.
    """
    rows = []
    for model, block in results["models"].items():
        sub = block.get(probe, block)
        perturbations = sub.get("perturbations", {})
        for pname, m in perturbations.items():
            if value in m:
                rows.append({"model": model, "perturbation": pname, value: m[value]})
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()
    return long.pivot(index="perturbation", columns="model", values=value)


def plot_robustness_heatmap(
    results: dict,
    value: str = "evasion_rate",
    probe: str = "toxic_probe",
    figsize=(7, 5),
):
    """Heatmap of evasion/FP-induction rate (rows = perturbations, cols = models).

    Higher = more brittle. ``value="evasion_rate", probe="toxic_probe"`` (default)
    is the toxic-comment evasion direction; ``value="fp_induction_rate",
    probe="nontoxic_probe"`` is the non-toxic false-positive-induction direction.
    """
    mat = robustness_matrix(results, value=value, probe=probe)
    fig, ax = plt.subplots(figsize=figsize)

    # A model that detects nothing at the threshold has an undefined evasion
    # rate (0/0 -> NaN). Left alone those cells render blank and read as a
    # missing result, so mark them explicitly: grey fill, "n/a" annotation.
    cmap = copy.copy(plt.get_cmap("Oranges"))
    cmap.set_bad("#d9d9d9")

    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1,
        cbar_kws={"label": value}, ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 8},
    )

    # seaborn skips annotations on NaN cells, so write "n/a" into them by hand.
    for xi, col in enumerate(mat.columns):
        for yi, idx in enumerate(mat.index):
            if pd.isna(mat.loc[idx, col]):
                ax.text(
                    xi + 0.5, yi + 0.5, "n/a",
                    ha="center", va="center", fontsize=8,
                    color="#6f6f6f", style="italic",
                )

    # Grey out the label of any fully-undefined model so the axis matches.
    undefined = [c for c in mat.columns if mat[c].isna().all()]
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in undefined:
            lbl.set_color("#8c8c8c")
            lbl.set_style("italic")

    title = "Evasion rate" if probe == "toxic_probe" else "FP-induction rate"
    ax.set_title(f"{title} under text perturbations (higher = more brittle)")
    ax.set_xlabel("")
    ax.set_ylabel("perturbation")
    fig.tight_layout()
    if undefined:
        # Placed on the figure (not the axes) and after tight_layout, so it sits
        # below the rotated tick labels instead of colliding with them.
        fig.subplots_adjust(bottom=0.42)
        fig.text(
            0.01, 0.02,
            "n/a = undefined: this model detects nothing at threshold 0.5, "
            "so the evasion denominator is zero",
            fontsize=7.5, color="#5a5a5a",
        )
    return fig, ax, mat


def plot_masking_ablation(results: dict, figsize=(7, 5)):
    """Inference-time masking: identity FPR with the term vs [MASK]ed, per model."""
    rows = []
    for model, r in results["inference_ablation"].items():
        rows.append({"model": model, "condition": "with identity term", "fpr": r["fpr_with_identity"]})
        rows.append({"model": model, "condition": "identity [MASK]ed", "fpr": r["fpr_masked"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x="model", y="fpr", hue="condition", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("identity false-positive rate")
    ax.set_xlabel("")
    ax.set_title("Masking the identity token collapses false positives\n(reliance on a lexical shortcut)")
    ax.legend(title="")
    fig.tight_layout()
    return fig, ax


def plot_masking_tradeoff(results: dict, figsize=(7, 5)):
    """SVM training-time masking: fairness gain (FPR down) vs cost (recall down)."""
    b = results["svm_training_masking"]
    metrics = [
        ("identity FPR", b["identity_fpr_before"], b["identity_fpr_after"]),
        ("identity-hate recall", b["identity_hate_recall_before"], b["identity_hate_recall_after"]),
        ("toxic recall", b["toxic_recall_before"], b["toxic_recall_after"]),
    ]
    rows = []
    for name, before, after in metrics:
        rows.append({"metric": name, "when": "before masking", "value": before})
        rows.append({"metric": name, "when": "after masking", "value": after})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x="metric", y="value", hue="when", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_xlabel("")
    ax.set_title("SVM trained without identity terms:\nlower identity FPR, but recall on real identity-hate drops")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig, ax


def save_figure(fig, path: Path, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
