"""Metrics for multi-label toxic comment evaluation."""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from paths import TOXIC_LABELS


def preds_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(int)


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        row = {
            "label": label,
            "support_pos": int(yt.sum()),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "accuracy": accuracy_score(yt, yp),
        }
        if y_prob is not None:
            try:
                row["pr_auc"] = average_precision_score(yt, y_prob[:, i])
            except ValueError:
                row["pr_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def multilabel_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    labels = list(labels or TOXIC_LABELS)
    per_label = per_label_metrics(y_true, y_pred, y_prob, labels)
    exact_match = float((y_true == y_pred).all(axis=1).mean())
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match_ratio": exact_match,
        "mean_label_accuracy": float(per_label["accuracy"].mean()),
        "mean_pr_auc": float(per_label["pr_auc"].mean()) if y_prob is not None else float("nan"),
    }


def evaluate_multilabel_frame(
    gold_df: pd.DataFrame,
    prob_df: pd.DataFrame,
    prob_cols: Sequence[str],
    pred_cols: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    merge_on: str = "comment_text",
) -> Dict[str, object]:
    """Join gold + model outputs and compute metrics."""
    labels = list(labels or TOXIC_LABELS)
    merged = gold_df[[merge_on] + labels].merge(prob_df, on=merge_on, how="inner")
    if len(merged) == 0:
        raise ValueError("No rows after merging gold and predictions")

    y_true = merged[labels].values.astype(int)
    if pred_cols is not None:
        y_pred = merged[list(pred_cols)].values.astype(int)
    else:
        probs = merged[list(prob_cols)].values.astype(float)
        y_pred = preds_from_probs(probs, threshold)

    y_prob = merged[list(prob_cols)].values.astype(float) if prob_cols else None
    per_label = per_label_metrics(y_true, y_pred, y_prob, labels)
    summary = multilabel_summary(y_true, y_pred, y_prob, labels)
    return {
        "n": len(merged),
        "threshold": threshold,
        "per_label": per_label,
        "summary": summary,
    }


def encoder_decoder_agreement(
    joined: pd.DataFrame,
    label: str = "toxic",
    enc_pred_col: Optional[str] = None,
    dec_pred_col: Optional[str] = None,
) -> Dict[str, float]:
    enc_pred_col = enc_pred_col or f"enc_pred_{label}"
    dec_pred_col = dec_pred_col or f"d1_pred_logprob_{label}"
    enc = joined[enc_pred_col]
    dec = joined[dec_pred_col]
    valid = enc.notna() & dec.notna()
    if not valid.any():
        return {"agreement": float("nan"), "n": 0}
    agree = (enc[valid].astype(int) == dec[valid].astype(int)).mean()
    return {"agreement": float(agree), "n": int(valid.sum())}


def agreement_by_label(
    joined: pd.DataFrame,
    labels: Optional[Sequence[str]] = None,
    enc_pred_tpl: str = "enc_pred_{}",
    dec_pred_tpl: str = "d1_pred_logprob_{}",
) -> pd.DataFrame:
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for label in labels:
        stats = encoder_decoder_agreement(
            joined,
            label=label,
            enc_pred_col=enc_pred_tpl.format(label),
            dec_pred_col=dec_pred_tpl.format(label),
        )
        rows.append({"label": label, **stats})
    return pd.DataFrame(rows)
