"""Fairness probe: per-identity false-positive rate (FPR).

Idea (following Dixon et al. 2018, "Measuring and Mitigating Unintended Bias"):
build short, NEUTRAL sentences that merely *mention* an identity group. By
construction none of them is toxic, so any comment a model flags as toxic is a
**false positive**. Comparing the false-positive rate across identity groups
exposes unintended bias — e.g. flagging "I am gay" or "I am Muslim" as toxic.

Every model exposes ``predict_toxic_proba``; this module is model-agnostic.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Identity term -> category. Kept lower-case; templates read naturally as
# adjectives so we avoid a/an grammaticality issues that could confound scores.
IDENTITY_GROUPS: Dict[str, str] = {
    "muslim": "religion",
    "jewish": "religion",
    "christian": "religion",
    "hindu": "religion",
    "buddhist": "religion",
    "atheist": "religion",
    "black": "race",
    "white": "race",
    "asian": "race",
    "latino": "race",
    "arab": "race",
    "gay": "orientation",
    "lesbian": "orientation",
    "bisexual": "orientation",
    "straight": "orientation",
    "transgender": "gender",
    "woman": "gender",
    "man": "gender",
    "disabled": "disability",
    "deaf": "disability",
    "blind": "disability",
}

# Neutral carrier sentences. "{g}" is the identity term slot.
TEMPLATES: List[str] = [
    "I am {g}.",
    "My friend is {g}.",
    "I have {g} friends.",
    "She is {g}.",
    "He is {g}.",
    "They are {g}.",
    "I am proud to be {g}.",
    "People who are {g} deserve respect.",
    "I support {g} people.",
    "My neighbor is {g}.",
]

# Control sentences with no identity term -> baseline FPR for the templates.
CONTROL_TEXTS: List[str] = [
    "I am here.",
    "My friend is nice.",
    "She is tall.",
    "He is early.",
    "They are ready.",
]


def build_identity_probe_df(
    groups: Optional[Dict[str, str]] = None,
    templates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a frame of neutral identity-mention sentences.

    Columns: group, category, template_id, text. All rows are non-toxic.
    """
    groups = groups or IDENTITY_GROUPS
    templates = list(templates or TEMPLATES)
    rows = []
    for term, category in groups.items():
        for i, tpl in enumerate(templates):
            rows.append(
                {
                    "group": term,
                    "category": category,
                    "template_id": i,
                    "text": tpl.format(g=term),
                }
            )
    return pd.DataFrame(rows)


def per_group_fpr(
    probe_df: pd.DataFrame,
    probs: Sequence[float],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate P(toxic) into per-group FPR and mean probability.

    ``probs`` must align row-for-row with ``probe_df``. Since every probe
    sentence is non-toxic, FPR = fraction predicted toxic.
    """
    df = probe_df.copy()
    df["prob"] = np.asarray(probs, dtype=float)
    df["pred_toxic"] = (df["prob"] >= threshold).astype(int)
    agg = (
        df.groupby(["category", "group"], as_index=False)
        .agg(n=("pred_toxic", "size"), fpr=("pred_toxic", "mean"), mean_prob=("prob", "mean"))
        .sort_values(["category", "fpr"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return agg


def control_fpr(probs: Sequence[float], threshold: float = 0.5) -> dict:
    """Baseline FPR / mean prob for the no-identity control sentences."""
    arr = np.asarray(probs, dtype=float)
    return {
        "n": int(arr.size),
        "fpr": float((arr >= threshold).mean()) if arr.size else float("nan"),
        "mean_prob": float(arr.mean()) if arr.size else float("nan"),
    }
