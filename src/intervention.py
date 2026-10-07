"""Masking intervention: probe *why* models over-flag identity mentions.

Two complementary masking views, applied to the main model (toxic-bert) and the
second-best, different-architecture model (TF-IDF+SVM):

1. Inference-time masking (both models, CPU): replace the identity term in a
   probe sentence with a mask token and re-score. If P(toxic) collapses, the
   model was firing on the identity *token itself* — a lexical shortcut, not
   understanding. This works on already-trained models (no GPU retrain).

2. Training-time masking (SVM only, CPU): retrain the SVM with identity terms
   removed from the text, then measure the fairness gain (lower identity FPR)
   AND its cost (lower recall on genuinely identity-directed hate). This is the
   causal version of the proposal's intervention, feasible without a GPU.
"""
import re
from typing import List, Sequence

import numpy as np

from fairness import IDENTITY_GROUPS

# The identity terms we mask (same set used by the fairness probe).
IDENTITY_TERMS: List[str] = list(IDENTITY_GROUPS.keys())

MASK_TOKEN = "[MASK]"  # BERT reads this as its mask token; for the SVM it is a
                       # near-neutral out-of-vocabulary stand-in.


def _term_pattern(terms: Sequence[str]) -> "re.Pattern":
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def mask_terms(text: str, terms: Sequence[str] = IDENTITY_TERMS, placeholder: str = MASK_TOKEN) -> str:
    """Replace whole-word identity terms with a placeholder token."""
    return _term_pattern(terms).sub(placeholder, text)


def strip_terms(text: str, terms: Sequence[str] = IDENTITY_TERMS) -> str:
    """Remove identity terms entirely (collapse extra spaces). For training masking."""
    out = _term_pattern(terms).sub(" ", text)
    return re.sub(r"\s+", " ", out).strip()


def inference_ablation(scorer, probe_df, threshold: float = 0.5) -> dict:
    """Compare a scorer's identity-probe FPR with vs without the identity term.

    ``probe_df`` needs a ``text`` column (neutral identity-mention sentences).
    Returns FPR and mean P(toxic) for original vs masked, and the reduction.
    """
    texts = probe_df["text"].tolist()
    masked = [mask_terms(t) for t in texts]
    p_orig = np.asarray(scorer.predict_toxic_proba(texts), dtype=float)
    p_mask = np.asarray(scorer.predict_toxic_proba(masked), dtype=float)
    fpr_orig = float((p_orig >= threshold).mean())
    fpr_mask = float((p_mask >= threshold).mean())
    return {
        "fpr_with_identity": fpr_orig,
        "fpr_masked": fpr_mask,
        "fpr_reduction": fpr_orig - fpr_mask,
        "mean_prob_with_identity": float(p_orig.mean()),
        "mean_prob_masked": float(p_mask.mean()),
    }
