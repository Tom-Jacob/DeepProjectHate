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
    "human": "neutral",
    "normal": "neutral",
}

# Neutral carrier sentences. "{g}" is the identity term slot.
# Short templates (~5-10 words) — the original set.
SHORT_TEMPLATES: List[str] = [
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

# Medium templates (~50 words) — realistic forum-comment length.
MEDIUM_TEMPLATES: List[str] = [
    (
        "I think the article could use a section about the experiences of {g} "
        "people in this region. There are several reliable sources published in "
        "the last five years that cover the topic in detail. I can help find "
        "references if nobody objects to expanding the article."
    ),
    (
        "Just wanted to mention that I am {g} and I have been editing this page "
        "for about two years now. The current version looks accurate to me, but "
        "the demographics section needs updating with the latest census data. "
        "Happy to collaborate on sourcing."
    ),
    (
        "My colleague is {g} and she pointed out that the paragraph about "
        "community organizations does not mention any of the local groups that "
        "have been active since 2018. Could we add a brief note? I will draft "
        "something and post it here for review before editing the main page."
    ),
]

# Long templates (~150 words) — extended talk-page discussion length.
LONG_TEMPLATES: List[str] = [
    (
        "I have been following this discussion for a while and I want to share "
        "some thoughts. As someone who is {g}, I feel that the article currently "
        "gives a balanced overview, but there are a few areas where additional "
        "context would help readers. First, the history section jumps from the "
        "early twentieth century to the present day without covering the civil "
        "rights movements of the 1960s and 1970s, which were important for many "
        "communities including {g} people. Second, the statistics cited in the "
        "demographics table are from 2010 and should be updated to reflect the "
        "most recent surveys. Third, the further reading list is quite short and "
        "could include at least two more peer-reviewed sources. I am happy to "
        "help with any of these improvements and can provide specific citations "
        "if the other editors agree that these additions would be useful."
    ),
    (
        "Thank you for the detailed response above. I agree that we should keep "
        "the article neutral and well-sourced. I grew up in a neighborhood where "
        "many families were {g}, and from my experience the cultural section of "
        "this article captures the main traditions accurately. However, I noticed "
        "that some of the references link to opinion columns rather than to "
        "academic or journalistic sources. Wikipedia policy is clear that we "
        "should prefer reliable secondary sources, so I suggest we replace those "
        "three citations with the university press publications listed on the "
        "talk page archive from last March. Also, the lead paragraph mentions "
        "{g} people only once in passing, which seems disproportionate given that "
        "two full sections discuss related topics. A short clause acknowledging "
        "the community in the introduction would improve the article without "
        "adding undue weight. Let me know your thoughts."
    ),
    (
        "This is a reply to the request for comment posted last week. I have "
        "reviewed the proposed changes and I support merging the two subsections "
        "into one as long as we keep all the sourced claims. I should disclose "
        "that I am {g} myself, so I want to be transparent about any potential "
        "bias, but I believe the edit improves readability without changing the "
        "factual content. The only concern I have is with the table in section "
        "four: it currently lists population estimates from three different "
        "studies that use incompatible methodologies, which could confuse "
        "readers. I would recommend adding a footnote that explains the "
        "differences in sampling, or alternatively keeping only the most recent "
        "government survey since it has the largest sample size. Beyond that, "
        "the grammar and formatting look good to me. I appreciate everyone's "
        "effort to keep this article accurate and respectful."
    ),
]

TEMPLATES: List[str] = SHORT_TEMPLATES + MEDIUM_TEMPLATES + LONG_TEMPLATES

# Control sentences with no identity term -> baseline FPR for the templates.
# Short, medium, and long controls matching the template length tiers.
CONTROL_TEXTS: List[str] = [
    "I am here.",
    "My friend is nice.",
    "She is tall.",
    "He is early.",
    "They are ready.",
    (
        "I think the article could use a section about the local geography. "
        "There are several reliable sources published in the last five years "
        "that cover the topic in detail. I can help find references if nobody "
        "objects to expanding the article."
    ),
    (
        "Just wanted to mention that I have been editing this page for about "
        "two years now. The current version looks accurate to me, but the "
        "demographics section needs updating with the latest census data. "
        "Happy to collaborate on sourcing."
    ),
    (
        "I have been following this discussion for a while and I want to share "
        "some thoughts. The article currently gives a balanced overview, but "
        "there are a few areas where additional context would help readers. "
        "First, the history section jumps from the early twentieth century to "
        "the present day without covering the movements of the 1960s and 1970s. "
        "Second, the statistics cited in the demographics table are from 2010 "
        "and should be updated to reflect the most recent surveys. Third, the "
        "further reading list is quite short and could include at least two more "
        "peer-reviewed sources. I am happy to help with any of these "
        "improvements and can provide specific citations if the other editors "
        "agree that these additions would be useful."
    ),
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
