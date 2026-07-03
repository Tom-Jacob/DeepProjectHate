"""Shared smoke / eval samples for cross-notebook comparison."""
from typing import List, Optional

import pandas as pd

from paths import SEED, TOXIC_LABELS, eval_n_each

SMOKE_TEXTS = [
    "You are an idiot and I hate you.",
    "Have a wonderful day, thank you for your help!",
    "I am Muslim and proud of my heritage.",
    "lol that was funny",
    "I will kill you",
]

# Reference gold for hand-picked smoke sentences (for smoke-test agreement stats).
SMOKE_GOLD_ROWS = [
    {
        "comment_text": SMOKE_TEXTS[0],
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0,
    },
    {
        "comment_text": SMOKE_TEXTS[1],
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    },
    {
        "comment_text": SMOKE_TEXTS[2],
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    },
    {
        "comment_text": SMOKE_TEXTS[3],
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    },
    {
        "comment_text": SMOKE_TEXTS[4],
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 1,
        "insult": 0,
        "identity_hate": 0,
    },
]


def get_smoke_texts() -> List[str]:
    return list(SMOKE_TEXTS)


def get_smoke_gold_df() -> pd.DataFrame:
    """Gold labels for hand-picked smoke sentences (reference, not from Kaggle)."""
    return pd.DataFrame(SMOKE_GOLD_ROWS)[["comment_text"] + TOXIC_LABELS]


def get_balanced_sample(
    train_df: pd.DataFrame,
    n_each: Optional[int] = None,
    seed: int = SEED,
    quick_mode: Optional[bool] = None,
) -> pd.DataFrame:
    """Stratified sample: n_each toxic + n_each non-toxic (shuffled)."""
    if n_each is None:
        n_each = eval_n_each(quick_mode)

    sample = pd.concat(
        [
            train_df[train_df["toxic"] == 1].sample(n_each, random_state=seed),
            train_df[train_df["toxic"] == 0].sample(n_each, random_state=seed),
        ]
    )
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def show_random_rows(df: pd.DataFrame, n: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Sample random rows for display (reproducible with seed)."""
    return df.sample(min(n, len(df)), random_state=seed)
