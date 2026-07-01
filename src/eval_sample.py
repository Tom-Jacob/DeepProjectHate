"""Shared smoke / eval samples for cross-notebook comparison."""
from typing import List, Optional

import pandas as pd

from paths import SEED, eval_n_each

SMOKE_TEXTS = [
    "You are an idiot and I hate you.",
    "Have a wonderful day, thank you for your help!",
    "I am Muslim and proud of my heritage.",
    "lol that was funny",
    "I will kill you",
]


def get_smoke_texts() -> List[str]:
    return list(SMOKE_TEXTS)


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
