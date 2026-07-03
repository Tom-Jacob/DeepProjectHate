"""Train / val / test index splits for formal evaluation."""
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from data import add_toxic_any, load_test_df, load_train_df
from paths import SEED, SPLIT_INDICES_PATH, TOXIC_LABELS, TRAIN_FRAC, VAL_FRAC


def make_split_indices(
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    """Split train.csv into train + val (stratified on toxic_any).

    Returns a long-format DataFrame: id, split in {train, val, test}.
    Test rows come from labeled Jigsaw test.csv + test_labels.csv.
    """
    train_df = add_toxic_any(load_train_df())
    train_idx, val_idx = train_test_split(
        train_df.index,
        train_size=train_frac,
        random_state=seed,
        stratify=train_df["toxic_any"],
    )

    rows = []
    for idx in train_idx:
        rows.append({"id": str(train_df.loc[idx, "id"]), "split": "train"})
    for idx in val_idx:
        rows.append({"id": str(train_df.loc[idx, "id"]), "split": "val"})

    test_df = load_test_df()
    for row_id in test_df["id"].astype(str):
        rows.append({"id": row_id, "split": "test"})

    return pd.DataFrame(rows)


def save_split_indices(path=None, **kwargs) -> pd.DataFrame:
    path = path or SPLIT_INDICES_PATH
    df = make_split_indices(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_split_indices(path=None) -> pd.DataFrame:
    path = path or SPLIT_INDICES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun: python scripts/make_splits.py"
        )
    return pd.read_parquet(path)


def get_split_frame(split: str, path=None) -> pd.DataFrame:
    """Load comments + gold labels for one split name."""
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val, or test — got {split!r}")

    indices = load_split_indices(path)
    ids = set(indices.loc[indices["split"] == split, "id"].astype(str))

    if split == "test":
        df = load_test_df()
    else:
        df = load_train_df()
        df = df[df["id"].astype(str).isin(ids)].copy()
        return df.reset_index(drop=True)

    df = df[df["id"].astype(str).isin(ids)].copy()
    return df.reset_index(drop=True)


def split_counts(path=None) -> pd.DataFrame:
    indices = load_split_indices(path)
    return indices.groupby("split", as_index=False).size().rename(columns={"size": "n"})


def sample_split_frame(
    split: str,
    n: Optional[int] = None,
    seed: int = SEED,
    path=None,
) -> pd.DataFrame:
    """Optional cap for quick formal eval runs."""
    df = get_split_frame(split, path=path)
    if n is None or n >= len(df):
        return df
    return df.sample(n, random_state=seed).reset_index(drop=True)
