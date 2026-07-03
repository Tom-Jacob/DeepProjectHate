"""Load and summarize Jigsaw 2018 toxic comment data."""
from pathlib import Path

import pandas as pd

from paths import (
    RAW_JIGSAW_DIR,
    SEED,
    TEST_CSV,
    TEST_LABELS_CSV,
    TOXIC_LABELS,
    TRAIN_CSV,
)


def verify_data_files() -> None:
    """Raise FileNotFoundError with setup instructions if CSVs are missing."""
    missing = [p for p in (TRAIN_CSV, TEST_CSV, TEST_LABELS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Jigsaw CSV files:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nDownload into {RAW_JIGSAW_DIR} — see README.md or run:\n"
            "  python scripts/download_jigsaw.py"
        )


def load_train_df() -> pd.DataFrame:
    verify_data_files()
    df = pd.read_csv(TRAIN_CSV)
    for label in TOXIC_LABELS:
        df[label] = df[label].astype(int)
    return df


def load_test_df() -> pd.DataFrame:
    verify_data_files()
    texts = pd.read_csv(TEST_CSV)
    labels = pd.read_csv(TEST_LABELS_CSV)
    df = texts.merge(labels, on="id")
    df = df[df["toxic"] != -1]
    for label in TOXIC_LABELS:
        df[label] = df[label].astype(int)
    return df


def add_toxic_any(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["toxic_any"] = (out[TOXIC_LABELS].sum(axis=1) > 0).astype(int)
    return out


def label_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for label in TOXIC_LABELS:
        count = int(df[label].sum())
        rows.append(
            {
                "label": label,
                "count": count,
                "pct": 100.0 * count / n if n else 0.0,
            }
        )
    toxic_any = int((df[TOXIC_LABELS].sum(axis=1) > 0).sum())
    rows.append(
        {
            "label": "toxic_any",
            "count": toxic_any,
            "pct": 100.0 * toxic_any / n if n else 0.0,
        }
    )
    return pd.DataFrame(rows)


def comment_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    chars = df["comment_text"].astype(str).str.len()
    words = df["comment_text"].astype(str).str.split().str.len()
    return pd.DataFrame(
        {
            "metric": ["chars_mean", "chars_median", "chars_max", "words_mean", "words_median", "words_max"],
            "value": [
                chars.mean(),
                chars.median(),
                chars.max(),
                words.mean(),
                words.median(),
                words.max(),
            ],
        }
    )


def label_cooccurrence(df: pd.DataFrame, mask_diagonal: bool = False) -> pd.DataFrame:
    """Pairwise co-label counts: entry (i, j) = comments where both labels are 1.

    Diagonal (i, i) equals the marginal count for label i alone — not a cross-label
    co-occurrence. Use mask_diagonal=True for heatmaps (sets diagonal to NaN).
    """
    cooc = df[TOXIC_LABELS].T.dot(df[TOXIC_LABELS])
    if not mask_diagonal:
        return cooc
    import numpy as np

    arr = cooc.values.astype(float)
    np.fill_diagonal(arr, np.nan)
    return pd.DataFrame(arr, index=cooc.index, columns=cooc.columns)


def sample_examples(
    df: pd.DataFrame, label: str, n: int = 3, toxic: bool = True, seed: int = SEED
) -> pd.DataFrame:
    subset = df[df[label] == (1 if toxic else 0)]
    k = min(n, len(subset))
    return subset[["comment_text"] + TOXIC_LABELS].sample(k, random_state=seed)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
