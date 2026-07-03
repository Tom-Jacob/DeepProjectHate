"""Merge encoder and decoder score tables."""
import pandas as pd

from paths import TOXIC_LABELS


def encoder_score_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("enc_") or c in ("model_id", "threshold")]


def join_encoder_decoder(encoder_df: pd.DataFrame, decoder_df: pd.DataFrame) -> pd.DataFrame:
    enc = encoder_df[["comment_text"] + encoder_score_columns(encoder_df)].copy()
    joined = decoder_df.merge(enc, on="comment_text", how="inner", suffixes=("", "_encdup"))
    dup = [c for c in joined.columns if c.endswith("_encdup")]
    if dup:
        joined = joined.drop(columns=dup)
    return joined
