"""Encoder (toxic-bert) inference wrapper."""
from typing import List, Optional, Sequence, Union

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from device import get_device
from paths import LOCAL_MODEL_DIR, MODEL_ID, TOXIC_LABELS

MAX_LENGTH = 256


class ToxicBertEncoder:
    """Batch scorer for unitary/toxic-bert multi-label probabilities."""

    def __init__(self, device: Optional[torch.device] = None, model_path: Optional[str] = None):
        self.device = device or get_device()
        path = model_path or (str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.model_id = MODEL_ID if path == MODEL_ID or LOCAL_MODEL_DIR.exists() else path

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        max_length: int = MAX_LENGTH,
    ) -> pd.DataFrame:
        texts = list(texts)
        if not texts:
            return pd.DataFrame(columns=TOXIC_LABELS)

        chunks: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.sigmoid(logits).cpu().numpy()
            chunks.append(probs)

        import numpy as np

        stacked = np.vstack(chunks)
        return pd.DataFrame(stacked, columns=TOXIC_LABELS, index=texts)

    def predict_df(
        self,
        texts: Sequence[str],
        threshold: float = 0.5,
        batch_size: int = 32,
        include_comment: bool = True,
    ) -> pd.DataFrame:
        """Return wide frame: comment_text, label probs, label preds."""
        probs = self.predict_proba(texts, batch_size=batch_size)
        out = probs.reset_index().rename(columns={"index": "comment_text"})
        for label in TOXIC_LABELS:
            out[f"enc_p_{label}"] = out[label]
            out[f"enc_pred_{label}"] = (out[label] >= threshold).astype(int)
            out = out.drop(columns=[label])
        if not include_comment:
            out = out.drop(columns=["comment_text"])
        return out


def load_toxic_bert(device: Optional[torch.device] = None) -> ToxicBertEncoder:
    return ToxicBertEncoder(device=device)
