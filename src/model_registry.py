"""Unified binary-toxicity scoring across heterogeneous classifiers.

Off-the-shelf toxicity models expose different heads and label schemes:

- ``unitary/toxic-bert``            : 6-way multi-label (sigmoid), 'toxic' = index 0
- ``martin-ha/toxic-comment-model`` : 2-way (softmax), 'toxic' = index 1
- ``s-nlp/roberta_toxicity_classifier`` : 2-way (softmax), 'toxic' = index 1

To compare them *fairly on the same task* we reduce every model to a single
``P(toxic)`` per comment via one common interface (``predict_toxic_proba``).
The TF-IDF + SVM baseline in ``baselines.py`` implements the same interface,
so the comparison script can treat neural and non-neural models identically.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from device import get_device

MAX_LENGTH = 256

# Substrings that mark the positive (toxic) class in an id2label map.
POSITIVE_KEYS = ("toxic", "offensive", "hate", "abusive")


def _find_positive_index(id2label: dict) -> int:
    """Return the class index that denotes toxicity.

    Skips negated labels like 'non-toxic'. Falls back to the last index
    (binary models conventionally put the positive class last).
    """
    for idx, label in id2label.items():
        text = str(label).lower()
        if text.startswith("non") or text.startswith("not"):
            continue
        if any(key in text for key in POSITIVE_KEYS):
            return int(idx)
    return max(int(i) for i in id2label)


@dataclass
class ModelSpec:
    """Metadata for one model in the comparison (for fair-comparison reporting)."""

    key: str            # short name used in tables/plots
    model_id: str       # HF hub id or "tfidf+linearsvm"
    family: str         # architecture family: BERT / DistilBERT / RoBERTa / non-neural
    specialization: str # in-domain / general-toxicity / non-neural
    kind: str = "hf"    # "hf" or "baseline"


# Roster for the encoder-side fair comparison (all reduced to P(toxic)).
MODEL_SPECS = {
    "toxic-bert": ModelSpec(
        key="toxic-bert",
        model_id="unitary/toxic-bert",
        family="BERT",
        specialization="in-domain (Jigsaw)",
    ),
    "distilbert-tox": ModelSpec(
        key="distilbert-tox",
        model_id="martin-ha/toxic-comment-model",
        family="DistilBERT",
        specialization="general-toxicity",
    ),
    "roberta-tox": ModelSpec(
        key="roberta-tox",
        model_id="s-nlp/roberta_toxicity_classifier",
        family="RoBERTa",
        specialization="general-toxicity",
    ),
    "tfidf-svm": ModelSpec(
        key="tfidf-svm",
        model_id="tfidf+linearsvm",
        family="non-neural",
        specialization="non-neural",
        kind="baseline",
    ),
}


class HFSeqClassScorer:
    """Wrap any HF sequence-classification model as a binary P(toxic) scorer.

    Handles both multi-label (sigmoid, one logit per label) and single-label
    (softmax over classes) heads, auto-detected from the model config.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[torch.device] = None,
        max_length: int = MAX_LENGTH,
    ):
        self.model_id = model_id
        self.device = device or get_device()
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

        problem = getattr(self.config, "problem_type", None)
        self.multi_label = problem == "multi_label_classification" or self.config.num_labels > 2
        self.pos_index = _find_positive_index(self.config.id2label)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict_toxic_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return P(toxic) in [0, 1] for each input comment."""
        texts = list(texts)
        if not texts:
            return np.array([], dtype=float)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            if self.multi_label:
                # Independent sigmoid per label; take the 'toxic' logit.
                probs = torch.sigmoid(logits)[:, self.pos_index]
            else:
                # Softmax over classes; take the toxic class probability.
                probs = torch.softmax(logits, dim=-1)[:, self.pos_index]
            chunks.append(probs.cpu().numpy())

        return np.concatenate(chunks)


def load_scorer(spec: ModelSpec, device: Optional[torch.device] = None):
    """Instantiate a scorer for a spec. Baselines are constructed unfit."""
    if spec.kind == "baseline":
        from baselines import TfidfSvmScorer  # local import: sklearn-only path

        return TfidfSvmScorer()
    return HFSeqClassScorer(spec.model_id, device=device)
