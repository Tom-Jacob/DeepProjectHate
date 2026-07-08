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
    kind: str = "hf"    # "hf" | "baseline" | "decoder"
    decoder_method: Optional[str] = None  # "d1" (yes/no) | "d2" (fill-in-blank)


# Core roster: same-data, different-architecture fair comparison.
# toxic-bert (BERT encoder, in-domain) is the evaluation basis; TF-IDF+SVM is
# the non-neural floor we train on the SAME Jigsaw data; DistilGPT-2 (decoder,
# in DECODER_SPECS) is the third architecture. All reduced to P(toxic).
MODEL_SPECS = {
    "toxic-bert": ModelSpec(
        key="toxic-bert",
        model_id="unitary/toxic-bert",
        family="BERT",
        specialization="in-domain (Jigsaw)",
    ),
    "tfidf-svm": ModelSpec(
        key="tfidf-svm",
        model_id="tfidf+linearsvm",
        family="non-neural",
        specialization="non-neural",
        kind="baseline",
    ),
}

# Off-the-shelf classifiers trained on OTHER data — excluded from the fair
# comparison (not same-data). Kept addressable as fallbacks, e.g. if our own
# distillation does not pan out and we need a distilled reference point.
FALLBACK_SPECS = {
    "distilbert-tox": ModelSpec(
        key="distilbert-tox",
        model_id="martin-ha/toxic-comment-model",
        family="DistilBERT",
        specialization="general-toxicity (other data)",
    ),
    "roberta-tox": ModelSpec(
        key="roberta-tox",
        model_id="s-nlp/roberta_toxicity_classifier",
        family="RoBERTa",
        specialization="general-toxicity (other data)",
    ),
}

# Decoder-side models (Tom's DistilGPT-2 prompting), wrapped to the same
# P(toxic) interface. Kept separate so the encoder scripts don't run them by
# default (per-comment generation is slow on CPU); request via --models.
DECODER_SPECS = {
    "distilgpt2-d1": ModelSpec(
        key="distilgpt2-d1",
        model_id="distilgpt2",
        family="DistilGPT-2",
        specialization="decoder yes/no prompt",
        kind="decoder",
        decoder_method="d1",
    ),
    "distilgpt2-d2": ModelSpec(
        key="distilgpt2-d2",
        model_id="distilgpt2",
        family="DistilGPT-2",
        specialization="decoder fill-in-blank",
        kind="decoder",
        decoder_method="d2",
    ),
}

# Every model addressable by --models in the eval scripts (core + fallback + decoder).
ALL_SPECS = {**MODEL_SPECS, **FALLBACK_SPECS, **DECODER_SPECS}


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


class DecoderToxicScorer:
    """Adapt Tom's DistilGPT-2 prompting (``src/decoder.py``) to ``predict_toxic_proba``.

    Reduces the decoder to a single P(toxic) using the 'toxic' label only, so it
    plugs into the exact same comparison / fairness / robustness harness as the
    encoders. ``method='d1'`` uses the yes/no prompt (P(yes)); ``method='d2'``
    uses the fill-in-blank prompt (P('toxic')). Non-invasive: this calls the
    functions in ``decoder.py`` unchanged.
    """

    def __init__(
        self,
        method: str = "d1",
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
        max_comment_tokens: int = 512,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from paths import DISTILGPT2_MODEL_ID, LOCAL_DISTILGPT2_DIR

        self.method = method
        self.device = device or get_device()
        # DistilGPT-2's context is 1024 tokens; long Jigsaw comments overflow it.
        # Truncate the comment so prompt + continuation stay well under the limit.
        self.max_comment_tokens = max_comment_tokens
        path = model_id or (
            str(LOCAL_DISTILGPT2_DIR) if LOCAL_DISTILGPT2_DIR.exists() else DISTILGPT2_MODEL_ID
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.model.eval()
        self.model_id = DISTILGPT2_MODEL_ID
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))

    def _truncate(self, text: str) -> str:
        ids = self.tokenizer.encode(text)
        if len(ids) > self.max_comment_tokens:
            return self.tokenizer.decode(ids[: self.max_comment_tokens])
        return text

    def predict_toxic_proba(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Return P(toxic) per comment via the 'toxic' label of Tom's decoder."""
        from decoder import (  # local import keeps decoder deps out of the base path
            D2_LABEL_CONTINUATIONS,
            build_d1_prompt,
            build_d2_prompt,
            score_fill_blank,
            score_yes_no,
        )

        out: List[float] = []
        for text in texts:
            text = self._truncate(text)
            if self.method == "d2":
                prompt = build_d2_prompt(text)
                pos, neg = D2_LABEL_CONTINUATIONS["toxic"]
                scored = score_fill_blank(self.model, self.tokenizer, prompt, self.device, pos, neg)
                out.append(scored.p_positive_logprob)
            else:
                prompt = build_d1_prompt(text, "toxic")
                scored = score_yes_no(self.model, self.tokenizer, prompt, self.device)
                out.append(scored.p_yes_logprob)
        return np.asarray(out, dtype=float)


def load_scorer(spec: ModelSpec, device: Optional[torch.device] = None):
    """Instantiate a scorer for a spec. Baselines are constructed unfit."""
    if spec.kind == "baseline":
        from baselines import TfidfSvmScorer  # local import: sklearn-only path

        return TfidfSvmScorer()
    if spec.kind == "decoder":
        return DecoderToxicScorer(method=spec.decoder_method, device=device)
    return HFSeqClassScorer(spec.model_id, device=device)
