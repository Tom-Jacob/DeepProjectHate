"""Unified binary-toxicity scoring across heterogeneous classifiers.

Off-the-shelf toxicity models expose different heads and label schemes:

- ``unitary/toxic-bert``            : 6-way multi-label (sigmoid), 'toxic' = index 0
- ``martin-ha/toxic-comment-model`` : 2-way (softmax), 'toxic' = index 1  [legacy, unused]
- ``s-nlp/roberta_toxicity_classifier`` : 2-way (softmax), 'toxic' = index 1  [legacy, unused]

To compare them *fairly on the same task* we reduce every model to a single
``P(toxic)`` per comment via one common interface (``predict_toxic_proba``).
The TF-IDF + SVM baseline in ``baselines.py`` implements the same interface,
so the comparison script can treat neural and non-neural models identically.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from device import get_device
from paths import BERT_BASE_MODEL_ID, LOCAL_BERT_BASE_DIR

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
    kind: str = "hf"    # "hf" | "baseline" | "decoder" | "student" | "mlm-prompt"
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
    "distilbert-student": ModelSpec(
        key="distilbert-student",
        model_id="distilbert-student",  # local models/distilbert-student
        family="DistilBERT",
        specialization="KD student (toxic-bert → DistilBERT, 6-label)",
        kind="student",
    ),
    "bert-base-classifier": ModelSpec(
        key="bert-base-classifier",
        model_id=str(LOCAL_BERT_BASE_DIR),
        family="BERT",
        specialization="generic, fine-tuned by us (classifier head)",
    ),
    "bert-base-mlm-prompt": ModelSpec(
        key="bert-base-mlm-prompt",
        model_id=BERT_BASE_MODEL_ID,
        family="BERT",
        specialization="generic, zero-shot (masked-LM prompt)",
        kind="mlm-prompt",
    ),
}
# bert-base-classifier / bert-base-mlm-prompt are the SAME bert-base-uncased
# checkpoint used two ways -- a trained classifier head vs. zero-shot masked-LM
# prompting -- isolating method from architecture family for the "dedicated
# classifier vs. LM prompting" question (see BertBaseMlmPromptScorer below).

# LEGACY — not used in the report or by any experiment script's default roster.
# Off-the-shelf classifiers trained on OTHER data, kept only as early-exploration
# reference points; excluded from the fair comparison (not same-data). Still
# addressable via --models for anyone who wants to probe them.
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
        specialization="decoder yes/no prompt (zero-shot)",
        kind="decoder",
        decoder_method="d1",
    ),
    "distilgpt2-d1-ft": ModelSpec(
        key="distilgpt2-d1-ft",
        model_id="distilgpt2-d1-ft",
        family="DistilGPT-2",
        specialization="decoder yes/no (fine-tuned on toxic)",
        kind="decoder",
        decoder_method="d1",
    ),
    "distilgpt2-d2": ModelSpec(
        key="distilgpt2-d2",
        model_id="distilgpt2",
        family="DistilGPT-2",
        specialization="decoder fill-in-blank (zero-shot)",
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
        # DistilBERT (and some saved students) reject token_type_ids even if the
        # tokenizer emits them (e.g. BertTokenizer saved next to DistilBERT weights).
        self._drop_token_type_ids = getattr(self.config, "model_type", "") == "distilbert"

    def _encode(self, batch: Sequence[str]) -> dict:
        enc = self.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {k: v.to(self.device) for k, v in enc.items()}
        if self._drop_token_type_ids:
            out.pop("token_type_ids", None)
        return out

    @torch.no_grad()
    def predict_toxic_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return P(toxic) in [0, 1] for each input comment."""
        texts = list(texts)
        if not texts:
            return np.array([], dtype=float)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._encode(batch)
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

        from paths import (
            DISTILGPT2_MODEL_ID,
            LOCAL_DISTILGPT2_D1_FT_DIR,
            LOCAL_DISTILGPT2_DIR,
        )

        self.method = method
        self.device = device or get_device()
        self.max_comment_tokens = max_comment_tokens

        mid = model_id or DISTILGPT2_MODEL_ID
        if mid in ("distilgpt2-d1-ft", "distilgpt2-ft"):
            path = str(LOCAL_DISTILGPT2_D1_FT_DIR)
            if not LOCAL_DISTILGPT2_D1_FT_DIR.exists():
                raise FileNotFoundError(
                    f"Fine-tuned decoder not found at {path}. "
                    "Train on Colab: notebooks/10_decoder_finetune_colab.ipynb "
                    "or: python scripts/train_decoder_ft.py"
                )
            self.model_id = "distilgpt2-d1-ft"
        elif Path(mid).exists():
            path = mid
            self.model_id = Path(mid).name
        else:
            path = str(LOCAL_DISTILGPT2_DIR) if LOCAL_DISTILGPT2_DIR.exists() else DISTILGPT2_MODEL_ID
            self.model_id = DISTILGPT2_MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.model.eval()
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


class BertBaseMlmPromptScorer:
    """Zero-shot toxicity classifier via bert-base-uncased's masked-LM head.

    The BERT analogue of the decoder's D2 fill-in-blank prompt: instead of a
    causal-LM continuation, we ask the model's native masked-LM objective to
    fill a single ``[MASK]`` in "Comment: {text}\\nThis comment is [MASK]."
    and compare the fill logit for a toxic vs. a non-toxic token. No training
    involved — this is the "prompting" half of the same-model comparison
    against ``bert-base-classifier`` (the trained-head half).

    If the input text already contains a literal "[MASK]" (e.g. the identity-
    masking intervention), the LAST mask in the sequence is scored — that is
    always the one this class appended for the fill-blank slot.
    """

    POSITIVE_WORD = "toxic"
    # Candidates tried in order; the first that tokenizes to a single
    # in-vocabulary token is used as the negative class word.
    NEGATIVE_WORD_CANDIDATES = ("fine", "clean", "nice", "acceptable", "harmless")

    def __init__(
        self,
        model_id: str = BERT_BASE_MODEL_ID,
        device: Optional[torch.device] = None,
        max_comment_tokens: int = 200,
    ):
        from transformers import AutoModelForMaskedLM

        self.device = device or get_device()
        self.max_comment_tokens = max_comment_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pos_token_id = self._single_token_id(self.POSITIVE_WORD)
        self.neg_token_id = None
        for candidate in self.NEGATIVE_WORD_CANDIDATES:
            token_id = self._single_token_id(candidate)
            if token_id is not None:
                self.neg_token_id = token_id
                break
        if self.neg_token_id is None:
            raise RuntimeError("No single-token negative word found for BertBaseMlmPromptScorer")

    def _single_token_id(self, word: str) -> Optional[int]:
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _truncate(self, text: str) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_comment_tokens:
            return self.tokenizer.decode(ids[: self.max_comment_tokens])
        return text

    def _build_prompt(self, text: str) -> str:
        return f"Comment: {self._truncate(text)}\nThis comment is {self.mask_token}."

    @torch.no_grad()
    def predict_toxic_proba(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.array([], dtype=float)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [self._build_prompt(t) for t in texts[start : start + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_comment_tokens + 32,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # (batch, seq, vocab)

            probs = np.empty(len(batch), dtype=float)
            for row in range(len(batch)):
                mask_positions = (enc["input_ids"][row] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    # Truncation swallowed the mask (pathological input) -> abstain at 0.
                    probs[row] = 0.0
                    continue
                last_mask_pos = mask_positions[-1].item()
                mask_logits = logits[row, last_mask_pos, :]
                pair = torch.stack([mask_logits[self.pos_token_id], mask_logits[self.neg_token_id]])
                p_pos = torch.softmax(pair, dim=0)[0].item()
                probs[row] = p_pos
            chunks.append(probs)

        return np.concatenate(chunks)


def load_scorer(spec: ModelSpec, device: Optional[torch.device] = None):
    """Instantiate a scorer for a spec. Baselines are constructed unfit."""
    if spec.kind == "baseline":
        from baselines import TfidfSvmScorer  # local import: sklearn-only path

        return TfidfSvmScorer()
    if spec.kind == "decoder":
        return DecoderToxicScorer(
            method=spec.decoder_method,
            model_id=spec.model_id,
            device=device,
        )
    if spec.kind == "student":
        from distill import DistilBertStudentScorer
        from paths import LOCAL_DISTILBERT_STUDENT_DIR

        return DistilBertStudentScorer(model_path=str(LOCAL_DISTILBERT_STUDENT_DIR), device=device)
    if spec.kind == "mlm-prompt":
        return BertBaseMlmPromptScorer(model_id=spec.model_id, device=device)
    # Prefer local toxic-bert mirror when present (avoids hub / None path issues).
    model_id = spec.model_id
    if spec.key == "toxic-bert":
        from paths import LOCAL_MODEL_DIR, MODEL_ID

        model_id = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID
        if model_id is None:
            raise TypeError("toxic-bert model id is None — check paths.MODEL_ID / LOCAL_MODEL_DIR")
    return HFSeqClassScorer(model_id, device=device)
