"""DistilGPT-2 prompting helpers for binary label scoring (D1 + D2).

D1 scoring paths (per label):
  - logprob_full: full continuation log P(yes) vs log P(no)
  - logprob_first_token (C): first-token logprob only
  - constrained (D): greedy decode restricted to yes/no token ids
  - generate_parse (B): free generation; abstain if unparseable

D2 scoring paths (per label): same four methods with positive/negative continuations
after "Comment: ... This comment is ___". Also supports argmax over label words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from paths import TOXIC_LABELS

# Positive vs negative continuations after "This comment is"
D2_LABEL_CONTINUATIONS: Dict[str, Tuple[str, str]] = {
    "toxic": (" toxic", " not"),
    "severe_toxic": (" severely toxic", " harmless"),
    "obscene": (" obscene", " clean"),
    "threat": (" threatening", " harmless"),
    "insult": (" insulting", " respectful"),
    "identity_hate": (" hateful", " respectful"),
}


def _label_phrase(label: str) -> str:
    return label.replace("_", " ")


def build_d1_prompt(text: str, label: str) -> str:
    phrase = _label_phrase(label)
    return (
        f"Is the following comment {phrase}? Answer yes or no.\n"
        f"Comment: {text}\n"
        f"Answer:"
    )


def build_d2_prompt(text: str) -> str:
    return f"Comment: {text}\nThis comment is"


def _encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _continuation_logprob(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    cont_ids: Sequence[int],
    device: torch.device,
) -> float:
    if not cont_ids:
        return float("-inf")
    input_ids = list(prompt_ids) + list(cont_ids)
    input_tensor = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(input_tensor).logits
        log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for k, token_id in enumerate(cont_ids):
        idx = len(prompt_ids) + k
        total += log_probs[0, idx - 1, token_id].item()
    return total


def _first_step_logprobs(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    input_tensor = torch.tensor([list(prompt_ids)], device=device)
    with torch.no_grad():
        logits = model(input_tensor).logits
        return torch.log_softmax(logits[0, -1, :], dim=-1)


def _softmax_two(log_a: float, log_b: float) -> Tuple[float, float]:
    logits = torch.tensor([log_a, log_b])
    probs = F.softmax(logits, dim=0)
    return probs[0].item(), probs[1].item()


def _softmax_many(logs: Sequence[float]) -> List[float]:
    t = torch.tensor(list(logs))
    return F.softmax(t, dim=0).tolist()


def _logprob_from_token_ids(step_logprobs: torch.Tensor, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return float("-inf")
    return step_logprobs[token_ids[0]].item()


def _logprob_best_in_set(step_logprobs: torch.Tensor, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return float("-inf")
    return max(step_logprobs[tid].item() for tid in token_ids)


def parse_yes_no(text: str) -> Optional[int]:
    """B (D1): parse free-form generation; None = abstain."""
    text = text.strip().lower()
    if not text:
        return None
    first = text.split()[0]
    if first in ("yes", "yeah", "y"):
        return 1
    if first in ("no", "nope", "n"):
        return 0
    if re.search(r"\byes\b", text):
        return 1
    if re.search(r"\bno\b", text):
        return 0
    return None


def _parse_fill_blank_labeled(text: str, positive_token: str, negative_token: str) -> Optional[int]:
    """B (D2): parse completion as positive vs negative phrase; None = abstain."""
    gen_lower = text.strip().lower()
    if not gen_lower:
        return None
    pos = positive_token.strip().lower()
    neg = negative_token.strip().lower()
    first = gen_lower.split()[0]
    pos_first = pos.split()[0] if pos else ""
    neg_first = neg.split()[0] if neg else ""
    if first == pos_first or gen_lower.startswith(pos):
        return 1
    if first == neg_first or gen_lower.startswith(neg):
        return 0
    if pos in gen_lower:
        return 1
    if neg in gen_lower:
        return 0
    return None


def _token_to_binary(token_id: int, pos_ids: Sequence[int], neg_ids: Sequence[int]) -> Optional[int]:
    if token_id in pos_ids:
        return 1
    if token_id in neg_ids:
        return 0
    return None


def _make_attention_mask(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_tensor)


@dataclass
class YesNoScore:
    p_yes_logprob: float
    p_no_logprob: float
    pred_logprob: int
    p_yes_first_token: float
    p_no_first_token: float
    pred_logprob_first_token: int
    p_yes_constrained: float
    p_no_constrained: float
    pred_constrained: int
    pred_generate: Optional[int]
    parse_abstain: bool
    raw_completion: str
    scoring_mismatch: bool


@dataclass
class FillBlankScore:
    p_positive_logprob: float
    p_negative_logprob: float
    pred_logprob: int
    p_positive_first_token: float
    p_negative_first_token: float
    pred_logprob_first_token: int
    p_positive_constrained: float
    p_negative_constrained: float
    pred_constrained: int
    pred_generate: Optional[int]
    parse_abstain: bool
    raw_completion: str
    scoring_mismatch: bool


def score_yes_no(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    yes_token: str = " yes",
    no_token: str = " no",
    max_new_tokens: int = 5,
) -> YesNoScore:
    prompt_ids = _encode(tokenizer, prompt)
    yes_ids = _encode(tokenizer, yes_token)
    no_ids = _encode(tokenizer, no_token)

    log_yes = _continuation_logprob(model, prompt_ids, yes_ids, device)
    log_no = _continuation_logprob(model, prompt_ids, no_ids, device)
    p_yes_logprob, p_no_logprob = _softmax_two(log_yes, log_no)
    pred_logprob = int(p_yes_logprob >= p_no_logprob)

    step_lp = _first_step_logprobs(model, prompt_ids, device)
    log_yes_first = _logprob_from_token_ids(step_lp, yes_ids)
    log_no_first = _logprob_from_token_ids(step_lp, no_ids)
    p_yes_first, p_no_first = _softmax_two(log_yes_first, log_no_first)
    pred_logprob_first = int(p_yes_first >= p_no_first)

    log_yes_c = _logprob_best_in_set(step_lp, yes_ids)
    log_no_c = _logprob_best_in_set(step_lp, no_ids)
    p_yes_constrained, p_no_constrained = _softmax_two(log_yes_c, log_no_c)
    pred_constrained = int(p_yes_constrained >= p_no_constrained)

    allowed_ids = list(set(yes_ids) | set(no_ids))

    def _prefix_allowed_tokens_fn(_batch_id: int, _input_ids: torch.Tensor) -> List[int]:
        return allowed_ids

    input_tensor = torch.tensor([prompt_ids], device=device)
    attention_mask = _make_attention_mask(input_tensor)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn,
        )
    constrained_token = out[0, len(prompt_ids)].item()
    constrained_pred = _token_to_binary(constrained_token, yes_ids, no_ids)
    if constrained_pred is not None:
        pred_constrained = constrained_pred

    with torch.no_grad():
        out_free = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out_free[0, len(prompt_ids) :]
    raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred_generate = parse_yes_no(raw_completion)
    parse_abstain = pred_generate is None
    scoring_mismatch = (pred_generate is not None) and (pred_logprob != pred_generate)

    return YesNoScore(
        p_yes_logprob=p_yes_logprob,
        p_no_logprob=p_no_logprob,
        pred_logprob=pred_logprob,
        p_yes_first_token=p_yes_first,
        p_no_first_token=p_no_first,
        pred_logprob_first_token=pred_logprob_first,
        p_yes_constrained=p_yes_constrained,
        p_no_constrained=p_no_constrained,
        pred_constrained=pred_constrained,
        pred_generate=pred_generate,
        parse_abstain=parse_abstain,
        raw_completion=raw_completion,
        scoring_mismatch=scoring_mismatch,
    )


def score_fill_blank(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    positive_token: str = " toxic",
    negative_token: str = " not",
    max_new_tokens: int = 5,
) -> FillBlankScore:
    prompt_ids = _encode(tokenizer, prompt)
    pos_ids = _encode(tokenizer, positive_token)
    neg_ids = _encode(tokenizer, negative_token)

    log_pos = _continuation_logprob(model, prompt_ids, pos_ids, device)
    log_neg = _continuation_logprob(model, prompt_ids, neg_ids, device)
    p_pos, p_neg = _softmax_two(log_pos, log_neg)
    pred_logprob = int(p_pos >= p_neg)

    step_lp = _first_step_logprobs(model, prompt_ids, device)
    log_pos_first = _logprob_from_token_ids(step_lp, pos_ids)
    log_neg_first = _logprob_from_token_ids(step_lp, neg_ids)
    p_pos_first, p_neg_first = _softmax_two(log_pos_first, log_neg_first)
    pred_logprob_first = int(p_pos_first >= p_neg_first)

    log_pos_c = _logprob_best_in_set(step_lp, pos_ids)
    log_neg_c = _logprob_best_in_set(step_lp, neg_ids)
    p_pos_c, p_neg_c = _softmax_two(log_pos_c, log_neg_c)
    pred_constrained = int(p_pos_c >= p_neg_c)

    allowed_ids = list(set(pos_ids) | set(neg_ids))

    def _prefix_allowed_tokens_fn(_batch_id: int, _input_ids: torch.Tensor) -> List[int]:
        return allowed_ids

    input_tensor = torch.tensor([prompt_ids], device=device)
    attention_mask = _make_attention_mask(input_tensor)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn,
        )
    constrained_token = out[0, len(prompt_ids)].item()
    constrained_pred = _token_to_binary(constrained_token, pos_ids, neg_ids)
    if constrained_pred is not None:
        pred_constrained = constrained_pred

    with torch.no_grad():
        out_free = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out_free[0, len(prompt_ids) :]
    raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred_generate = _parse_fill_blank_labeled(raw_completion, positive_token, negative_token)
    parse_abstain = pred_generate is None
    scoring_mismatch = (pred_generate is not None) and (pred_logprob != pred_generate)

    return FillBlankScore(
        p_positive_logprob=p_pos,
        p_negative_logprob=p_neg,
        pred_logprob=pred_logprob,
        p_positive_first_token=p_pos_first,
        p_negative_first_token=p_neg_first,
        pred_logprob_first_token=pred_logprob_first,
        p_positive_constrained=p_pos_c,
        p_negative_constrained=p_neg_c,
        pred_constrained=pred_constrained,
        pred_generate=pred_generate,
        parse_abstain=parse_abstain,
        raw_completion=raw_completion,
        scoring_mismatch=scoring_mismatch,
    )


def score_d2_argmax(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Single-label probe: which positive continuation is most likely at the first step."""
    labels = list(labels or TOXIC_LABELS)
    prompt = build_d2_prompt(text)
    prompt_ids = _encode(tokenizer, prompt)
    step_lp = _first_step_logprobs(model, prompt_ids, device)

    log_scores = []
    for label in labels:
        pos_token = D2_LABEL_CONTINUATIONS[label][0]
        pos_ids = _encode(tokenizer, pos_token)
        log_scores.append(_logprob_best_in_set(step_lp, pos_ids))

    probs = _softmax_many(log_scores)
    best_idx = int(max(range(len(labels)), key=lambda i: log_scores[i]))

    row = {"d2_argmax_label": labels[best_idx]}
    for i, label in enumerate(labels):
        row[f"d2_argmax_logscore_{label}"] = log_scores[i]
        row[f"d2_argmax_prob_{label}"] = probs[i]
    return row


def _yesno_row_columns(label: str, scored: YesNoScore) -> dict:
    return {
        f"d1_p_yes_{label}": scored.p_yes_logprob,
        f"d1_pred_logprob_{label}": scored.pred_logprob,
        f"d1_p_yes_first_{label}": scored.p_yes_first_token,
        f"d1_pred_logprob_first_{label}": scored.pred_logprob_first_token,
        f"d1_p_yes_constrained_{label}": scored.p_yes_constrained,
        f"d1_pred_constrained_{label}": scored.pred_constrained,
        f"d1_pred_generate_{label}": scored.pred_generate,
        f"d1_parse_abstain_{label}": scored.parse_abstain,
        f"d1_raw_completion_{label}": scored.raw_completion,
        f"d1_scoring_mismatch_{label}": scored.scoring_mismatch,
    }


def _fillblank_row_columns(label: str, scored: FillBlankScore) -> dict:
    return {
        f"d2_p_positive_{label}": scored.p_positive_logprob,
        f"d2_pred_logprob_{label}": scored.pred_logprob,
        f"d2_p_positive_first_{label}": scored.p_positive_first_token,
        f"d2_pred_logprob_first_{label}": scored.pred_logprob_first_token,
        f"d2_p_positive_constrained_{label}": scored.p_positive_constrained,
        f"d2_pred_constrained_{label}": scored.pred_constrained,
        f"d2_pred_generate_{label}": scored.pred_generate,
        f"d2_parse_abstain_{label}": scored.parse_abstain,
        f"d2_raw_completion_{label}": scored.raw_completion,
        f"d2_scoring_mismatch_{label}": scored.scoring_mismatch,
    }


def predict_multilabel_d1(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for text in texts:
        row = {"comment_text": text}
        for label in labels:
            prompt = build_d1_prompt(text, label)
            scored = score_yes_no(model, tokenizer, prompt, device)
            row.update(_yesno_row_columns(label, scored))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_multilabel_d2(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    labels: Optional[Sequence[str]] = None,
    include_argmax: bool = True,
) -> pd.DataFrame:
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for text in texts:
        row = {"comment_text": text}
        prompt = build_d2_prompt(text)
        for label in labels:
            pos_tok, neg_tok = D2_LABEL_CONTINUATIONS[label]
            scored = score_fill_blank(model, tokenizer, prompt, device, pos_tok, neg_tok)
            row.update(_fillblank_row_columns(label, scored))
        if include_argmax:
            row.update(score_d2_argmax(model, tokenizer, text, device, labels))
        rows.append(row)
    return pd.DataFrame(rows)


def predict_toxic_d2(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
) -> pd.DataFrame:
    """Backward-compatible wrapper: toxic-only D2 columns."""
    full = predict_multilabel_d2(model, tokenizer, texts, device, labels=["toxic"], include_argmax=False)
    return full[
        [
            "comment_text",
            "d2_p_positive_toxic",
            "d2_pred_logprob_toxic",
            "d2_pred_generate_toxic",
            "d2_parse_abstain_toxic",
            "d2_raw_completion_toxic",
            "d2_scoring_mismatch_toxic",
        ]
    ].rename(
        columns={
            "d2_p_positive_toxic": "d2_p_toxic",
            "d2_pred_logprob_toxic": "d2_pred_logprob",
            "d2_pred_generate_toxic": "d2_pred_generate",
            "d2_parse_abstain_toxic": "d2_parse_abstain",
            "d2_raw_completion_toxic": "d2_raw_completion",
            "d2_scoring_mismatch_toxic": "d2_scoring_mismatch",
        }
    )


def _method_summary(
    compare: pd.DataFrame,
    prefix: str,
    methods: Sequence[Tuple[str, str]],
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for label in labels:
        gold = compare[label]
        for method_name, col_tpl in methods:
            pred_col = col_tpl.format(label)
            preds = compare[pred_col]
            if method_name == "generate_parse":
                valid = preds.notna()
                agreement = (gold[valid] == preds[valid]).mean() if valid.any() else float("nan")
                abstain_rate = (~valid).mean()
            else:
                agreement = (gold == preds).mean()
                abstain_rate = 0.0
            rows.append(
                {
                    "label": label,
                    "method": method_name,
                    "agreement": agreement,
                    "abstain_rate": abstain_rate,
                }
            )
    return pd.DataFrame(rows)


def d1_method_summary(compare: pd.DataFrame, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    methods = [
        ("logprob_full", "d1_pred_logprob_{}"),
        ("logprob_first_token", "d1_pred_logprob_first_{}"),
        ("constrained", "d1_pred_constrained_{}"),
        ("generate_parse", "d1_pred_generate_{}"),
    ]
    return _method_summary(compare, "d1", methods, labels)


def d2_method_summary(compare: pd.DataFrame, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    methods = [
        ("logprob_full", "d2_pred_logprob_{}"),
        ("logprob_first_token", "d2_pred_logprob_first_{}"),
        ("constrained", "d2_pred_constrained_{}"),
        ("generate_parse", "d2_pred_generate_{}"),
    ]
    return _method_summary(compare, "d2", methods, labels)


def d2_argmax_summary(compare: pd.DataFrame, labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """How often argmax label matches each gold label (diagnostic; mutually exclusive)."""
    labels = list(labels or TOXIC_LABELS)
    rows = []
    for gold_label in labels:
        mask = compare[gold_label] == 1
        if not mask.any():
            rows.append({"gold_label": gold_label, "argmax_rate_when_gold": float("nan"), "n_gold": 0})
            continue
        rate = (compare.loc[mask, "d2_argmax_label"] == gold_label).mean()
        rows.append({"gold_label": gold_label, "argmax_rate_when_gold": rate, "n_gold": int(mask.sum())})
    rows.append(
        {
            "gold_label": "any_toxic",
            "argmax_rate_when_gold": (
                compare.loc[compare[labels].any(axis=1), "d2_argmax_label"].isin(labels).mean()
                if (compare[labels].any(axis=1)).any()
                else float("nan")
            ),
            "n_gold": int(compare[labels].any(axis=1).sum()),
        }
    )
    return pd.DataFrame(rows)
