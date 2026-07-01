"""DistilGPT-2 prompting helpers for binary label scoring (D1 + D2)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from paths import TOXIC_LABELS


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


def _softmax_two(log_a: float, log_b: float) -> Tuple[float, float]:
    logits = torch.tensor([log_a, log_b])
    probs = F.softmax(logits, dim=0)
    return probs[0].item(), probs[1].item()


def _parse_yes_no(text: str) -> int:
    text = text.strip().lower()
    if re.search(r"\byes\b", text):
        return 1
    if re.search(r"\bno\b", text):
        return 0
    return 0


@dataclass
class YesNoScore:
    p_yes: float
    p_no: float
    pred_logprob: int
    pred_generate: int
    disagree: bool
    generated_text: str


@dataclass
class FillBlankScore:
    p_positive: float
    p_negative: float
    pred_logprob: int
    pred_generate: int
    disagree: bool
    generated_text: str


def score_yes_no(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    yes_token: str = " yes",
    no_token: str = " no",
    max_new_tokens: int = 5,
) -> YesNoScore:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    yes_ids = tokenizer.encode(yes_token, add_special_tokens=False)
    no_ids = tokenizer.encode(no_token, add_special_tokens=False)

    log_yes = _continuation_logprob(model, prompt_ids, yes_ids, device)
    log_no = _continuation_logprob(model, prompt_ids, no_ids, device)
    p_yes, p_no = _softmax_two(log_yes, log_no)
    pred_logprob = int(p_yes >= p_no)

    input_tensor = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, len(prompt_ids) :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred_generate = _parse_yes_no(generated_text)

    return YesNoScore(
        p_yes=p_yes,
        p_no=p_no,
        pred_logprob=pred_logprob,
        pred_generate=pred_generate,
        disagree=pred_logprob != pred_generate,
        generated_text=generated_text,
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
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos_ids = tokenizer.encode(positive_token, add_special_tokens=False)
    neg_ids = tokenizer.encode(negative_token, add_special_tokens=False)

    log_pos = _continuation_logprob(model, prompt_ids, pos_ids, device)
    log_neg = _continuation_logprob(model, prompt_ids, neg_ids, device)
    p_pos, p_neg = _softmax_two(log_pos, log_neg)
    pred_logprob = int(p_pos >= p_neg)

    input_tensor = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, len(prompt_ids) :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    gen_lower = generated_text.strip().lower()
    if "toxic" in gen_lower and "not" not in gen_lower.split()[0:1]:
        pred_generate = 1
    elif gen_lower.startswith("not") or "not toxic" in gen_lower:
        pred_generate = 0
    else:
        pred_generate = pred_logprob

    return FillBlankScore(
        p_positive=p_pos,
        p_negative=p_neg,
        pred_logprob=pred_logprob,
        pred_generate=pred_generate,
        disagree=pred_logprob != pred_generate,
        generated_text=generated_text,
    )


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
            row[f"d1_p_yes_{label}"] = scored.p_yes
            row[f"d1_pred_logprob_{label}"] = scored.pred_logprob
            row[f"d1_pred_generate_{label}"] = scored.pred_generate
            row[f"d1_disagree_{label}"] = scored.disagree
            row[f"d1_generated_{label}"] = scored.generated_text
        rows.append(row)
    return pd.DataFrame(rows)


def predict_toxic_d2(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for text in texts:
        prompt = build_d2_prompt(text)
        scored = score_fill_blank(model, tokenizer, prompt, device)
        rows.append(
            {
                "comment_text": text,
                "d2_p_toxic": scored.p_positive,
                "d2_pred_logprob": scored.pred_logprob,
                "d2_pred_generate": scored.pred_generate,
                "d2_disagree": scored.disagree,
                "d2_generated": scored.generated_text,
            }
        )
    return pd.DataFrame(rows)
