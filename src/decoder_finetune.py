"""Fine-tune DistilGPT-2 on D1 yes/no completions (binary toxic).

Loss is applied only on the answer tokens (`` yes`` / `` no``), matching the
zero-shot ``score_yes_no`` scoring path used at eval time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from decoder import build_d1_prompt
from device import get_device
from paths import (
    DISTILGPT2_MODEL_ID,
    LOCAL_DISTILGPT2_D1_FT_DIR,
    LOCAL_DISTILGPT2_DIR,
    SEED,
)

YES_CONT = " yes"
NO_CONT = " no"
MAX_PROMPT_TOKENS = 480  # leave room for answer inside 1024 context


@dataclass
class DecoderFTConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 2
    batch_size: int = 8
    warmup_ratio: float = 0.06
    max_train_samples: int = 8000
    max_val_samples: int = 1000
    seed: int = SEED
    base_model: str = DISTILGPT2_MODEL_ID
    output_dir: str = ""
    max_prompt_tokens: int = MAX_PROMPT_TOKENS

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_base(base_model: str) -> str:
    if base_model in (DISTILGPT2_MODEL_ID, "distilgpt2") and LOCAL_DISTILGPT2_DIR.exists():
        return str(LOCAL_DISTILGPT2_DIR)
    if Path(base_model).exists():
        return base_model
    return base_model


def truncate_comment(tokenizer, text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """Truncate comment so the full D1 prompt fits with room for the answer."""
    # Rough: encode prompt with empty comment budget by truncating comment alone.
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Keep head of comment; prompt template adds ~30 tokens.
    budget = max(32, max_tokens - 40)
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget])


def build_example_text(comment: str, toxic: int, tokenizer) -> Tuple[str, str]:
    comment = truncate_comment(tokenizer, comment)
    prompt = build_d1_prompt(comment, "toxic")
    answer = YES_CONT if int(toxic) == 1 else NO_CONT
    return prompt, answer


class YesNoLMDataset(Dataset):
    """Causal LM examples: labels = -100 on prompt, answer token ids on continuation."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.examples: List[dict] = []
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")

        for text, y in zip(texts, labels):
            prompt, answer = build_example_text(text, int(y), tokenizer)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if not answer_ids:
                continue
            # Truncate prompt from the left (keep Answer: suffix) if still too long.
            max_len = 1024
            total = prompt_ids + answer_ids
            if len(total) > max_len:
                overflow = len(total) - max_len
                prompt_ids = prompt_ids[overflow:]
                total = prompt_ids + answer_ids
            labels_ids = [-100] * len(prompt_ids) + list(answer_ids)
            self.examples.append(
                {
                    "input_ids": total,
                    "labels": labels_ids,
                    "attention_mask": [1] * len(total),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_pad(batch: List[dict], pad_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def eval_answer_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    """Fraction of examples where argmax next-token after prompt matches gold answer.

    Approximated via teacher-forcing: check whether loss on answer tokens is lower
    for the gold path by comparing predicted token at first answer position.
    """
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # For each seq, find first non -100 label position; predict from previous logit
        for b in range(input_ids.size(0)):
            ans_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(ans_pos) == 0:
                continue
            pos = int(ans_pos[0].item())
            pred = int(logits[b, pos - 1].argmax().item()) if pos > 0 else int(logits[b, 0].argmax().item())
            gold = int(labels[b, pos].item())
            correct += int(pred == gold)
            total += 1
    return float(correct) / max(total, 1)


def train_decoder_ft(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    cfg: DecoderFTConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir or LOCAL_DISTILGPT2_D1_FT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cap(texts, labels, n):
        texts, labels = list(texts), list(labels)
        if n is None or n >= len(texts):
            return texts, labels
        rng = np.random.RandomState(cfg.seed)
        # Stratified-ish: half toxic if possible
        labels_arr = np.asarray(labels, dtype=int)
        pos = np.where(labels_arr == 1)[0]
        neg = np.where(labels_arr == 0)[0]
        n_pos = min(len(pos), n // 2)
        n_neg = min(len(neg), n - n_pos)
        idx = np.concatenate(
            [
                rng.choice(pos, n_pos, replace=False) if n_pos else np.array([], dtype=int),
                rng.choice(neg, n_neg, replace=False) if n_neg else np.array([], dtype=int),
            ]
        )
        rng.shuffle(idx)
        return [texts[i] for i in idx], [int(labels[i]) for i in idx]

    train_texts, train_labels = _cap(train_texts, train_labels, cfg.max_train_samples)
    val_texts, val_labels = _cap(val_texts, val_labels, cfg.max_val_samples)

    base = _resolve_base(cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base).to(device)

    train_ds = YesNoLMDataset(train_texts, train_labels, tokenizer, cfg.max_prompt_tokens)
    val_ds = YesNoLMDataset(val_texts, val_labels, tokenizer, cfg.max_prompt_tokens)
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    history = []
    best_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running_loss += float(loss.detach())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        val_acc = eval_answer_accuracy(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_answer_acc": val_acc}
        history.append(row)
        print(f"epoch {epoch+1}/{cfg.epochs}  loss={train_loss:.4f}  val_answer_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    cfg.output_dir = str(out_dir)
    meta = {
        "config": cfg.to_dict(),
        "base_model": base,
        "best_val_answer_acc": best_acc,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "history": history,
        "task": "binary_toxic_d1_yes_no",
    }
    with open(out_dir / "decoder_ft_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
