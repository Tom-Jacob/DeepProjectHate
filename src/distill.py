"""Soft knowledge distillation: toxic-bert (teacher) → DistilBERT (6-label student)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from device import get_device
from paths import (
    LOCAL_DISTILBERT_STUDENT_DIR,
    LOCAL_MODEL_DIR,
    MODEL_ID,
    SEED,
    STUDENT_INIT_ID,
    TOXIC_LABELS,
)

MAX_LENGTH = 256


@dataclass
class DistillConfig:
    temperature: float = 2.0
    alpha: float = 0.7  # weight on KD vs hard BCE
    beta: float = 0.0   # weight on soft-F1 regularization (0 = off)
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2
    batch_size: int = 16
    max_length: int = MAX_LENGTH
    warmup_ratio: float = 0.06
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    seed: int = SEED
    teacher_id: str = MODEL_ID
    student_init: str = STUDENT_INIT_ID
    output_dir: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CommentLabelDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def soft_f1_loss(
    student_logits: torch.Tensor,
    hard_labels: torch.Tensor,
) -> torch.Tensor:
    """Differentiable macro F1 loss: 1 - mean(soft_F1 per label).

    Uses sigmoid probabilities as soft predictions so the loss is smooth and
    back-propagable, unlike the hard-threshold F1 metric.
    """
    eps = 1e-7
    probs = torch.sigmoid(student_logits)
    tp = (hard_labels * probs).sum(dim=0)
    fp = ((1.0 - hard_labels) * probs).sum(dim=0)
    fn = (hard_labels * (1.0 - probs)).sum(dim=0)
    soft_f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return 1.0 - soft_f1.mean()


def kd_bce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
    beta: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Soft KD (sigmoid KL at temperature T) + hard multilabel BCE + optional soft-F1."""
    t = temperature
    student_soft = torch.sigmoid(student_logits / t)
    teacher_soft = torch.sigmoid(teacher_logits / t).detach()
    eps = 1e-7
    student_soft = student_soft.clamp(eps, 1 - eps)
    teacher_soft = teacher_soft.clamp(eps, 1 - eps)
    kd = (
        teacher_soft * (teacher_soft.log() - student_soft.log())
        + (1 - teacher_soft) * ((1 - teacher_soft).log() - (1 - student_soft).log())
    ).mean()
    kd = kd * (t * t)
    hard = F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    loss = alpha * kd + (1.0 - alpha) * hard
    parts = {
        "kd": float(kd.detach()),
        "hard": float(hard.detach()),
    }
    if beta > 0:
        f1_reg = soft_f1_loss(student_logits, hard_labels)
        loss = loss + beta * f1_reg
        parts["f1_reg"] = float(f1_reg.detach())
    parts["loss"] = float(loss.detach())
    return loss, parts


def build_student(student_init: str = STUDENT_INIT_ID, num_labels: int = 6):
    id2label = {i: name for i, name in enumerate(TOXIC_LABELS)}
    label2id = {name: i for i, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        student_init,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(student_init)
    return model, tokenizer


def load_teacher(device: torch.device, teacher_id: Optional[str] = None):
    path = teacher_id or (str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, path


def multilabel_f1_np(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    y_pred = (y_prob >= thr).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_student(student, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    from sklearn.metrics import f1_score

    student.eval()
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for batch in loader:
        enc = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
        logits = student(**enc).logits
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    y_prob = np.vstack(probs_all)
    y_true = np.vstack(labels_all)
    metrics = multilabel_f1_np(y_true, y_prob)
    toxic_idx = TOXIC_LABELS.index("toxic")
    metrics["toxic_f1"] = float(
        f1_score(y_true[:, toxic_idx], (y_prob[:, toxic_idx] >= 0.5).astype(int), zero_division=0)
    )
    return metrics


def train_distill(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    val_texts: Sequence[str],
    val_labels: np.ndarray,
    cfg: DistillConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Train student with soft KD; save to cfg.output_dir."""
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.output_dir or LOCAL_DISTILBERT_STUDENT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.max_train_samples and len(train_texts) > cfg.max_train_samples:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(train_texts), cfg.max_train_samples, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]
    if cfg.max_val_samples and len(val_texts) > cfg.max_val_samples:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(val_texts), cfg.max_val_samples, replace=False)
        val_texts = [val_texts[i] for i in idx]
        val_labels = val_labels[idx]

    teacher, _, teacher_path = load_teacher(device, cfg.teacher_id)
    student, student_tok = build_student(cfg.student_init)
    student = student.to(device)

    val_ds = CommentLabelDataset(val_texts, val_labels, student_tok, cfg.max_length)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    teacher_tok = AutoTokenizer.from_pretrained(
        str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else (cfg.teacher_id or MODEL_ID)
    )

    class TeacherAlignedLoader:
        def __init__(
            self,
            texts,
            labels,
            student_tok,
            teacher,
            teacher_tok,
            batch_size,
            max_length,
            device,
        ):
            self.texts = list(texts)
            self.labels = labels
            self.student_tok = student_tok
            self.teacher = teacher
            self.teacher_tok = teacher_tok
            self.batch_size = batch_size
            self.max_length = max_length
            self.device = device
            self.n = len(self.texts)

        def __iter__(self):
            order = np.random.permutation(self.n)
            for start in range(0, self.n, self.batch_size):
                idxs = order[start : start + self.batch_size]
                batch_texts = [self.texts[i] for i in idxs]
                hard = torch.tensor(self.labels[idxs], dtype=torch.float32, device=self.device)
                s_enc = self.student_tok(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # DistilBERT student rejects token_type_ids (BertTokenizer may emit them).
                s_enc = {
                    k: v.to(self.device)
                    for k, v in s_enc.items()
                    if k in ("input_ids", "attention_mask")
                }
                t_enc = self.teacher_tok(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                t_enc = {k: v.to(self.device) for k, v in t_enc.items()}
                with torch.no_grad():
                    t_logits = self.teacher(**t_enc).logits
                yield s_enc, t_logits, hard

        def __len__(self):
            return (self.n + self.batch_size - 1) // self.batch_size

    train_iter = TeacherAlignedLoader(
        train_texts,
        train_labels,
        student_tok,
        teacher,
        teacher_tok,
        cfg.batch_size,
        cfg.max_length,
        device,
    )

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_iter) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    history = []
    best_macro = -1.0
    for epoch in range(cfg.epochs):
        student.train()
        tracked = ["kd", "hard", "loss"] + (["f1_reg"] if cfg.beta > 0 else [])
        running = {k: 0.0 for k in tracked}
        running["n"] = 0
        for s_enc, t_logits, hard in train_iter:
            opt.zero_grad()
            s_logits = student(**s_enc).logits
            loss, parts = kd_bce_loss(s_logits, t_logits, hard, cfg.temperature, cfg.alpha, cfg.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in tracked:
                running[k] += parts.get(k, 0.0)
            running["n"] += 1
        n = max(running["n"], 1)
        train_stats = {k: running[k] / n for k in tracked}
        val_metrics = evaluate_student(student, val_loader, device)
        row = {"epoch": epoch + 1, **train_stats, **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch+1}/{cfg.epochs}  loss={train_stats['loss']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.3f}  toxic_f1={val_metrics['toxic_f1']:.3f}"
        )
        if val_metrics["macro_f1"] >= best_macro:
            best_macro = val_metrics["macro_f1"]
            student.save_pretrained(out_dir)
            student_tok.save_pretrained(out_dir)

    cfg.output_dir = str(out_dir)
    meta = {
        "config": cfg.to_dict(),
        "teacher_path": teacher_path,
        "best_val_macro_f1": best_macro,
        "history": history,
        "labels": TOXIC_LABELS,
    }
    with open(out_dir / "distill_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class DistilBertStudentScorer:
    """6-label DistilBERT student with binary ``predict_toxic_proba`` for the harness."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        max_length: int = MAX_LENGTH,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        path = model_path or str(LOCAL_DISTILBERT_STUDENT_DIR)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Student weights not found at {path}. "
                "Train on Colab: notebooks/09_distillation_colab.ipynb "
                "or: python scripts/train_distill.py --max-train-samples 2000"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.model_id = "distilbert-student"
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.toxic_index = TOXIC_LABELS.index("toxic")

    @staticmethod
    def _batch_to_device(enc, device: torch.device) -> dict:
        # Student may be saved with a BertTokenizer that emits token_type_ids;
        # DistilBertForSequenceClassification rejects that kwarg.
        return {
            k: v.to(device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask")
        }

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, len(TOXIC_LABELS)), dtype=float)
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = self._batch_to_device(enc, self.device)
            probs = torch.sigmoid(self.model(**enc).logits).cpu().numpy()
            chunks.append(probs)
        return np.vstack(chunks)

    @torch.no_grad()
    def predict_toxic_proba(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self.predict_proba(texts, batch_size=batch_size)[:, self.toxic_index]
