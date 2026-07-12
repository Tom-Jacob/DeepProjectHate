# Developer guide — toxic-comment audit

How the code is organized, the core abstraction, and how to run every experiment
**both from the CLI and programmatically** (Python / notebook). For results and
the narrative see [`PROGRESS.md`](PROGRESS.md); for project context see
[`../CLAUDE.md`](../CLAUDE.md).

---

## 1. Design in one idea

Every model — the BERT encoder, the non-neural baseline, and the DistilGPT-2
decoder — is reduced to a single method:

```python
scorer.predict_toxic_proba(texts) -> np.ndarray   # P(toxic) in [0, 1], one per comment
```

Because they share this contract, the comparison, fairness, robustness, and
masking code treat all architectures identically. Adding a model = adding one
class + one `ModelSpec`; every experiment then works with it for free.

There are **two evaluation levels**:
- **Binary, cross-architecture** (this half): the shared `predict_toxic_proba`
  harness — comparison / fairness / robustness / masking.
- **Multi-label agreement** (Tom's `evaluate.py`, `join_scores.py`, notebook 04):
  toxic-bert's 6 labels vs the decoder's 6 labels.

---

## 2. Module map

```
src/
  paths.py           constants: paths, SEED, TOXIC_LABELS, model ids
  device.py          get_device() — CUDA-usable probe, else CPU
  data.py            load_train_df / load_test_df, label stats        (pre-existing)
  splits.py          get_split_frame / sample_split_frame(split, n)   (pre-existing)

  model_registry.py  ★ the hub: ModelSpec, MODEL_SPECS / FALLBACK_SPECS /
                     DECODER_SPECS / ALL_SPECS, load_scorer(spec),
                     HFSeqClassScorer, DecoderToxicScorer
  baselines.py       TfidfSvmScorer (TF-IDF + calibrated LinearSVM)
  encoder.py         ToxicBertEncoder — 6-label toxic-bert            (pre-existing)
  decoder.py         DistilGPT-2 D1/D2 prompting                       (Tom's)

  fairness.py        identity templates, build_identity_probe_df, per_group_fpr
  perturb.py         PERTURBATIONS registry, apply_perturbation
  intervention.py    mask_terms / strip_terms, inference_ablation
  distill.py         soft KD toxic-bert → DistilBERT (6-label) + DistilBertStudentScorer
  evaluate.py        multi-label metrics + agreement                   (Tom's)
  plots.py           all figure helpers + save_figure

scripts/             CLI wrappers (incl. train_distill.py) — see §5
```

`★` = the file to read first; everything routes through `load_scorer`.
Registry keys include `distilbert-student` (`kind=student`) once
`models/distilbert-student/` exists.

---

## 3. The scorer interface and its three implementations

```python
from model_registry import load_scorer, ALL_SPECS, MODEL_SPECS

# By spec (recommended) — knows kind (hf / baseline / decoder):
scorer = load_scorer(ALL_SPECS["toxic-bert"])
probs = scorer.predict_toxic_proba(["you are an idiot", "have a nice day"])
# -> array([0.99, 0.001])
```

| key | class | notes |
|---|---|---|
| `toxic-bert` | `HFSeqClassScorer` | 6-label sigmoid, uses the `toxic` logit |
| `tfidf-svm` | `TfidfSvmScorer` | must `.fit(texts, y)` before scoring |
| `distilgpt2-d1` / `-d2` | `DecoderToxicScorer` | wraps `decoder.py`; truncates to 512 tok |
| `distilbert-tox`, `roberta-tox` | `HFSeqClassScorer` | fallback only (`FALLBACK_SPECS`) |

The non-neural baseline needs training first:

```python
from baselines import TfidfSvmScorer
from splits import get_split_frame

train = get_split_frame("train")
svm = TfidfSvmScorer().fit(train["comment_text"].tolist(), train["toxic"].values)
svm.predict_toxic_proba(["kill yourself"])   # -> array([0.9x])
```

---

## 4. Programmatic recipes (notebook-friendly)

Put this at the top of a notebook so `src/` is importable:
```python
import sys; sys.path.insert(0, "src")   # from the repo root
```

**Comparison metrics for any model:**
```python
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from splits import sample_split_frame
from model_registry import load_scorer, ALL_SPECS

test = sample_split_frame("test", n=500)
y = test["toxic"].values
p = load_scorer(ALL_SPECS["toxic-bert"]).predict_toxic_proba(test["comment_text"].tolist())
print("F1", f1_score(y, p >= 0.5), "PR-AUC", average_precision_score(y, p))
```

**Fairness (per-identity FPR):**
```python
from fairness import build_identity_probe_df, per_group_fpr
probe = build_identity_probe_df()
p = load_scorer(ALL_SPECS["toxic-bert"]).predict_toxic_proba(probe["text"].tolist())
per_group_fpr(probe, p).sort_values("fpr", ascending=False).head()
```

**Robustness (perturb + measure evasion):**
```python
from perturb import apply_perturbation
toxic = "you are a disgusting idiot"
print(apply_perturbation("leetspeak", toxic))   # y0u @r3 @ d1$gu$71ng 1d10t
```

**Masking intervention (why it over-flags identities):**
```python
from intervention import inference_ablation, mask_terms
print(mask_terms("I am gay"))                    # "I am [MASK]"
inference_ablation(load_scorer(ALL_SPECS["toxic-bert"]), build_identity_probe_df())
# -> {'fpr_with_identity': 0.15, 'fpr_masked': 0.0, ...}
```

**Encoder vs decoder agreement (multi-label, Tom's layer):**
```python
from encoder import load_toxic_bert
from decoder import predict_multilabel_d1
from evaluate import agreement_by_label
# see scripts/run_agreement.py for the full join + truncation
```

**Plot + save any figure:**
```python
import json
from plots import plot_model_comparison_bars, save_figure
res = json.load(open("data/processed/model_comparison.json", encoding="utf-8"))
fig, ax, df = plot_model_comparison_bars(res, split="test")
save_figure(fig, "report/figures/model_comparison_bars_test.png")
```

---

## 5. CLI reference

```powershell
python scripts\run_model_comparison.py --quick                     # or --eval-n N
python scripts\run_fairness.py
python scripts\run_robustness.py --quick
python scripts\run_masking.py
python scripts\train_distill.py --epochs 2             # Colab GPU preferred
python scripts\train_decoder_ft.py --max-train-samples 8000
python scripts\run_agreement.py --n 150
python scripts\plot_model_comparison.py --split test
python scripts\plot_fairness_robustness.py
```
Common flags: `--models <keys...>` (any of `ALL_SPECS`, including `distilbert-student`,
`distilgpt2-d1-ft`), `--eval-n` / `--n`
(sample size), `--target toxic|toxic_any`, `--threshold`. Add the decoder to any
run with `--models toxic-bert tfidf-svm distilgpt2-d1`.

---

## 6. Extending the code

**Add a model to the comparison** — one `ModelSpec` in `model_registry.py`:
```python
MODEL_SPECS["my-model"] = ModelSpec("my-model", "org/my-hf-model", "BERT", "in-domain")
```
If it needs a custom scoring path, add a class exposing `predict_toxic_proba`
and a branch in `load_scorer`.

**Add a perturbation** — a function `(text, **kw) -> str` registered in
`perturb.PERTURBATIONS`.

**Add identity terms** — extend `fairness.IDENTITY_GROUPS`
(`{term: category}`); the probe, FPR, and masking all pick it up automatically.

---

## 7. Gotchas

- **CPU only.** The RTX 5060 (sm_120) can't run torch 2.3/cu121; `device.py`
  falls back to CPU. Set `DEEPHATE_FORCE_CPU=1` to force. GPU training → Colab.
- **`PYTHONIOENCODING=utf-8`** — required (Hebrew path + cp1252 console).
- **Decoder is slow + context-limited.** DistilGPT-2 has a 1024-token context;
  `DecoderToxicScorer` truncates comments to 512 tokens. Keep decoder sample
  sizes small (`--eval-n`, `--n`).
- **Results JSON is gitignored** (`data/processed/`). Figures in `report/figures/`
  are committed. Re-run the script to regenerate a JSON.
- **Don't `2>$null` decoder runs** — it hides tracebacks; filter stderr instead.
