# Project progress guide — toxic-comment audit (encoder side)

A complete record of what has been built and how to run it, as of the current
work session. Focus is the **encoder / toxic-bert** half of the project (Omer).
For the living project context see [`../CLAUDE.md`](../CLAUDE.md).

> Note: some tables below reflect **earlier** runs (an initial pass that included
> RoBERTa / off-the-shelf DistilBERT, then a 3-architecture roster of toxic-bert ·
> TF-IDF+SVM · DistilGPT-2). The roster is now **5 model configurations**: the
> above three plus `bert-base-classifier` and `bert-base-mlm-prompt` — the same
> `bert-base-uncased` checkpoint used as a trained classifier vs. a zero-shot
> masked-LM prompt (see `CLAUDE.md`). For the **final** numbers and the full
> write-up see [`../report/report_draft.md`](../report/report_draft.md).

---

## 1. What the project is

A controlled **audit** of toxic-comment detectors on Jigsaw 2018, comparing
models across architecture, specialization, compression, **fairness**, and
**robustness** — not a "fine-tune BERT, report F1" project. Three hypotheses:

- **H1** — the specialized encoder (toxic-bert) wins accuracy/robustness but pays
  a fairness cost (higher identity-term false positives).
- **H2** — smaller/distilled models lose accuracy AND amplify identity bias.
- **H3** — decoder prompting (DistilGPT-2, Tom's half) is prompt-sensitive / less
  calibrated than encoder classifiers.

---

## 2. Environment (one-time)

- Conda env **`deep_learn`** (Python 3.8, torch 2.3.0+cu121).
  Interpreter: `C:\Users\omerr\miniconda3\envs\deep_learn\python.exe`
- Extra deps installed on top of the course env:
  `pip install -r requirements-project.txt`
  (transformers 4.46, datasets 3.1, huggingface_hub, pyarrow, scikit-learn, kaggle)

### Machine-specific gotchas (important)

- **GPU is unusable → everything runs on CPU.** The laptop's RTX 5060 is Blackwell
  (sm_120); torch 2.3.0+cu121 only supports up to sm_90, so CUDA kernels crash.
  `src/device.py` probes CUDA and auto-falls-back to CPU. Force it with env
  `DEEPHATE_FORCE_CPU=1`. GPU training (distillation) must run on Colab.
- **Always set `PYTHONIOENCODING=utf-8`** before running scripts — the repo path
  contains Hebrew characters and the Windows console (cp1252) otherwise crashes
  on `print()`.

---

## 3. Data & models (already downloaded locally; gitignored)

```powershell
python scripts\download_jigsaw.py       # Jigsaw CSVs (Kaggle or HF mirror fallback)
python scripts\download_toxic_bert.py   # unitary/toxic-bert weights (~418 MB)
python scripts\make_splits.py           # train/val/test index (135,635 / 23,936 / 63,978)
```

Off-the-shelf comparison models download automatically on first use:
`martin-ha/toxic-comment-model` (DistilBERT), `s-nlp/roberta_toxicity_classifier`.

---

## 4. What has been built (by phase)

### Phase 0 — setup & infra  ✅
Env, data, splits, CPU fallback (`src/device.py`), and a verified toxic-bert
inference pipeline.

Formal 6-label toxic-bert benchmark (`scripts/run_formal_eval.py`, quick=500/split):
| split | macro-F1 | micro-F1 | mean PR-AUC |
|---|:--:|:--:|:--:|
| val | 0.585 | 0.853 | 0.811 |
| test | 0.550 | 0.730 | 0.655 |
(`identity_hate` F1 = 0 with tiny support — motivates the fairness axis.)

### Phase 1 — fair model comparison  ✅
Every model reduced to a single **P(toxic)** and evaluated on the same binary
task, with parameter count + inference latency.
Files: `src/model_registry.py`, `src/baselines.py`, `scripts/run_model_comparison.py`,
`scripts/plot_model_comparison.py`. Figures: `report/figures/model_comparison_bars_test.png`,
`accuracy_latency_test.png`.

| model | family | params | test F1 | PR-AUC | ms/comment |
|---|---|--:|:--:|:--:|--:|
| toxic-bert | BERT | 109M | 0.770 | 0.843 | 210 |
| roberta-tox | RoBERTa | 125M | 0.769 | 0.869 | 213 |
| tfidf-svm | non-neural | 50k | 0.737 | 0.856 | 0.09 |
| distilbert-tox | DistilBERT | 67M | 0.619 | 0.701 | 106 |

Finding: TF-IDF+SVM nearly matches the transformers (beats toxic-bert on PR-AUC)
at ~2000× lower latency; the small transformer is weakest.

### Phase 2 — fairness: per-identity FPR  ✅
Neutral identity-mention sentences (all non-toxic) scored by every model; any
"toxic" prediction is a false positive.
Files: `src/fairness.py`, `scripts/run_fairness.py`, `scripts/plot_fairness_robustness.py`.
Figure: `report/figures/fairness_fpr_heatmap.png`.

| model | overall identity FPR | control FPR |
|---|:--:|:--:|
| roberta-tox | 8.6% | 0% |
| tfidf-svm | 8.6% | 0% |
| toxic-bert | 15.2% | 0% |
| distilbert-tox | 22.4% | 0% |

Finding: control FPR is 0% everywhere → false positives are specifically
identity-driven. "I am gay" is flagged by all models. Specialized toxic-bert is
less fair than the non-neural baseline (**H1**); the small model is worst (**H2**).

### Phase 3 — robustness: perturbation evasion  ✅
Genuinely-toxic comments perturbed (typos, leetspeak, spacing, repetition,
casing, punctuation); "evasion" = a flagged comment slipping through.
Files: `src/perturb.py`, `scripts/run_robustness.py`.
Figure: `report/figures/robustness_evasion_heatmap.png`.

| model | baseline | leetspeak evasion | space-out | punct-noise |
|---|:--:|:--:|:--:|:--:|
| toxic-bert | 0.92 | 0.28 | 0.03 | 0.15 |
| roberta-tox | 0.83 | 0.75 | 0.65 | 0.80 |
| tfidf-svm | 0.72 | 0.74 | 0.76 | 0.76 |
| distilbert-tox | 0.73 | 0.93 | 0.95 | 0.94 |

Finding: toxic-bert is by far the most robust; all others are trivially evaded by
obfuscation. So toxic-bert is **most robust but least fair** — a genuine trade-off.

---

## 4b. Cross-part compatibility (encoder ↔ decoder)

The evaluation runs at **two complementary levels**, and both work with Tom's
decoder code:

**Level 1 — binary P(toxic), cross-architecture (this half's harness).**
Every model — encoders, the non-neural baseline, AND Tom's DistilGPT-2 — exposes
one method, `predict_toxic_proba(texts) -> P(toxic)[]`. The decoder plugs in via
a thin adapter (`DecoderToxicScorer` in `src/model_registry.py`) that calls Tom's
`decoder.py` functions unchanged and reads the 'toxic' label (`d1` yes/no or `d2`
fill-in-blank). So the decoder appears in the same comparison / fairness /
robustness tables and figures as the encoders — just add it via `--models`:

```powershell
python scripts\run_model_comparison.py --models toxic-bert distilgpt2-d1 distilgpt2-d2 --eval-n 40
python scripts\run_fairness.py   --models toxic-bert distilgpt2-d1
python scripts\run_robustness.py --models toxic-bert distilgpt2-d1 --n 60
```
Registry keys: encoders/baseline in `MODEL_SPECS`, decoder in `DECODER_SPECS`,
union in `ALL_SPECS`. The decoder is not in the default set (per-comment
generation is slow on CPU) — request it explicitly, and use `--eval-n` to cap.

**Level 2 — multi-label agreement (Tom's pre-existing infra).**
`src/evaluate.py` (`encoder_decoder_agreement`, `agreement_by_label`),
`src/join_scores.py`, and `notebooks/04_...` compare toxic-bert's 6-label output
(`enc_p_{label}`) against the decoder's 6-label output (`d1_*` / `d2_*`). This is
the deeper per-label comparison; the binary baselines don't join here. Untouched
by this half's work.

Sanity check (n=40): toxic-bert F1 ≈ 0.82 vs DistilGPT-2 prompting ≈ 0.28 with
chance-level ROC-AUC — decoder prompting is far weaker at this scale (**H3**).

## 5. How to run everything

```powershell
conda activate deep_learn
cd "...\פרויקט\DeepProjectHate"
$env:PYTHONIOENCODING = "utf-8"

python scripts\run_formal_eval.py --quick            # toxic-bert 6-label benchmark
python scripts\run_model_comparison.py --quick       # multi-model fair comparison
python scripts\plot_model_comparison.py --split test # -> report/figures/
python scripts\run_fairness.py                       # per-identity FPR
python scripts\run_robustness.py --quick             # perturbation evasion
python scripts\plot_fairness_robustness.py           # -> report/figures/
```
Drop `--quick` for full runs (slower on CPU). Restrict models with e.g.
`--models toxic-bert tfidf-svm`.

---

## 6. Repo structure (new / key files)

```
src/
  device.py           # CUDA-usable probe + CPU fallback
  model_registry.py   # unified P(toxic) scorer over heterogeneous models
  baselines.py        # TF-IDF + calibrated LinearSVM
  fairness.py         # identity templates + per-group FPR
  perturb.py          # text perturbations (typos, leetspeak, ...)
  encoder.py          # toxic-bert 6-label wrapper (pre-existing)
  evaluate.py, plots.py, data.py, splits.py, paths.py
scripts/
  run_formal_eval.py, run_model_comparison.py, run_fairness.py, run_robustness.py
  plot_model_comparison.py, plot_fairness_robustness.py
  download_*.py, make_splits.py
report/figures/       # generated PNGs for the report
data/processed/       # results JSON + splits (gitignored)
```

---

## 7. Status & what's left

Done: Phase 0 (setup), 1 (fair comparison), 2 (fairness), 3 (robustness), 5
(masking intervention — both inference-time ablation across all models and
SVM training-time retrain), 7 (encoder-vs-decoder agreement, disagreement case
studies). Phase 6 (decoder) is Tom's, plugged in via `DecoderToxicScorer`.

New: **Phase 5b — same-model classifier vs. prompting.** `bert-base-uncased`
fine-tuned as a classifier (`scripts/train_bert_base_classifier.py`) and used
zero-shot via masked-LM prompting (`BertBaseMlmPromptScorer` in
`model_registry.py`). Restores the specialization axis (vs. toxic-bert) and
directly tests "dedicated classifier vs. LM prompting" on a matched checkpoint.
Both wired into the comparison/fairness/robustness/masking scripts; pairwise
agreement in `scripts/run_bertbase_agreement.py`.

Not done (explicit future work, not by oversight):
- **hateBERT** cross-domain encoder and a **real distillation** run
  (toxic-bert → DistilBERT) — no usable local GPU; would need Colab.
- **Few-shot** decoder prompting (only zero-shot D1/D2 implemented).
- **ECE / calibration curves** — not implemented for any model.
- **Neural masked fine-tune** — the training-time masking intervention only
  retrains the TF-IDF+SVM; no transformer has been fine-tuned with identity
  terms masked.
- **Phase 8** deliverables (slides + video) — not started; report/README are
  in progress.

Consolidated narrative for the report: specialization (toxic-bert) buys accuracy
and robustness but costs fairness; the same base model (bert-base) is far weaker
as a zero-shot prompt than as a trained classifier; a non-neural baseline is a
shockingly strong, fast floor; identity bias is data-driven (masking erases it)
rather than architecture-specific.
