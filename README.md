# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

Technion **EE046211 Deep Learning** course project.
Authors: **Omer Ramon**, **Tom Jacob**.

A controlled **audit** of toxic-comment detectors on the Jigsaw 2018 dataset. The
question is not "which model has the best F1" but **which kind of error each
architecture makes, and why**. Eight model configurations are compared on the same
data, through the same interface, on the same hardware, then stress-tested for
identity bias, robustness to obfuscation, and the mechanism behind their errors.

- **Report:** [`report/report.tex`](report/report.tex) — the submission document.
- **Narrative + all numbers:** [`report/report_draft.md`](report/report_draft.md).
- **Presentation:** [`report/PRESENTATION_OUTLINE.md`](report/PRESENTATION_OUTLINE.md).

---

## 1. The models, and exactly where each one came from

Every model is reduced to a single scalar `P(toxic)` (see §4), so all eight are
directly comparable regardless of architecture.

| ID | Model | Where the weights come from | Who trained it | How to obtain it |
|---|---|---|---|---|
| **E2** | `toxic-bert` (109M) | [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) on HuggingFace — Unitary's *Detoxify* release, already fine-tuned on the full Jigsaw dataset | Unitary (third party) | `python scripts/download_toxic_bert.py` → `models/toxic-bert/` (~418 MB) |
| **E1** | `bert-base-classifier` (109M) | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) on HuggingFace (Devlin et al.), then **fine-tuned by us** with a binary classification head | **Us** (this repo, CPU) | `python scripts/train_bert_base_classifier.py` → `models/bert-base-toxic/` |
| **E1-p** | `bert-base-mlm-prompt` (109M) | The **identical** `bert-base-uncased` checkpoint, used through its own pretrained masked-LM head | Nobody — zero-shot, no training | Nothing to train; downloaded on first use |
| **E0** | `tfidf-svm` (~50k features) | No pretrained weights at all — scikit-learn TF-IDF + `LinearSVC` | **Us**, retrained from scratch on every run | Automatic (fits in seconds inside each script) |
| **D1** | `distilgpt2-d1` (82M) | [`distilgpt2`](https://huggingface.co/distilgpt2) on HuggingFace (HuggingFace's distilled GPT-2), prompted yes/no | Nobody — zero-shot | `python scripts/download_distilgpt2.py` |
| **D2** | `distilgpt2-d2` (82M) | Same `distilgpt2` weights, different (fill-in-blank) prompt | Nobody — zero-shot | same as D1 |
| **D1-ft** | `distilgpt2-d1-ft` (82M) | `distilgpt2`, then **fine-tuned by us** on the D1 prompt's answer tokens | **Us** | `python scripts/train_decoder_ft.py` → `models/distilgpt2-d1-ft/` |
| **S1** | `distilbert-student` (67M) | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) initialization, then **distilled by us** from E2 (soft KD) | **Us** | `python scripts/train_distill.py` → `models/distilbert-student/` |

**Dataset:** the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
(Kaggle, 2018) — real Wikipedia talk-page comments with six binary labels. Fetched by
`python scripts/download_jigsaw.py` (via Kaggle if credentials are configured,
otherwise from the `Heliosoph/Jigsaw-Toxic-Comments` HuggingFace mirror, which needs
no account).

> **Model weights and data are gitignored** — they are hundreds of MB and are not in
> this repository. Every one of them is reproducible from the commands above.
> `python -c "import sys; sys.path.insert(0,'src'); import roster; roster.print_roster_status()"`
> reports which models are actually present locally.

### Why this particular roster

The comparison is built around **controlled pairs**, not a leaderboard:

- **E1 vs. E1-p** — the *same pretrained weights*, one with a trained classification
  head, one prompted zero-shot. Architecture and pretraining are held fixed, so the
  gap between them is attributable to the **method alone**. This is the cleanest test
  in the project.
- **E1 vs. E2** — the same architecture and parameter count, differing only in the
  *scale and quality of supervised fine-tuning* (ours: 12k rows on CPU; Unitary's:
  the full dataset on GPU).
- **E0 vs. everything** — a non-neural floor. Whatever E0 achieves is by construction
  achievable with no contextual understanding at all.
- **D1 vs. D2** — same decoder, same weights, different prompt wording: a direct
  measure of prompt sensitivity.
- **D1 vs. D1-ft** — separates "decoders can't do this" from "this decoder was never
  told what to do."
- **E2 vs. S1** — does compressing a model amplify its bias?

---

## 2. Quick start

```bash
conda activate deep_learn          # Python 3.8, torch 2.3.0+cu121
pip install -r requirements-project.txt

python scripts/download_jigsaw.py       # dataset  → data/raw/jigsaw/
python scripts/download_toxic_bert.py   # E2       → models/toxic-bert/
python scripts/download_distilgpt2.py   # D1/D2    → cached by transformers
python scripts/make_splits.py           # splits   → data/processed/split_indices.parquet
```

Splits: **135,635 train / 23,936 val** (85/15 stratified from `train.csv`) and
**63,978 test** (the officially-labeled rows of `test.csv`).

> **Always set `PYTHONIOENCODING=utf-8`.** The Windows console is cp1252 and the
> repository path contains Hebrew characters, so `print()`-ing a path crashes
> without it.

### Training the models we train ourselves

```bash
python scripts/train_bert_base_classifier.py        # E1  (~1.5-2 h on CPU)
python scripts/train_distill.py                     # S1  (GPU recommended)
python scripts/train_decoder_ft.py                  # D1-ft (GPU recommended)
```

All three accept `--max-train-samples` / `--train-n`, `--epochs` and `--batch-size`,
so they can be run at a reduced, CPU-friendly scale. Each checkpoints after every
epoch, so an interrupted run is not lost.

### Running the experiments

```bash
python scripts/run_model_comparison.py --eval-n 800   # accuracy, PR-AUC, ROC-AUC, latency
python scripts/run_fairness.py                        # per-identity FPR (368-sentence probe)
python scripts/run_robustness.py --n 100              # evasion + false-positive induction
python scripts/run_masking.py                         # masking intervention (the "why")
python scripts/run_agreement.py --n 150               # toxic-bert vs. D1-ft, 6 labels
python scripts/run_bertbase_agreement.py              # E1 vs. E1-p (same checkpoint)
python scripts/plot_model_comparison.py               # regenerate figures
python scripts/plot_fairness_robustness.py            # regenerate figures
```

Every script takes `--models` to restrict the roster (e.g.
`--models toxic-bert tfidf-svm`), skips models whose weights are missing, and
**merges** into the existing results JSON rather than overwriting it — so you can add
one model without re-scoring the rest. (`run_agreement.py` is the one exception —
it's a fixed encoder-vs-`distilgpt2-d1-ft` comparison, so it needs
`models/distilgpt2-d1-ft/` specifically; it prints `[SKIP]` and exits cleanly if that
Colab-only checkpoint isn't present, rather than crashing.)

Results land in `data/processed/*.json`; figures in `report/figures/`.

> **Runtime, roughly, on CPU:** `run_model_comparison.py` and
> `run_bertbase_agreement.py` score at most a few thousand rows per model and finish in
> minutes. `run_fairness.py` and `run_masking.py` run against the fixed 368-sentence
> identity probe, so their cost barely depends on `--eval-n`. `run_robustness.py`
> scores every sampled comment once per perturbation type, so it scales with `--n` ×
> (number of perturbations) — start with a small `--n`. Whichever script includes a
> DistilGPT-2 decoder is the slow one (~650 ms/comment on CPU, see §5); reduce its
> `--n`/`--eval-n` first if a run is taking too long.

---

## 3. Headline findings

1. **Toxicity detection is substantially lexical, but not purely so.** A 50k-feature
   linear SVM reaches ~93% of toxic-bert's F1 at ~1750× lower latency. Scale buys a
   real but *bounded* gain, not a transformative one.
2. **The encoder's real advantage is robustness — and it is earned, not
   architectural.** toxic-bert resists leetspeak evasion (0.280) far better than the
   SVM (0.711), but our own `bert-base-classifier` — identical architecture and size —
   is nearly as brittle (0.719). The advantage tracks toxic-bert's specific
   large-scale fine-tuning, not "being a BERT."
3. **A trained head beats prompting, shown most cleanly on one checkpoint.** On the
   identical `bert-base-uncased` weights, the trained head dominates zero-shot
   prompting on every metric (F1 0.675 vs. 0.000). The zero-shot F1 of 0.000 is a
   *calibration* artifact, not absence of signal — its ROC-AUC of 0.722 sits far above
   the GPT-2 decoders' ≈0.4.
4. **Identity bias is data-driven, but its mechanism is not universal.** Masking the
   identity token erases the bias for toxic-bert (0.087 → 0.000) and the SVM
   (0.049 → 0.000) — a clean lexical shortcut — but *increases* it for our generic
   classifier (0.103 → 0.188). A debiasing fix validated on one model cannot be
   assumed to transfer to another.
5. **Robustness is direction-dependent.** The same perturbations that barely help a
   toxic comment evade toxic-bert will get a *neutral* comment falsely flagged
   65–91% of the time (vs. an 11% baseline). "Robust" is not a single property.
6. **Compression did not cost fairness.** The soft-KD student matches its teacher's
   toxic F1 at lower latency *and* lower identity FPR — partially rejecting the
   hypothesis that compression amplifies bias.

---

## 4. How the code is organized

The whole comparison rests on one abstraction: **every model exposes
`predict_toxic_proba(texts) -> P(toxic)`**. A BERT classifier reads a sigmoid, a
masked-LM prompt softmaxes two vocabulary logits, a decoder compares yes/no
log-probs, and the SVM applies Platt scaling — but every evaluation script sees the
same interface, which is what makes an architecture-agnostic audit possible.

| Path | What it does |
|---|---|
| `src/model_registry.py` | The hub: `ModelSpec`, `MODEL_SPECS`/`DECODER_SPECS`/`ALL_SPECS`, `load_scorer`, and the scorer classes for each family |
| `src/roster.py` | `model_ready()` / `print_roster_status()` — which models have local weights |
| `src/encoder.py` | toxic-bert 6-label inference |
| `src/baselines.py` | `TfidfSvmScorer`, the non-neural floor |
| `src/decoder.py` | DistilGPT-2 prompting (D1/D2) |
| `src/distill.py` | Soft-KD distillation (E2 → S1) and the FPR-regularized second stage |
| `src/decoder_finetune.py` | D1 answer-token fine-tune |
| `src/fairness.py` | The 368-sentence identity probe (23 groups × 16 templates, 3 length tiers) |
| `src/perturb.py` | The six obfuscations (leetspeak, space-out, punct-noise, typos, repeats, caps) |
| `src/intervention.py` | Identity masking, at inference and training time |
| `src/evaluate.py` | Metrics: per-label F1/PR-AUC, macro/micro, model-vs-model agreement |
| `src/plots.py` | All figure generation |
| `src/data.py`, `src/splits.py`, `src/paths.py` | Loading, splitting, constants |
| `scripts/` | One CLI entry point per download / training run / experiment / plot |
| `notebooks/` | Exploration and Colab runs (01 EDA, 02 decoder smoke, 03–04 comparisons, 09–10 Colab training, 11–16 per-family audits) |

See [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for the programmatic API and
a walk-through of each experiment.

---

## 5. Environment notes (read this before debugging)

- **Conda env `deep_learn`** — Python 3.8, `torch 2.3.0+cu121`, transformers 4.46.3.
- **The GPU does not work here, by design of the version pin.** The machine has an
  RTX 5060 Laptop (Blackwell, `sm_120`); torch 2.3.0+cu121 only supports up to
  `sm_90`, so CUDA kernels fail with "no kernel image available". `src/device.py`
  probes CUDA with a real operation and **falls back to CPU automatically**. Force it
  with `DEEPHATE_FORCE_CPU=1`. For GPU work, use Colab (notebooks 09, 10, 15) rather
  than upgrading the shared course environment.
- Because everything runs on CPU, the decoders are slow (~650 ms/comment). This is
  why headline numbers use a seeded 800-comment test subset rather than all 63,978
  rows — every model is scored under identical conditions, which matters more for an
  audit than raw sample size.
- **If you *do* have a working CUDA GPU**, nothing needs to change: `src/device.py`
  probes CUDA with a real op and uses the GPU automatically whenever that probe
  succeeds. The CPU fallback here is specific to this machine's unsupported
  `sm_120` architecture, not to the `torch==2.3.0+cu121` pin itself — the same install
  uses the GPU normally on any `sm_90`-or-below card (which covers most Colab/cloud
  GPUs and the RTX 30/40-series).

---

## 6. Troubleshooting / FAQ

- **`download_jigsaw.py` doesn't find a Kaggle CLI / has no credentials.** That's
  expected if you haven't set up `~/.kaggle/kaggle.json` — the script detects this and
  falls back automatically to the `Heliosoph/Jigsaw-Toxic-Comments` Hugging Face
  mirror, which needs no login, just internet access to huggingface.co. You'll see
  `"Kaggle CLI not available — using Hugging Face mirror."` printed; that's not an
  error.
- **A script prints `[SKIP] <model>: missing models/...`.** This is the roster check
  (`src/roster.py`) working as intended, not a bug — that model's local weights don't
  exist in your checkout (they're gitignored). The message names the exact command to
  run to produce them (a `download_*`/`train_*` script, or a Colab notebook for the
  two models Tom trained on GPU: `distilbert-student`, `distilgpt2-d1-ft`). Run
  `python -c "import sys; sys.path.insert(0,'src'); import roster; roster.print_roster_status()"`
  any time to see the full picture at once.
- **`ModuleNotFoundError` for something in `src/`.** Run scripts as
  `python scripts/<name>.py` from the repo root (not `cd scripts && python <name>.py`).
  Every script inserts `<repo_root>/src` onto `sys.path` itself at import time, but
  that insertion is relative to the script's own location, so it only resolves
  correctly when the file layout is intact — no manual `PYTHONPATH` setup needed.
- **Garbled output or a crash the moment a script prints a path.** You forgot
  `PYTHONIOENCODING=utf-8` — see the note in §2. This repo's own path contains
  non-ASCII characters, which the default Windows console encoding (cp1252) can't
  print.
- **A run feels stuck / is taking far longer than expected.** See the runtime note in
  §2 — the DistilGPT-2 decoders are the usual cause on CPU. Cut `--n`/`--eval-n`, pass
  `--quick` where the script supports it, or move that piece of work to Colab (the
  project already does this for the three GPU-heavy training runs; see `notebooks/09`,
  `10`, `15`).

---

## 7. Repository map

| Path | Contents |
|---|---|
| `report/report.tex` | **The submission report.** |
| `report/report_draft.md` | Working draft — every experiment, with full protocol and numbers |
| `report/PRESENTATION_OUTLINE.md` | Slide-by-slide outline for the ≤6-minute video |
| `report/figures/` | All generated figures |
| `report/Words_That_Harm_Slides.pptx` | Slide deck |
| `docs/DEVELOPER_GUIDE.md` | Code architecture and how to run everything |
| `docs/HANDOFF.md` | Distillation / decoder fine-tune handoff notes |
| `src/`, `scripts/`, `notebooks/` | See §4 |
| `Project_Proposal.pdf` | The original approved proposal |
| `data/`, `models/` | Gitignored — created by the download/training scripts |

## 7. Work split

- **Omer** — encoder side: toxic-bert, the `bert-base` classifier/prompt pair,
  TF-IDF+SVM, the fairness probe, perturbations, the masking intervention.
- **Tom** — decoder side: DistilGPT-2 prompting and fine-tune, knowledge
  distillation, FPR-regularized fine-tuning, report and slides.
