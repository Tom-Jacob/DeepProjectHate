# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

Technion **EE046211 Deep Learning** course project. Authors: **Omer Ramon, Tom Jacob**.

A controlled **audit** of toxic-comment detectors on the Jigsaw 2018 dataset. Rather
than chasing accuracy, we compare three architectures **on the same data** and
stress-test each for bias, robustness, and failure mechanism.

| Architecture | Model | Role |
|---|---|---|
| Encoder classifier | `unitary/toxic-bert` (BERT, 109M) | in-domain specialist |
| Non-neural | TF-IDF + linear SVM (we train it) | fair "floor" |
| Decoder (prompting) | DistilGPT-2 (82M) | generative LM as zero-shot classifier |
| Distilled student | DistilBERT (soft KD from toxic-bert, **6 labels**) | compression / H2 |

Every model is reduced to a single `P(toxic)` so all evaluations are
architecture-agnostic. We measure accuracy, latency, **per-identity fairness**,
**robustness to obfuscation**, run a **masking intervention**, and **distill**
toxic-bert → DistilBERT to test whether compression preserves fairness.

## Headline findings
- **Toxicity detection is largely lexical** — the linear SVM matches toxic-bert's F1
  and beats its PR-AUC at ~1700× lower latency.
- **The encoder's real edge is robustness**, not accuracy (leetspeak evasion 0.28 vs
  0.71 for the SVM).
- **Discriminative heads ≫ generative prompting at small scale** — DistilGPT-2
  zero-shot prompting is near chance.
- **Identity bias is data-driven** — present even in the linear model; masking the
  identity token drops the false-positive rate to zero (a lexical shortcut).

## Documentation map
| Document | What it is |
|---|---|
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Architecture, the `predict_toxic_proba` contract, and how to run everything **programmatically** and via CLI |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Build log: environment gotchas, per-phase results, command reference |
| [`docs/SYNC_STATUS.md`](docs/SYNC_STATUS.md) | Git sync check (local `main` vs `origin/main`) |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | What was finished after partner sync + remaining team tasks |
| [`report/report_draft.md`](report/report_draft.md) | **Source of truth** for the ≤8-page report (numbers + narrative) |
| [`report/Words_That_Harm_Report.docx`](report/Words_That_Harm_Report.docx) | Word export of the draft (figures embedded) |
| [`report/PRESENTATION_OUTLINE.md`](report/PRESENTATION_OUTLINE.md) | ~10-slide / ≤6-min video outline |
| [`report/Words_That_Harm_Slides.pptx`](report/Words_That_Harm_Slides.pptx) | PowerPoint deck (figures embedded) |
| [`CLAUDE.md`](CLAUDE.md) | Project context (goal, hypotheses, verified env) |
| `report/figures/` | All generated figures for the report |
| `report/archive/` | Superseded mockups (do not submit) |

## Submission checklist

1. `git fetch` / stay on `main` (see [`docs/SYNC_STATUS.md`](docs/SYNC_STATUS.md)).
2. Re-run audits at submission size (drop `--quick` / raise `--eval-n`) and refresh `report/figures/`.
3. Update numbers in [`report/report_draft.md`](report/report_draft.md); re-export Word if needed.
4. **Write ethics §6.3 yourselves** (no LLM) — prompts are in the draft.
5. Build slides from [`report/PRESENTATION_OUTLINE.md`](report/PRESENTATION_OUTLINE.md); record ≤6 min video.
6. Do **not** submit files under `report/archive/`.

## Setup

```bash
conda activate deep_learn
pip install -r requirements-project.txt
```

Two machine-specific notes (see `docs/PROGRESS.md` for details):
- **CPU-only on some machines:** an RTX 50-series (Blackwell) GPU can't run
  `torch 2.3/cu121`; `src/device.py` auto-falls-back to CPU. GPU training runs on Colab.
- **Windows + non-ASCII path:** set `PYTHONIOENCODING=utf-8` before running scripts.

### Download data & models (one-time)
```bash
python scripts/download_jigsaw.py        # Jigsaw CSVs (Kaggle or HF-mirror fallback)
python scripts/download_toxic_bert.py    # unitary/toxic-bert weights
python scripts/make_splits.py            # train/val/test split indices
```
The DistilGPT-2 and TF-IDF+SVM models are obtained/trained automatically on first use.

## Run the audit

```bash
python scripts/run_model_comparison.py --quick        # accuracy / latency / params
python scripts/run_fairness.py                        # per-identity FPR
python scripts/run_robustness.py --quick              # perturbation evasion
python scripts/run_masking.py                         # masking intervention (why)
python scripts/train_distill.py --epochs 2            # soft KD 6-label (Colab GPU preferred)
python scripts/train_decoder_ft.py --max-train-samples 8000  # D1 yes/no FT (Colab)
python scripts/run_agreement.py --n 150               # encoder vs decoder
# figures:
python scripts/plot_model_comparison.py --split test
python scripts/plot_fairness_robustness.py
```
Add the decoder or student: `--models toxic-bert tfidf-svm distilgpt2-d1 distilgpt2-d1-ft distilbert-student`.

### Notebooks
| Notebook | Role |
|---|---|
| 01 | EDA + encoder smoke (keep) |
| 02 | DistilGPT-2 decoder smoke (keep) |
| 03 | **Fair comparison** — accuracy / latency / params (cache-first) |
| 04 | **Fair stress tests** — fairness + robustness + masking (cache-first) |
| 09 | **Colab distillation** (soft KD, 6 labels) |
| 10 | **Colab decoder fine-tune** (D1 yes/no, binary toxic) |
| 11 | **SVM rigorous audit** — metrics, ablations, fairness, robustness, masking |
| 12 | **Encoder rigorous audit** — toxic-bert · distilbert-tox · distilbert-student |
| 13 | **Decoder rigorous audit** — distilgpt2-d1 · d2 · d1-ft (skips FT if missing) |
| 14 | **Soft-F1 distillation** — beta=0.3/0.5 vs baseline; accuracy + fairness |
| 15 | **Colab full eval** — full test (63k enc, 5k dec), fairness+perturb, agreement D1-ft, bidirectional robustness |
| `archive/03–08` | Superseded exploration/audit wrappers (reference only) |

Roster in 03–04: `toxic-bert`, `tfidf-svm`, `distilgpt2-d1`, `distilgpt2-d1-ft`, `distilbert-student` (skips missing weights with a clear message). Prefer cached `data/processed/*.json`; optional `LIVE_RERUN` for a small live pass.

Distillation: Colab notebook 09 → `models/distilbert-student/`.  
Decoder FT: Colab notebook 10 → `models/distilgpt2-d1-ft/` (keep zero-shot `distilgpt2-d1` in tables).  
SVM deep-dive: notebook 11 → `data/processed/svm_rigorous_audit.json`.  
Encoder deep-dive: notebook 12 → `data/processed/encoder_rigorous_audit.json`.  
Decoder deep-dive: notebook 13 → `data/processed/decoder_rigorous_audit.json`.

## Repository layout
```
src/            model_registry, roster, distill, encoder, decoder, baselines, fairness,
                perturb, intervention, evaluate, plots, data, splits, device, paths
scripts/        download_*, make_splits, train_distill, train_decoder_ft, run_*, plot_*, export_report_docx
notebooks/      01–02 exploration · 03–04 fair comparison · 09–10 Colab · 11–13 rigorous audits
                archive/ (old 03–08)
docs/           DEVELOPER_GUIDE, PROGRESS, HANDOFF, SYNC_STATUS
report/         report_draft.md, Words_That_Harm_Report.docx, Slides.pptx, figures/
data/           raw/ (gitignored) + processed/ (gitignored results JSON)
```

## Licenses
- Jigsaw data: CC0 (comment text under Wikipedia CC-SA-3.0)
- `unitary/toxic-bert`: Apache 2.0 · DistilGPT-2: Apache 2.0
