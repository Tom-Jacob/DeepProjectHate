# Words That Harm — Architectures and Fairness in Toxic-Comment Detection

Technion **EE046211 Deep Learning** course project. Authors: **Omer Ramon, Tom Jacob**.

A controlled **audit** of toxic-comment detectors on the Jigsaw 2018 dataset. Rather
than chasing accuracy, we compare five model configurations **on the same data** and
stress-test each for bias, robustness, and failure mechanism.

| Architecture | Model | Role |
|---|---|---|
| Encoder classifier | `unitary/toxic-bert` (BERT, 109M) | in-domain specialist |
| Encoder classifier | `bert-base-uncased`, fine-tuned by us (BERT, 109M) | generic specialization baseline |
| Encoder prompting | `bert-base-uncased`, zero-shot masked-LM fill-blank | same checkpoint as above, no training |
| Non-neural | TF-IDF + linear SVM (we train it) | fair "floor" |
| Decoder (prompting) | DistilGPT-2 (82M), D1 yes/no + D2 fill-blank | generative LM as zero-shot classifier |

`bert-base-classifier` and `bert-base-mlm-prompt` are the SAME pretrained checkpoint
used two ways — a trained classification head vs. zero-shot prompting — giving a
controlled answer to "dedicated classifier vs. LM prompting" that isn't confounded by
architecture family (unlike toxic-bert vs. DistilGPT-2). hateBERT, a true
toxic-bert→DistilBERT distillation, few-shot decoder prompting, and calibration (ECE)
were considered but are explicit future work (see `report/report_draft.md` §5.4) —
not implemented, due to no usable local GPU.

Every model is reduced to a single `P(toxic)` so all evaluations are
architecture-agnostic. We measure accuracy, latency, **per-identity fairness**,
**robustness to obfuscation**, and run a **masking intervention** to explain *why*
models over-flag identity mentions.

## Headline findings
- **Toxicity detection is substantially lexical, but not purely so** — the linear
  SVM reaches ~93% of toxic-bert's F1 (~94% of its PR-AUC) at ~3300× lower latency,
  but toxic-bert keeps a real edge at full scale, not just a latency trade-off.
- **The encoder's real edge is robustness, and it isn't free with the architecture**
  — toxic-bert resists leetspeak (evasion 0.29) far better than the SVM (0.75), but
  our own `bert-base-classifier` — same architecture, same size — is nearly as
  brittle (0.74) as the SVM. The robustness comes from toxic-bert's specific
  fine-tuning, not from "being a BERT".
- **Discriminative heads ≫ generative prompting at small scale** — on the exact same
  `bert-base-uncased` checkpoint, the trained classifier head dominates zero-shot
  masked-LM prompting (F1 0.68 vs. 0.00); DistilGPT-2 zero-shot prompting is
  near-chance too, though BERT's zero-shot prompt still ranks far better
  (ROC-AUC 0.72 vs. ≈0.4) than GPT-2's.
- **Identity bias is data-driven but not uniform in mechanism** — present in every
  classifier, including our own generic one (which is *less* fair than the
  specialized toxic-bert). Masking the identity token erases the bias for
  toxic-bert and the SVM (a lexical shortcut) but *increases* it for our generic
  classifier — the same data-driven bias shows up differently model to model.

## Documentation map
| Document | What it is |
|---|---|
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Architecture, the `predict_toxic_proba` contract, and how to run everything **programmatically** and via CLI |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Build log: environment gotchas, per-phase results, command reference |
| [`report/report_draft.md`](report/report_draft.md) | The sole live report draft (≤8 pages) — numbers, conclusions, references, ethics template. An earlier `.docx` skeleton is archived in `report/archive/` and is not current. |
| [`CLAUDE.md`](CLAUDE.md) | Project context (goal, hypotheses, verified env) |
| `report/figures/` | All generated figures for the report |

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
python scripts/download_jigsaw.py            # Jigsaw CSVs (Kaggle or HF-mirror fallback)
python scripts/download_toxic_bert.py        # unitary/toxic-bert weights
python scripts/make_splits.py                # train/val/test split indices
python scripts/train_bert_base_classifier.py # fine-tune bert-base-classifier (CPU, ~1.5-2h)
```
The DistilGPT-2, TF-IDF+SVM, and `bert-base-mlm-prompt` models need no separate
download/training step — they're obtained or fit automatically on first use.

## Run the audit

```bash
python scripts/run_model_comparison.py --quick        # accuracy / latency / params, all 5 models
python scripts/run_fairness.py                        # per-identity FPR
python scripts/run_robustness.py --quick              # perturbation evasion
python scripts/run_masking.py                         # masking intervention (why)
python scripts/run_agreement.py --n 150               # encoder (toxic-bert) vs decoder, multi-label
python scripts/run_bertbase_agreement.py               # bert-base classifier vs mlm-prompt, same checkpoint
# figures:
python scripts/plot_model_comparison.py --split test
python scripts/plot_fairness_robustness.py
```
Add the decoder to any comparison: `--models toxic-bert tfidf-svm distilgpt2-d1`.
Drop `--quick` (or raise `--eval-n`) for full-size runs. See the developer guide for
the Python API.

## Repository layout
```
src/            model_registry (hub), encoder, decoder, baselines, fairness,
                perturb, intervention, evaluate, plots, data, splits, device, paths
scripts/        download_*, make_splits, train_bert_base_classifier,
                run_* (comparison/fairness/robustness/masking/agreement/
                bertbase_agreement), plot_*
notebooks/      01 EDA · 02 decoder · 03 formal eval · 04 encoder/decoder
docs/           DEVELOPER_GUIDE.md, PROGRESS.md
report/         report_draft.md, figures/, archive/ (superseded drafts)
data/           raw/ (gitignored) + processed/ (gitignored results JSON)
```

## Licenses
- Jigsaw data: CC0 (comment text under Wikipedia CC-SA-3.0)
- `unitary/toxic-bert`: Apache 2.0 · DistilGPT-2: Apache 2.0
