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

Every model is reduced to a single `P(toxic)` so all evaluations are
architecture-agnostic. We measure accuracy, latency, **per-identity fairness**,
**robustness to obfuscation**, and run a **masking intervention** to explain *why*
models over-flag identity mentions.

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
| [`report/report_draft.md`](report/report_draft.md) | The ≤8-page report draft (submission mockup) with final numbers, conclusions, references, ethics template |
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
python scripts/run_agreement.py --n 150               # encoder vs decoder
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
scripts/        download_*, make_splits, run_* (comparison/fairness/robustness/
                masking/agreement), plot_*
notebooks/      01 EDA · 02 decoder · 03 formal eval · 04 encoder/decoder
docs/           DEVELOPER_GUIDE.md, PROGRESS.md
report/         report_draft.md, figures/
data/           raw/ (gitignored) + processed/ (gitignored results JSON)
```

## Licenses
- Jigsaw data: CC0 (comment text under Wikipedia CC-SA-3.0)
- `unitary/toxic-bert`: Apache 2.0 · DistilGPT-2: Apache 2.0
