# CLAUDE.md — Words That Harm (Toxic Comment Audit)

Deep Learning course project (Technion EE046211). Pairs project, 30% of grade.
This file is persistent context for AI sessions. Keep it short and current.

## What the project is

A **controlled audit** of toxic-comment detectors on the **Jigsaw 2018** dataset
(multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate).
It is NOT "fine-tune BERT, report F1" — the supervisor (Roy) explicitly wants a
model-auditing angle. Axes of comparison:

- **Architecture:** encoder classifier (toxic-bert) vs decoder prompting (DistilGPT-2)
- **Specialization:** generic (DistilBERT) vs toxicity-trained (toxic-bert) vs cross-domain (hateBERT)
- **Compression:** distillation (toxic-bert -> DistilBERT)
- **Fairness:** per-identity false-positive rate (FPR) on neutral identity-mention sentences
- **Robustness:** perturbations (typos, leetspeak, template polarity swaps)
- **Intervention:** one training run masking identity terms, measure delta-FPR

Target hypotheses (pick 2-3 for the report): specialized encoders trade fairness for
accuracy; decoder prompting is prompt-sensitive / less calibrated; distillation
preserves accuracy but amplifies identity FPR.

## Work split

- **Omer** — encoder side: `unitary/toxic-bert`, fairness FPR, perturbations, distillation teacher.
- **Tom** — decoder side: DistilGPT-2 prompting.

## Environment (verified 2026-07)

- Conda env: **`deep_learn`** — Python 3.8, `torch 2.3.0+cu121`.
  Interpreter: `C:\Users\omerr\miniconda3\envs\deep_learn\python.exe`
- Project deps installed on top: transformers 4.46.3, datasets 3.1.0, huggingface_hub 0.36,
  pyarrow 17, scikit-learn, kaggle. (`pip install -r requirements-project.txt`)
- **GPU CAVEAT:** the machine has an **RTX 5060 Laptop (Blackwell, sm_120)**. torch 2.3.0+cu121
  only supports up to sm_90, so **CUDA kernels fail** ("no kernel image available"). Everything
  runs on **CPU**. `src/device.py` now probes CUDA and auto-falls-back to CPU; force it with
  env `DEEPHATE_FORCE_CPU=1`. For GPU-heavy work (distillation training) use Colab/cloud, or a
  newer Python env with a cu128 torch build — do NOT casually upgrade the shared `deep_learn` env.
- Windows console is cp1252 and the repo path contains Hebrew chars. Always set
  `PYTHONIOENCODING=utf-8` when running scripts, or `print()` of paths crashes.

## Data & models (local, gitignored)

- Jigsaw CSVs: `data/raw/jigsaw/{train,test,test_labels}.csv` — via `python scripts/download_jigsaw.py`
  (Kaggle if configured, else Heliosoph/Jigsaw-Toxic-Comments HF mirror — no account needed).
- toxic-bert weights: `models/toxic-bert/` — via `python scripts/download_toxic_bert.py` (~418 MB, 109M params).
- Splits: `data/processed/split_indices.parquet` — via `python scripts/make_splits.py`
  (train 135,635 / val 23,936 / test 63,978; test = labeled Jigsaw test rows only).

## Common commands

```bash
conda activate deep_learn                       # or use the full python.exe path above
python scripts/make_splits.py                   # build split_indices.parquet
python scripts/run_formal_eval.py --quick       # toxic-bert eval, 500/split -> benchmark_results.json
python scripts/run_formal_eval.py               # full eval (slow on CPU)
```

## Repo map

- `src/encoder.py` — `ToxicBertEncoder`, batch sigmoid multi-label inference (accepts `device=`).
- `src/evaluate.py` — multi-label metrics (per-label F1/PR-AUC, macro/micro, enc/dec agreement).
- `src/data.py`, `src/splits.py`, `src/paths.py` — loading, splits, constants.
- `src/decoder.py`, `src/plots.py`, `src/eval_sample.py` — decoder side, plotting, sampling.
- `scripts/` — CLI helpers (download, splits, formal eval, join scores).
- `notebooks/` — 01 EDA+smoke, 02 decoder smoke, 03 formal eval, 04 encoder/decoder comparison.

## Conventions

- Git: local-only for now (Omer reviews & pushes; do not push). Tom also works on `main`.
- Use COMMENTS (course grades on it). Keep a very detailed README (25% of grade).
- Report structure: intro / method / experiments+results / conclusion + references + ethics template.
