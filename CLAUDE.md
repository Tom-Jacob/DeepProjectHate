# CLAUDE.md — Words That Harm (Toxic Comment Audit)

Deep Learning course project (Technion EE046211). Pairs project, 30% of grade.
This file is persistent context for AI sessions. Keep it short and current.

## What the project is

A **controlled audit** of toxic-comment detectors on the **Jigsaw 2018** dataset
(multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate).
It is NOT "fine-tune BERT, report F1" — the supervisor (Roy) explicitly wants a
model-auditing angle. Roster (all reduced to a single P(toxic)): `toxic-bert`,
`bert-base-classifier`, `bert-base-mlm-prompt`, `tfidf-svm`, `distilbert-student`
(KD from toxic-bert), `distilgpt2-d1`/`-d2` (zero-shot), `distilgpt2-d1-ft`
(fine-tuned). Axes of comparison:

- **Architecture:** encoder classifiers (toxic-bert, bert-base) vs decoder prompting (DistilGPT-2)
- **Specialization:** generic (bert-base, fine-tuned by us) vs toxicity-trained (toxic-bert)
- **Method, same checkpoint:** `bert-base-uncased` as a trained classifier vs. as a
  zero-shot masked-LM prompt ("This comment is [MASK]") — the controlled version of
  Roy's "dedicated classifier vs LM prompting" question
- **Compression:** soft-KD distillation (toxic-bert → DistilBERT student, `src/distill.py`) —
  matches the teacher's F1 at lower latency and *lower* identity FPR (H2 partially rejected)
- **Decoder supervision:** zero-shot DistilGPT-2 prompting vs. the same model fine-tuned
  on the D1 prompt (`src/decoder_finetune.py`) — fine-tuning recovers most of the gap
- **Fairness:** per-identity false-positive rate (FPR) on neutral identity-mention sentences
  (`src/fairness.py`: 368 sentences, 23 groups incl. neutral baselines, 3 length tiers)
- **Robustness:** perturbations (typos, leetspeak, spacing, casing, punctuation)
- **Intervention:** inference-time identity masking (all models) + training-time
  masking retrain (TF-IDF+SVM) + FPR-regularized second-stage fine-tune (`scripts/train_fpr_finetune.py`)

hateBERT, few-shot decoder prompting, and ECE calibration remain explicit
**future work** (see `report/report_draft.md` §4). Distillation and decoder
fine-tuning were originally future work too but are now done — see
`docs/HANDOFF.md` for what's done vs. pending.

Target hypotheses (pick 2-3 for the report): specialized encoders trade fairness for
accuracy; decoder/zero-shot prompting is prompt-sensitive / less calibrated than a
trained classifier head, even on the same checkpoint; identity bias is data-driven,
not architectural (see the masking intervention results); compression need not
amplify bias (distillation result).

## Work split

- **Omer** — encoder side: `unitary/toxic-bert`, `bert-base` (classifier + MLM
  prompt), TF-IDF+SVM, fairness FPR, perturbations, masking intervention.
- **Tom** — decoder side: DistilGPT-2 prompting + fine-tune, distillation
  (toxic-bert → DistilBERT student), FPR-regularized fine-tuning, report/slides.

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
- bert-base-toxic weights (our fine-tune): `models/bert-base-toxic/` — via
  `python scripts/train_bert_base_classifier.py` (trains on a ~12k-row subsample, CPU, ~1.5-2h).
- distilbert-student, distilgpt2-d1-ft weights: `models/distilbert-student/`,
  `models/distilgpt2-d1-ft/` — Tom's Colab GPU outputs (`notebooks/09_distillation_colab.ipynb`,
  `notebooks/10_decoder_finetune_colab.ipynb`); gitignored, not present unless you
  run those notebooks yourself. `python -c "import sys; sys.path.insert(0,'src'); import roster; roster.print_roster_status()"`
  reports what's actually available locally.
- Splits: `data/processed/split_indices.parquet` — via `python scripts/make_splits.py`
  (train 135,635 / val 23,936 / test 63,978; test = labeled Jigsaw test rows only).

## Common commands

```bash
conda activate deep_learn                       # or use the full python.exe path above
python scripts/make_splits.py                   # build split_indices.parquet
python scripts/train_bert_base_classifier.py    # one-time: fine-tune bert-base-classifier
python scripts/run_model_comparison.py --quick --models toxic-bert bert-base-classifier bert-base-mlm-prompt tfidf-svm
python scripts/run_fairness.py --models toxic-bert bert-base-classifier bert-base-mlm-prompt tfidf-svm
python scripts/run_robustness.py --quick        # perturbation evasion
python scripts/run_masking.py                   # masking intervention (why)
python scripts/run_agreement.py --n 150         # encoder (toxic-bert) vs decoder (multi-label)
python scripts/run_bertbase_agreement.py        # bert-base classifier vs mlm-prompt (same checkpoint)
```
Omit `--models` to use all of `MODEL_SPECS` (now includes `distilbert-student`,
which needs Tom's Colab weights — pass an explicit `--models` list to skip it if
you don't have `models/distilbert-student/` locally).

## Repo map

- `src/model_registry.py` — the hub: `ModelSpec`, `MODEL_SPECS`/`DECODER_SPECS`/`ALL_SPECS`,
  `load_scorer`, `HFSeqClassScorer`, `DecoderToxicScorer`, `BertBaseMlmPromptScorer`.
  Every model reduces to `predict_toxic_proba(texts) -> P(toxic)`.
- `src/roster.py` — `FAIR_ROSTER`, `model_ready`/`print_roster_status`: checks which
  models have local weights available before you try to score with them.
- `src/baselines.py` — `TfidfSvmScorer` (non-neural floor).
- `src/encoder.py` — `ToxicBertEncoder`, batch sigmoid multi-label inference (accepts `device=`).
- `src/distill.py`, `src/decoder_finetune.py` — Tom's soft-KD distillation
  (toxic-bert → DistilBERT student) and DistilGPT-2 D1 fine-tune; trained on Colab
  GPU via `notebooks/09_distillation_colab.ipynb` / `10_decoder_finetune_colab.ipynb`.
- `src/fairness.py`, `src/perturb.py`, `src/intervention.py` — fairness probe, perturbations, masking.
- `src/evaluate.py` — multi-label metrics (per-label F1/PR-AUC, macro/micro, enc/dec agreement).
- `src/data.py`, `src/splits.py`, `src/paths.py` — loading, splits, constants.
- `src/decoder.py`, `src/plots.py`, `src/eval_sample.py` — decoder side, plotting, sampling.
- `scripts/` — CLI helpers: download, splits, `train_bert_base_classifier`,
  `train_distill`/`train_decoder_ft`/`train_fpr_finetune`, `run_model_comparison`/
  `run_fairness`/`run_robustness`/`run_masking`/`run_agreement`/`run_bertbase_agreement`,
  `plot_*`, `export_report_docx.py`.
- `notebooks/` — 01 EDA, 02 decoder smoke, 03-04 fair comparison/stress tests,
  09 distillation (Colab), 10 decoder fine-tune (Colab), 11-13 rigorous per-family
  audits, 14 soft-F1 distillation, 15 full eval (Colab), 16 FPR fine-tuning;
  `notebooks/archive/` has superseded early notebooks.

## Conventions

- Git: Tom also works on `main`. Commit messages are plain and human-voiced (no
  AI co-author trailers, no phase codes) — this is a graded submission, keep it that way.
- Use COMMENTS (course grades on it). Keep a very detailed README (25% of grade).
- Report structure: intro / method / experiments+results / conclusion + references + ethics template.
- The report's ethics reflection (independent, non-LLM section) must be written by
  Omer/Tom themselves — never draft that section.
