# Words That Harm — Toxic Comment Audit

EE046211 / ECE046211 Deep Learning course project. Phase 1: Jigsaw 2018 data exploration and `unitary/toxic-bert` smoke tests.

## Setup

```bash
conda activate deep_learn
cd Project
pip install -r requirements-project.txt
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

If missing, install from https://pytorch.org/get-started/locally/

## Project layout

```
Project/
├── data/raw/jigsaw/     # Kaggle CSVs (not in git)
├── data/processed/      # splits (Phase 2)
├── models/toxic-bert/   # optional local model copy (not in git)
├── notebooks/           # EDA and experiments
├── src/                 # shared loaders and helpers
└── requirements-project.txt
```

## Download Jigsaw 2018 data (Kaggle)

The Hugging Face dataset `google/jigsaw_toxicity_pred` requires local Kaggle files.

### One-time Kaggle API setup

1. Create an account at https://www.kaggle.com
2. Join the competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
3. Go to Account → Settings → **Create New API Token** (downloads `kaggle.json`)
4. Place the file at:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
5. Install CLI: `pip install kaggle`

### Download into this project

```bash
cd Project
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/raw/jigsaw
```

If you get a zip file, extract it into `data/raw/jigsaw/`. Kaggle often nests CSVs inside inner zips (`train.csv.zip`, etc.) — extract those too until you have the three `.csv` files.

| File | Description |
|------|-------------|
| `train.csv` | ~159k labeled Wikipedia talk comments |
| `test.csv` | Test comment texts |
| `test_labels.csv` | Labels for labeled test subset |

Or use the helper script (uses Kaggle if configured, otherwise Hugging Face mirror):

```bash
python scripts/download_jigsaw.py
```

**No Kaggle account?** The script automatically falls back to [Heliosoph/Jigsaw-Toxic-Comments](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments) on Hugging Face.

## Download toxic-bert weights

```bash
python scripts/download_toxic_bert.py
```

This caches `unitary/toxic-bert` from Hugging Face and saves a copy to `models/toxic-bert/`.

Model: https://huggingface.co/unitary/toxic-bert (Apache 2.0)

## Run the EDA notebook

```bash
jupyter notebook notebooks/01_eda_and_smoke_tests.ipynb
```

Run all cells top-to-bottom after data and model are downloaded.

## Licenses

- Jigsaw data: CC0 (comment text governed by Wikipedia CC-SA-3.0)
- toxic-bert: Apache 2.0
