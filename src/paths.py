"""Project paths and constants."""
from pathlib import Path
from typing import Optional

SEED = 211

# Toggle in notebooks for faster iteration (20/20 vs 50/50 balanced eval).
QUICK_MODE = True
EVAL_N_EACH_FULL = 50
EVAL_N_EACH_QUICK = 20


def eval_n_each(quick_mode: Optional[bool] = None) -> int:
    if quick_mode is None:
        quick_mode = QUICK_MODE
    return EVAL_N_EACH_QUICK if quick_mode else EVAL_N_EACH_FULL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_JIGSAW_DIR = DATA_DIR / "raw" / "jigsaw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TRAIN_CSV = RAW_JIGSAW_DIR / "train.csv"
TEST_CSV = RAW_JIGSAW_DIR / "test.csv"
TEST_LABELS_CSV = RAW_JIGSAW_DIR / "test_labels.csv"

TOXIC_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_ID = "unitary/toxic-bert"
LOCAL_MODEL_DIR = MODELS_DIR / "toxic-bert"

DISTILGPT2_MODEL_ID = "distilgpt2"
LOCAL_DISTILGPT2_DIR = MODELS_DIR / "distilgpt2"
LOCAL_DISTILGPT2_D1_FT_DIR = MODELS_DIR / "distilgpt2-d1-ft"
DECODER_FT_RESULTS_PATH = PROCESSED_DIR / "decoder_ft_results.json"

STUDENT_INIT_ID = "distilbert-base-uncased"
LOCAL_DISTILBERT_STUDENT_DIR = MODELS_DIR / "distilbert-student"
DISTILL_RESULTS_PATH = PROCESSED_DIR / "distill_results.json"

BERT_BASE_MODEL_ID = "bert-base-uncased"
LOCAL_BERT_BASE_DIR = MODELS_DIR / "bert-base-toxic"

DECODER_SMOKE_SCORES_PATH = PROCESSED_DIR / "decoder_smoke_scores.parquet"
ENCODER_SMOKE_SCORES_PATH = PROCESSED_DIR / "encoder_smoke_scores.parquet"
JOINED_SCORES_PATH = PROCESSED_DIR / "joined_smoke_scores.parquet"
SPLIT_INDICES_PATH = PROCESSED_DIR / "split_indices.parquet"
BENCHMARK_RESULTS_PATH = PROCESSED_DIR / "benchmark_results.json"

# Train/val fraction from train.csv (remainder is val). Test = labeled Jigsaw test split.
TRAIN_FRAC = 0.85
VAL_FRAC = 0.15

# Cap per split when --quick is passed to run_formal_eval.py
FORMAL_EVAL_QUICK_N = 500
