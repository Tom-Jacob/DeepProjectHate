"""Project paths and constants."""
from pathlib import Path

SEED = 211

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
