"""Prefetch unitary/toxic-bert from Hugging Face and save a local copy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paths import LOCAL_MODEL_DIR, MODEL_ID  # noqa: E402

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


def main() -> None:
    print(f"Downloading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved to {LOCAL_MODEL_DIR}")
    print(f"Parameters: {n_params:,}")


if __name__ == "__main__":
    main()
