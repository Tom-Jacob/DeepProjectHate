"""Prefetch distilgpt2 from Hugging Face and save a local copy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paths import DISTILGPT2_MODEL_ID, LOCAL_DISTILGPT2_DIR  # noqa: E402

from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402


def main() -> None:
    print(f"Downloading {DISTILGPT2_MODEL_ID} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(DISTILGPT2_MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(DISTILGPT2_MODEL_ID)

    LOCAL_DISTILGPT2_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_DISTILGPT2_DIR)
    model.save_pretrained(LOCAL_DISTILGPT2_DIR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved to {LOCAL_DISTILGPT2_DIR}")
    print(f"Parameters: {n_params:,}")


if __name__ == "__main__":
    main()
