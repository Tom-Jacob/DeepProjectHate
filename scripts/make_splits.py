#!/usr/bin/env python
"""Create train/val/test index file for formal evaluation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paths import SPLIT_INDICES_PATH
from splits import save_split_indices, split_counts


def main() -> None:
    df = save_split_indices()
    print(f"Saved {len(df):,} row ids to {SPLIT_INDICES_PATH}")
    print(split_counts().to_string(index=False))


if __name__ == "__main__":
    main()
