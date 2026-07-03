#!/usr/bin/env python
"""Join encoder + decoder smoke parquets on comment_text."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from join_scores import join_encoder_decoder
from paths import (
    DECODER_SMOKE_SCORES_PATH,
    ENCODER_SMOKE_SCORES_PATH,
    JOINED_SCORES_PATH,
    TOXIC_LABELS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge encoder and decoder smoke score parquets")
    parser.add_argument(
        "--score-encoder",
        action="store_true",
        help="Run score_encoder_smoke.py if encoder parquet is missing",
    )
    args = parser.parse_args()

    if not DECODER_SMOKE_SCORES_PATH.exists():
        print(f"Missing {DECODER_SMOKE_SCORES_PATH}\nRun notebook 02 (§13) first.")
        sys.exit(1)

    if not ENCODER_SMOKE_SCORES_PATH.exists():
        if args.score_encoder:
            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "score_encoder_smoke.py"), "--from-decoder"])
        else:
            print(
                f"Missing {ENCODER_SMOKE_SCORES_PATH}\n"
                "Run: python scripts/score_encoder_smoke.py --from-decoder\n"
                "Or re-run with --score-encoder"
            )
            sys.exit(1)

    enc = pd.read_parquet(ENCODER_SMOKE_SCORES_PATH)
    dec = pd.read_parquet(DECODER_SMOKE_SCORES_PATH)
    joined = join_encoder_decoder(enc, dec)

    if len(joined) == 0:
        print("No overlapping comment_text rows between encoder and decoder parquets.")
        sys.exit(1)

    JOINED_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(JOINED_SCORES_PATH, index=False)
    print(f"Joined {len(joined)} rows → {JOINED_SCORES_PATH}")
    print(f"  Gold labels: {TOXIC_LABELS}")


if __name__ == "__main__":
    main()
