"""Download Jigsaw 2018 data into data/raw/jigsaw/.

Tries Kaggle CLI first; falls back to Hugging Face mirror (no Kaggle account needed):
Heliosoph/Jigsaw-Toxic-Comments
"""
import gzip
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import RAW_JIGSAW_DIR, TEST_CSV, TEST_LABELS_CSV, TRAIN_CSV  # noqa: E402

HF_REPO = "Heliosoph/Jigsaw-Toxic-Comments"
HF_FILES = {
    "train.csv.gz": TRAIN_CSV,
    "test.csv.gz": TEST_CSV,
    "test_labels.csv.gz": TEST_LABELS_CSV,
}


def verify_files() -> bool:
    return all(p.exists() for p in (TRAIN_CSV, TEST_CSV, TEST_LABELS_CSV))


def _extract_zips(directory: Path) -> None:
    for zip_path in list(directory.glob("*.zip")):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)


def _download_from_kaggle() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle CLI not available — using Hugging Face mirror.")
        return False

    RAW_JIGSAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading from Kaggle ...")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "jigsaw-toxic-comment-classification-challenge",
                "-p",
                str(RAW_JIGSAW_DIR),
            ],
            check=True,
        )
        _extract_zips(RAW_JIGSAW_DIR)
        return verify_files()
    except subprocess.CalledProcessError as exc:
        print(f"Kaggle download failed: {exc}")
        return False


def _download_from_huggingface() -> None:
    from huggingface_hub import hf_hub_download

    RAW_JIGSAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Hugging Face: {HF_REPO} ...")

    for gz_name, out_path in HF_FILES.items():
        if out_path.exists():
            print(f"  skip {out_path.name} (exists)")
            continue
        cached = hf_hub_download(repo_id=HF_REPO, filename=gz_name, repo_type="dataset")
        print(f"  decompress {gz_name} -> {out_path.name}")
        with gzip.open(cached, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main() -> None:
    if verify_files():
        print("Jigsaw CSVs already present.")
        return

    if not _download_from_kaggle():
        _download_from_huggingface()

    if not verify_files():
        raise FileNotFoundError(
            f"Expected train.csv, test.csv, test_labels.csv in {RAW_JIGSAW_DIR}. "
            "See README.md for manual setup."
        )
    print("Download complete.")


if __name__ == "__main__":
    main()
